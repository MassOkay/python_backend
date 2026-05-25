# evaluate.py
# JSQuADを使って3種のチャンキング手法を Precision / Recall / F1 で比較評価する

import json
import re
import os
import time
import urllib.request
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv

load_dotenv()  # .envを読み込む

# ============================================================
# 設定
# ============================================================
MODEL_NAME = 'sonoisa/sentence-bert-base-ja-mean-tokens-v2'
DIMENSION = 768
JSQUAD_URL = 'https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jsquad-v1.3/valid-v1.3.json'
JSQUAD_CACHE = 'data/jsquad_valid.json'
EVAL_DATA_DIR = 'data/eval'

# 評価するKの値（Top-K検索）
K_VALUES = [1, 3, 5]

# 評価に使うQA数の上限（Noneで全件）
MAX_QA = 500

# Fixed-size パラメータ
FIXED_CHUNK_SIZE = 100
FIXED_CHUNK_OVERLAP = 20

# LLM パラメータ（使う場合）
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_RETRY_LIMIT = 3
LLM_RETRY_WAIT = 5

# ============================================================
# チャンキング手法（main.pyと同じ実装）
# ============================================================

def split_fixed_size(text: str) -> List[str]:
    """① Fixed-size: 固定文字数でスライディングウィンドウ分割"""
    chunks, start = [], 0
    text = text.strip()
    while start < len(text):
        chunk = text[start:start + FIXED_CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += FIXED_CHUNK_SIZE - FIXED_CHUNK_OVERLAP
    return chunks

def split_into_sentences(text: str) -> List[str]:
    """② Sentence-based: 句点・改行で文単位に分割"""
    sentences = re.split(r'(?<=[。！？.])\s*|\n+', text)
    return [s.strip() for s in sentences if s.strip()]

def split_by_llm(text: str) -> List[str]:
    """③ LLM-based: Claude APIでトピック境界を検出して分割"""
    try:
        import anthropic
    except ImportError:
        print("  [警告] anthropicパッケージ未インストール。sentence fallback.")
        return split_into_sentences(text)

    if not ANTHROPIC_API_KEY:
        print("  [警告] ANTHROPIC_API_KEY未設定。sentence fallback.")
        return split_into_sentences(text)

    prompt = f"""以下の日本語テキストを読み、話題・トピックが切り替わる箇所に <boundary> タグを挿入してください。

ルール：
- 意味が完結するかたまりになるよう分割すること
- タグはトピック転換点にのみ挿入し、不要な分割は避けること
- テキストの内容は一切変更しないこと
- <boundary> タグ以外の余計な説明・コメントは不要

テキスト:
{text}"""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    for attempt in range(LLM_RETRY_LIMIT):
        try:
            message = client.messages.create(
                model=LLM_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_chunks = message.content[0].text.split("<boundary>")
            chunks = [c.strip() for c in raw_chunks if c.strip()]
            return chunks if chunks else split_into_sentences(text)
        except anthropic.RateLimitError:
            if attempt < LLM_RETRY_LIMIT - 1:
                print(f"  Rate limit. {LLM_RETRY_WAIT}s待機... ({attempt+1}/{LLM_RETRY_LIMIT})")
                time.sleep(LLM_RETRY_WAIT)
            else:
                print("  Rate limit超過。sentence fallback.")
                return split_into_sentences(text)
        except Exception as e:
            print(f"  API error: {e}. sentence fallback.")
            return split_into_sentences(text)

CHUNKING_METHODS = {
    "fixed":    split_fixed_size,
    "sentence": split_into_sentences,
    "llm":      split_by_llm,
}

# ============================================================
# JSQuADのロード
# ============================================================

def load_jsquad() -> Dict:
    """JSQuADをキャッシュから読む、なければダウンロード"""
    os.makedirs('data', exist_ok=True)
    if os.path.exists(JSQUAD_CACHE):
        print(f"JSQuADキャッシュ読み込み: {JSQUAD_CACHE}")
        with open(JSQUAD_CACHE, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f"JSQuADをダウンロード中: {JSQUAD_URL}")
    req = urllib.request.Request(JSQUAD_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    with open(JSQUAD_CACHE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print("ダウンロード完了・キャッシュ保存")
    return data

def extract_qa_pairs(data: Dict, max_qa: int = None) -> List[Dict]:
    """
    JSQuADから評価用QAペアを抽出する

    Returns:
        [{"question": str, "context_id": int, "context": str, "answer": str}, ...]
    """
    qa_pairs = []
    context_id = 0

    for article in data['data']:
        for para in article['paragraphs']:
            # [SEP] タグを除去してクリーンなコンテキストに
            context = para['context'].replace('[SEP]', '').strip()
            ctx_id = context_id
            context_id += 1

            for qa in para['qas']:
                if not qa['answers']:
                    continue
                qa_pairs.append({
                    "question":   qa['question'],
                    "context_id": ctx_id,
                    "context":    context,
                    "answer":     qa['answers'][0]['text'],
                })

        if max_qa and len(qa_pairs) >= max_qa:
            break

    if max_qa:
        qa_pairs = qa_pairs[:max_qa]

    return qa_pairs

def extract_all_contexts(data: Dict) -> List[Dict]:
    """
    インデックス用に全コンテキストを抽出する

    Returns:
        [{"context_id": int, "context": str, "article_title": str}, ...]
    """
    contexts = []
    context_id = 0
    for article in data['data']:
        for para in article['paragraphs']:
            context = para['context'].replace('[SEP]', '').strip()
            contexts.append({
                "context_id":    context_id,
                "context":       context,
                "article_title": article['title'],
            })
            context_id += 1
    return contexts

# ============================================================
# インデックス構築
# ============================================================

def build_index(contexts: List[Dict], method_name: str, model: SentenceTransformer) -> Tuple[faiss.Index, List[Dict]]:
    """
    コンテキストをチャンク化 → 埋め込み → FAISSインデックス化

    Returns:
        (faiss_index, chunk_list)
        chunk_list: [{"context_id": int, "text": str}, ...]
    """
    os.makedirs(EVAL_DATA_DIR, exist_ok=True)
    index_path = f"{EVAL_DATA_DIR}/faiss_{method_name}.bin"
    chunk_path  = f"{EVAL_DATA_DIR}/chunks_{method_name}.json"

    # キャッシュがあればロード
    if os.path.exists(index_path) and os.path.exists(chunk_path):
        print(f"  キャッシュ読み込み: {method_name}")
        index = faiss.read_index(index_path)
        with open(chunk_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        return index, chunks

    # チャンキング
    print(f"  チャンキング中 ({method_name})...")
    chunk_fn = CHUNKING_METHODS[method_name]
    chunks = []
    for ctx in contexts:
        for chunk_text in chunk_fn(ctx['context']):
            if chunk_text:
                chunks.append({
                    "context_id": ctx['context_id'],
                    "text":       chunk_text,
                })

    print(f"  チャンク数: {len(chunks)}")

    # 埋め込み
    print(f"  埋め込み計算中...")
    texts = [c['text'] for c in chunks]
    embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=True).astype(np.float32)
    faiss.normalize_L2(embeddings)

    # インデックス作成
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(embeddings)

    # 保存
    faiss.write_index(index, index_path)
    with open(chunk_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False)

    return index, chunks

# ============================================================
# 評価
# ============================================================

def evaluate_method(
    qa_pairs: List[Dict],
    index: faiss.Index,
    chunks: List[Dict],
    model: SentenceTransformer,
    k: int,
) -> Dict[str, float]:
    """
    1手法・1KでPrecision / Recall / F1 を計算する

    評価の考え方：
      - 各質問に対して正解は1つのcontext_idを持つ
      - Top-K検索で取得したチャンクのcontext_idセットに正解が含まれるかどうかで判定
        TP = 正解context_idが取得済み  → 1
        FN = 正解context_idが未取得   → 0
        FP = 取得したがすべて不正解   → K件（最悪ケース）
      - Precision@K = TP / K
      - Recall@K    = TP / 1（正解は常に1件）
      - F1@K        = 調和平均
    """
    total_precision = 0.0
    total_recall    = 0.0
    total_f1        = 0.0
    n = len(qa_pairs)

    # クエリを一括エンコード（高速化）
    questions = [qa['question'] for qa in qa_pairs]
    query_vecs = model.encode(questions, convert_to_tensor=False, show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(query_vecs)

    scores_all, indices_all = index.search(query_vecs, k)

    for i, qa in enumerate(qa_pairs):
        correct_ctx_id = qa['context_id']
        retrieved_ctx_ids = {chunks[idx]['context_id'] for idx in indices_all[i] if idx >= 0}

        hit = 1 if correct_ctx_id in retrieved_ctx_ids else 0

        precision = hit / k
        recall    = hit / 1  # 正解は1件
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        total_precision += precision
        total_recall    += recall
        total_f1        += f1

    return {
        "precision": total_precision / n,
        "recall":    total_recall    / n,
        "f1":        total_f1        / n,
    }

# ============================================================
# メイン
# ============================================================

def main():
    print("=" * 60)
    print("JSQuAD チャンキング手法 比較評価")
    print("=" * 60)

    # JSQuADロード
    data = load_jsquad()
    all_contexts = extract_all_contexts(data)
    qa_pairs     = extract_qa_pairs(data, max_qa=MAX_QA)
    print(f"\nコンテキスト数: {len(all_contexts)}")
    print(f"評価QA数: {len(qa_pairs)}")

    # 埋め込みモデルロード
    print(f"\n埋め込みモデルロード: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 結果格納
    results = {}  # {method: {k: {precision, recall, f1}}}

    # 各手法でインデックス構築 → 評価
    for method_name in CHUNKING_METHODS.keys():
        print(f"\n{'─'*40}")
        print(f"手法: {method_name}")
        print(f"{'─'*40}")

        index, chunks = build_index(all_contexts, method_name, model)
        results[method_name] = {}

        for k in K_VALUES:
            print(f"  評価中 K={k}...")
            metrics = evaluate_method(qa_pairs, index, chunks, model, k)
            results[method_name][k] = metrics
            print(f"    Precision@{k}: {metrics['precision']:.4f}  "
                  f"Recall@{k}: {metrics['recall']:.4f}  "
                  f"F1@{k}: {metrics['f1']:.4f}")

    # 結果サマリー表示
    print(f"\n{'=' * 60}")
    print("結果サマリー")
    print(f"{'=' * 60}")

    for k in K_VALUES:
        print(f"\n--- Top-{k} ---")
        print(f"{'手法':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 45)
        for method in CHUNKING_METHODS.keys():
            m = results[method][k]
            print(f"{method:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    # JSON保存
    os.makedirs(EVAL_DATA_DIR, exist_ok=True)
    result_path = f"{EVAL_DATA_DIR}/results.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存: {result_path}")

if __name__ == "__main__":
    main()
