# api_vector_search.py

import matplotlib
matplotlib.use('Agg') # この行を追加

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
import anthropic
import faiss
import numpy as np
import json
import re
import os
import time
from typing import List, Dict, Any
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- 設定値 ---
MODEL_NAME = 'sonoisa/sentence-bert-base-ja-mean-tokens-v2'
DIMENSION = 768
SOURCE_JSON_PATH = "data/extracted.json"
VISUALIZATION_IMAGE_PATH = "embedding_visualization.png"
SEARCH_K = 30

# チャンキング手法の選択: "fixed" | "sentence" | "llm"
CHUNKING_METHOD = "fixed"

# ① Fixed-size チャンキングのパラメータ
FIXED_CHUNK_SIZE = 100   # チャンク1つあたりの最大文字数
FIXED_CHUNK_OVERLAP = 20 # 前後チャンクとの重複文字数

# ③ LLM-based チャンキングのパラメータ
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")  # 環境変数から取得
LLM_MODEL = "claude-haiku-4-5-20251001"  # コスト効率の良いモデルを使用
LLM_RETRY_LIMIT = 3     # APIエラー時の最大リトライ回数
LLM_RETRY_WAIT = 5      # リトライ間隔（秒）

# 手法ごとにファイルパスを分ける（手法を切り替えても上書きしない）
FAISS_INDEX_PATH    = f"data/faiss_index_{CHUNKING_METHOD}.bin"
DOC_EMBEDDINGS_PATH = f"data/doc_embeddings_{CHUNKING_METHOD}.npy"
CHUNK_DATA_PATH     = f"data/document_chunks_{CHUNKING_METHOD}.json"

# グローバル変数（model, index, documents, document_chunks）は削除しました


def load_documents(path: str) -> List[Dict[str, Any]]:
    """JSONファイルからドキュメントを読み込む"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Source JSON file not found at {path}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Failed to decode JSON from {path}")

def split_into_sentences(text: str) -> List[str]:
    """② Sentence-based: 句点・改行で文単位に分割する"""
    sentences = re.split(r'(?<=[。！？.])\s*|\n+', text)
    return [s.strip() for s in sentences if s.strip()]

def split_fixed_size(text: str, chunk_size: int = FIXED_CHUNK_SIZE, overlap: int = FIXED_CHUNK_OVERLAP) -> List[str]:
    """① Fixed-size: 固定文字数でスライディングウィンドウ分割する（ベースライン）

    Args:
        text:       分割対象のテキスト
        chunk_size: チャンク1つの最大文字数
        overlap:    隣接チャンクとの重複文字数（文脈の連続性を保つため）
    Returns:
        チャンク文字列のリスト
    """
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # overlap分だけ戻って次のチャンクを開始
        start += chunk_size - overlap
    return chunks

def split_by_llm(text: str) -> List[str]:
    """③ LLM-based: Claude APIにトピック転換点を検出させて分割する（提案手法）

    LLMに <boundary> タグを挿入させ、そのタグで split することで
    意味的に完結したチャンクを生成する。
    レートリミット対策のリトライ付き。

    Args:
        text: 分割対象のテキスト
    Returns:
        意味境界で分割されたチャンク文字列のリスト
        （APIエラー時は sentence-based にフォールバック）
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY が設定されていません。環境変数を確認してください。")

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
            response_text = message.content[0].text

            # <boundary> タグで分割し、空文字を除去
            raw_chunks = response_text.split("<boundary>")
            chunks = [c.strip() for c in raw_chunks if c.strip()]
            return chunks if chunks else split_into_sentences(text)  # 空なら fallback

        except anthropic.RateLimitError:
            if attempt < LLM_RETRY_LIMIT - 1:
                print(f"  Rate limit hit. Waiting {LLM_RETRY_WAIT}s before retry ({attempt + 1}/{LLM_RETRY_LIMIT})...")
                time.sleep(LLM_RETRY_WAIT)
            else:
                print("  Rate limit exceeded. Falling back to sentence-based chunking.")
                return split_into_sentences(text)

        except anthropic.APIError as e:
            print(f"  Anthropic API error: {e}. Falling back to sentence-based chunking.")
            return split_into_sentences(text)

def chunk_document(text: str, method: str) -> List[str]:
    """手法名に応じてチャンキング関数を呼び分けるディスパッチャ"""
    if method == "fixed":
        return split_fixed_size(text)
    elif method == "sentence":
        return split_into_sentences(text)
    elif method == "llm":
        return split_by_llm(text)
    else:
        raise ValueError(f"Unknown chunking method: '{method}'. Choose from: fixed | sentence | llm")

def create_chunk_embeddings(chunks: List[Dict[str, Any]], model: SentenceTransformer) -> np.ndarray:
    """チャンクリストからベクトルを作成する"""
    texts_to_encode = [chunk["text"] for chunk in chunks]
    return model.encode(texts_to_encode, convert_to_tensor=False, show_progress_bar=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーション起動時にモデルとインデックスをロードする"""
    print("Starting up and loading resources...")
    
    # ローカル変数として初期化
    model = SentenceTransformer(MODEL_NAME)
    documents = load_documents(SOURCE_JSON_PATH)
    document_chunks = []
    index = None

    # 生成済みファイルがすべて存在する場合のみ、それらをロードする
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNK_DATA_PATH):
        print(f"Loading existing Faiss index from {FAISS_INDEX_PATH}")
        index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"Loading document chunks from {CHUNK_DATA_PATH}")
        with open(CHUNK_DATA_PATH, 'r', encoding='utf-8') as f:
            document_chunks = json.load(f)
    else:
        print("One or more generated files not found. Regenerating index and chunks.")
        
        # 1. ドキュメントをチャンクに分割
        print(f"Chunking documents with method='{CHUNKING_METHOD}'...")
        for i, doc in enumerate(documents):
            title = doc.get("title", "")
            full_text = f"{doc.get('講義概要', '')}\n{doc.get('授業科目の内容・目的・方法・到達目標', '')}".strip()
            chunks = chunk_document(full_text, CHUNKING_METHOD)
            document_chunks.extend([
                {"original_doc_id": i, "text": f"{title}: {chunk}", "method": CHUNKING_METHOD}
                for chunk in chunks if chunk
            ])
        
        with open(CHUNK_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(document_chunks, f, ensure_ascii=False)
        print(f"Saved {len(document_chunks)} chunks to {CHUNK_DATA_PATH}")

        # 2. チャンクのベクトルを作成・保存
        print("Creating new embeddings... (this may take a while)")
        doc_embeddings = create_chunk_embeddings(document_chunks, model).astype(np.float32)
        np.save(DOC_EMBEDDINGS_PATH, doc_embeddings)
        print(f"Saved embeddings to {DOC_EMBEDDINGS_PATH}")
        
        # 3. Faissインデックスを作成・保存
        faiss.normalize_L2(doc_embeddings)
        
        print("Using IndexFlatIP for cosine similarity search.")
        index = faiss.IndexFlatIP(DIMENSION)
        index.add(doc_embeddings)

        # インデックスを保存
        print(f"Saving Faiss index to {FAISS_INDEX_PATH}")
        faiss.write_index(index, FAISS_INDEX_PATH)

    # アプリケーションの状態として辞書をyieldする（これが request.state になります）
    yield {
        "model": model,
        "index": index,
        "documents": documents,
        "document_chunks": document_chunks
    }
    
    # シャットダウン時の処理
    print("Shutting down...")

# FastAPIインスタンスに lifespan を登録
app = FastAPI(lifespan=lifespan)

# CORS設定などが必要な場合はコメントアウトを外してください
# origins = [ ... ]
# app.add_middleware( ... )


@app.get("/show")
def show_documents(request: Request) -> Dict[str, Any]:
    # request.state からドキュメントを取得
    return {"documents": request.state.documents}

@app.get("/search")
def search(request: Request, q: str = Query(..., description="検索ワード")) -> Dict[str, Any]:
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # request.state からモデルやインデックスを取得
    model = request.state.model
    index = request.state.index
    documents = request.state.documents
    document_chunks = request.state.document_chunks

    query_vec = model.encode([q]).astype(np.float32)
    faiss.normalize_L2(query_vec)
    
    scores, chunk_indices = index.search(query_vec, SEARCH_K)
    
    doc_scores = {}
    for chunk_idx, score in zip(chunk_indices[0], scores[0]):
        if chunk_idx < 0 or score < 0.5: continue
        
        original_doc_id = document_chunks[chunk_idx]["original_doc_id"]
        
        if original_doc_id not in doc_scores or score > doc_scores[original_doc_id]["score"]:
            doc_scores[original_doc_id] = {
                "score": score,
                "hit_chunk": document_chunks[chunk_idx]["text"]
            }

    sorted_doc_ids = sorted(doc_scores.keys(), key=lambda doc_id: doc_scores[doc_id]["score"], reverse=True)

    results = []
    for doc_id in sorted_doc_ids:
        doc = documents[doc_id]
        results.append({
            "title": doc.get("title", ""),
            "overview_snippet": (doc.get("講義概要", "") or "")[:100] + "...",
            "score": min(1.0, max(0.0, float(doc_scores[doc_id]["score"]))),
            "hit_chunk": doc_scores[doc_id]["hit_chunk"]
        })
    return {"query": q, "method": CHUNKING_METHOD, "results": results}

@app.get("/visualize")
def visualize():
    """t-SNEでベクトルを2次元に削減し、画像として返す"""
    # このエンドポイントはファイルから直接読み込むため request.state は不要です
    if not os.path.exists(DOC_EMBEDDINGS_PATH):
        raise HTTPException(status_code=500, detail=f"Embeddings file not found: {DOC_EMBEDDINGS_PATH}. Please run the indexing process first.")

    doc_embeddings = np.load(DOC_EMBEDDINGS_PATH)

    perplexity = min(30, len(doc_embeddings) - 1)
    if perplexity <= 0:
        raise HTTPException(status_code=500, detail="Not enough data points to visualize.")
        
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced = tsne.fit_transform(doc_embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1])
    plt.title("t-SNE Visualization of Document Embeddings")
    plt.savefig(VISUALIZATION_IMAGE_PATH)
    plt.close()
    
    return FileResponse(VISUALIZATION_IMAGE_PATH, media_type="image/png")