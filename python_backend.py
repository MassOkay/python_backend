# api_vector_search.py

from fastapi import FastAPI, Query, HTTPException
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import re
import os
from typing import List, Dict, Any
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse

app = FastAPI()

# --- 設定値 ---
MODEL_NAME = 'sonoisa/sentence-bert-base-ja-mean-tokens-v2'  # 正しい日本語特化モデル名に修正
DIMENSION = 768  # モデルの次元数に合わせて変更
FAISS_INDEX_PATH = "faiss_index.bin"
DOC_EMBEDDINGS_PATH = "doc_embeddings.npy"
CHUNK_DATA_PATH = "document_chunks.json" # チャンク情報を保存するファイル
SOURCE_JSON_PATH = "sections_extracted.json"  # 入力JSONファイルのパス
VISUALIZATION_IMAGE_PATH = "embedding_visualization.png"
SEARCH_K = 30

# --- グローバル変数 ---
model: SentenceTransformer = None
index: faiss.Index = None
documents: List[Dict[str, Any]] = []
# ドキュメントをチャンクに分割したものを保持するリスト
# 各要素は {"original_doc_id": int, "text": str} の形式
document_chunks: List[Dict[str, Any]] = []

def load_documents(path: str) -> List[Dict[str, Any]]:
    """JSONファイルからドキュメントを読み込む"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Source JSON file not found at {path}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Failed to decode JSON from {path}")

def create_chunk_embeddings(chunks: List[Dict[str, Any]], model: SentenceTransformer) -> np.ndarray:
    """チャンクリストからベクトルを作成する"""
    texts_to_encode = [chunk["text"] for chunk in chunks]
    # テキストのリストをバッチ処理で一度にエンコード
    return model.encode(texts_to_encode, convert_to_tensor=False, show_progress_bar=True)

def split_into_sentences(text: str) -> List[str]:
    """テキストを文に分割する"""
    # 空白や改行で繋がれた文を考慮し、句点や改行で分割後、不要な空白を削除
    sentences = re.split(r'(?<=[。！？.])\s*|\n+', text)
    return [s.strip() for s in sentences if s.strip()]

@app.on_event("startup")
def startup_event():
    """アプリケーション起動時にモデルとインデックスをロードする"""
    global model, index, documents, document_chunks

    model = SentenceTransformer(MODEL_NAME)
    documents = load_documents(SOURCE_JSON_PATH)

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
        print("Chunking documents...")
        for i, doc in enumerate(documents):
            title = doc.get("title", "")
            full_text = f"{doc.get('講義概要', '')}\n{doc.get('授業科目の内容・目的・方法・到達目標', '')}".strip()
            chunks = split_into_sentences(full_text)
            document_chunks.extend([{"original_doc_id": i, "text": f"{title}: {chunk}"} for chunk in chunks if chunk])
        
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
        index = faiss.IndexFlatIP(DIMENSION) # 内積（コサイン類似度）を計算するインデックスに変更
        index.add(doc_embeddings)

        # インデックスを保存
        print(f"Saving Faiss index to {FAISS_INDEX_PATH}")
        faiss.write_index(index, FAISS_INDEX_PATH)

@app.get("/show")
def show_documents() -> Dict[str, Any]:
    return {"documents": documents}

@app.get("/search")
def search(q: str = Query(..., description="検索ワード")) -> Dict[str, Any]:
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query_vec = model.encode([q]).astype(np.float32)
    faiss.normalize_L2(query_vec) # クエリベクトルも正規化
    # IndexFlatIPは内積（類似度スコア）を返す。値が大きいほど類似度が高い。
    scores, chunk_indices = index.search(query_vec, SEARCH_K)
    
    # チャンクの検索結果を元のドキュメントに集約する
    doc_scores = {}
    for chunk_idx, score in zip(chunk_indices[0], scores[0]):
        # スコアが極端に低い場合やインデックスが無効な場合はスキップ
        if chunk_idx < 0 or score < 0.1: continue
        
        original_doc_id = document_chunks[chunk_idx]["original_doc_id"]
        
        # 同じドキュメントが複数回ヒットした場合、最も高いスコアを採用（ロジックはこれで正しい）
        if original_doc_id not in doc_scores or score > doc_scores[original_doc_id]["score"]:
            doc_scores[original_doc_id] = {
                "score": score,
                "hit_chunk": document_chunks[chunk_idx]["text"] # どのチャンクがヒットしたか
            }

    # スコアの高い順にソート（IndexFlatIPを使ったので、このロジックで正しい）
    sorted_doc_ids = sorted(doc_scores.keys(), key=lambda doc_id: doc_scores[doc_id]["score"], reverse=True)

    # 最終的なレスポンスを構築
    results = []
    for doc_id in sorted_doc_ids:
        doc = documents[doc_id]
        results.append({
            "title": doc.get("title", ""),
            "overview_snippet": (doc.get("講義概要", "") or "")[:100] + "...",
            # スコアを0.0から1.0の範囲に収めて表示
            "score": min(1.0, max(0.0, float(doc_scores[doc_id]["score"]))),
            "hit_chunk": doc_scores[doc_id]["hit_chunk"]
        })
    return {"query": q, "results": results}

@app.get("/visualize")
def visualize():
    """t-SNEでベクトルを2次元に削減し、画像として返す"""
    if not os.path.exists(DOC_EMBEDDINGS_PATH):
        raise HTTPException(status_code=500, detail=f"Embeddings file not found: {DOC_EMBEDDINGS_PATH}. Please run the indexing process first.")

    doc_embeddings = np.load(DOC_EMBEDDINGS_PATH)

    # t-SNEで2次元に削減
    # 注意: t-SNEはデータセット全体で計算するため、リクエストごとに実行するのは計算コストが高いです。
    # 本番環境では、事前に画像を生成しておくか、キャッシュ戦略を検討してください。
    perplexity = min(30, len(doc_embeddings) - 1)
    if perplexity <= 0:
        raise HTTPException(status_code=500, detail="Not enough data points to visualize.")
        
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced = tsne.fit_transform(doc_embeddings)

    # 可視化
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1])
    plt.title("t-SNE Visualization of Document Embeddings")
    plt.savefig(VISUALIZATION_IMAGE_PATH)
    plt.close()
    
    return FileResponse(VISUALIZATION_IMAGE_PATH, media_type="image/png")
