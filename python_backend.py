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
MODEL_NAME = 'all-MiniLM-L6-v2'
DIMENSION = 384  # モデル 'all-MiniLM-L6-v2' の次元数
FAISS_INDEX_PATH = "faiss_index.bin"
DOC_EMBEDDINGS_PATH = "doc_embeddings.npy"
SOURCE_JSON_PATH = r'C:\Users\gulen\Documents\GitHub\python_backend\sections_extracted.json'
VISUALIZATION_IMAGE_PATH = "embedding_visualization.png"
SEARCH_K = 30

# --- グローバル変数 ---
model: SentenceTransformer = None
index: faiss.Index = None
documents: List[Dict[str, Any]] = []

# 文分割用関数（日本語の句点・改行で分割）
def split_sentences(text: str) -> List[str]:
    return [s for s in re.split(r'[。！？.\n]', text) if s.strip()]

def load_documents(path: str) -> List[Dict[str, Any]]:
    """JSONファイルからドキュメントを読み込む"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Source JSON file not found at {path}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Failed to decode JSON from {path}")

def create_embeddings(docs: List[Dict[str, Any]], model: SentenceTransformer) -> np.ndarray:
    """ドキュメントリストからベクトルを作成する"""
    doc_embeddings_list = []
    for doc in docs:
        # 辞書でない場合や、必要なキーがない場合も考慮
        text = f'{doc.get("講義概要", "")}\n{doc.get("授業科目の内容・目的・方法・到達目標", "")}'
        sentences = split_sentences(text.strip())
        if sentences:
            sent_embeds = model.encode(sentences, convert_to_tensor=False)
            mean_embed = np.mean(sent_embeds, axis=0)
            doc_embeddings_list.append(mean_embed)
        else:
            # テキストが空の場合、ゼロベクトルを追加
            doc_embeddings_list.append(np.zeros(DIMENSION, dtype=np.float32))
    return np.array(doc_embeddings_list, dtype=np.float32)

@app.on_event("startup")
def startup_event():
    """アプリケーション起動時にモデルとインデックスをロードする"""
    global model, index, documents

    model = SentenceTransformer(MODEL_NAME)
    documents = load_documents(SOURCE_JSON_PATH)

    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing Faiss index from {FAISS_INDEX_PATH}")
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        print("Creating new Faiss index.")
        if os.path.exists(DOC_EMBEDDINGS_PATH):
            print(f"Loading embeddings from {DOC_EMBEDDINGS_PATH}")
            doc_embeddings = np.load(DOC_EMBEDDINGS_PATH)
        else:
            print("Creating new embeddings.")
            doc_embeddings = create_embeddings(documents, model)
            print(f"Saving document embeddings to {DOC_EMBEDDINGS_PATH}")
            np.save(DOC_EMBEDDINGS_PATH, doc_embeddings)
        
        # Faissインデックスの作成
        index = faiss.IndexFlatL2(DIMENSION)
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

    query_vec = model.encode([q], convert_to_tensor=False).astype(np.float32)
    distances, indices = index.search(query_vec, SEARCH_K)
    
    # titleだけ抽出
    results = [
        documents[i].get("title", "")
        for i in indices[0]
        if 0 <= i < len(documents)
    ]
    return {"query": q, "results": results, "distances": distances[0].tolist()}

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
