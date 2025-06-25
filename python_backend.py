# api_vector_search.py

from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import re
import os

app = FastAPI()

# モデルとFAISSインデックスのロード
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384  # モデルに応じて変更
index = faiss.IndexFlatL2(dimension)  # ここではFlat（類似度計算のみ）

# 文分割用関数（日本語の句点・改行で分割）
def split_sentences(text):
    return [s for s in re.split(r'[。！？\n]', text) if s.strip()]

with open(r'C:\Users\gulen\Documents\GitHub\python_backend\data.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

# 各長文を文ごとに分割し、文ベクトルの平均を計算
if os.path.exists("doc_embeddings.npy"):
    doc_embeddings = np.load("doc_embeddings.npy")
else:
    doc_embeddings = []
    for doc in documents:
        text = ""
        if isinstance(doc, dict):
            sections = doc.get("sections", {})
            text = (sections.get("講義概要", "") or "") + "\n" + (sections.get("授業科目の内容・目的・方法・到達目標", "") or "")
        else:
            text = str(doc)
        sentences = split_sentences(text)
        if sentences:
            sent_embeds = model.encode(sentences)
            mean_embed = np.mean(sent_embeds, axis=0).astype(np.float32)
            doc_embeddings.append(mean_embed)
        else:
            doc_embeddings.append(np.zeros(dimension, dtype=np.float32))

    doc_embeddings = np.array(doc_embeddings, dtype=np.float32)
    np.save("doc_embeddings.npy", doc_embeddings)

index.add(doc_embeddings)

@app.get("/show")
def show_documents():
    return {"documents": documents}

@app.get("/search")
def search(q: str = Query(..., description="検索ワード")):
    query_vec = model.encode([q]).astype(np.float32)
    k = 3
    distances, indices = index.search(query_vec, k)
    # titleだけ抽出
    results = [documents[i].get("title", "") for i in indices[0]]
    return {"query": q, "results": results}
