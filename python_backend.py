# api_vector_search.py

from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

app = FastAPI()

# モデルとFAISSインデックスのロード
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384  # モデルに応じて変更
index = faiss.IndexFlatL2(dimension)  # ここではFlat（類似度計算のみ）

# ダミーデータ（本番ではDBから読み込む）
with open(r'C:\Users\gulen\Documents\GitHub\python_backend\data.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

# ベクトル化してインデックスに追加
doc_embeddings = model.encode(documents)
index.add(np.array(doc_embeddings))

@app.get("/show")
def show_documents():
    return {"documents": documents}

@app.get("/search")
def search(q: str = Query(..., description="検索ワード")):
    query_vec = model.encode([q])
    k = 3  # 上位3件
    distances, indices = index.search(np.array(query_vec), k)

    results = [documents[i] for i in indices[0]]
    return {"query": q, "results": results}
