# 1. ベースイメージの指定
FROM python:3.11-buster

# 2. 環境変数の設定 (警告を修正)
ENV PYTHONUNBUFFERED=1

# 3. 作業ディレクトリの設定
# コンテナ内の /app ディレクトリを現在の作業ディレクトリとします。
WORKDIR /app

# 4. 依存関係ファイルのコピーとインストール
# ホスト（ローカル）のrequirements.txtをコンテナの/appにコピー
COPY requirements.txt .

# pip自体を最新にアップグレードし、依存関係をインストールします。
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. アプリケーションコードのコピー
# ホストの現在のディレクトリ（Dockerfileがある場所）のすべての内容を
# コンテナの作業ディレクトリ（/app）にコピーします。
# これにより、main.py がコンテナの /app/main.py として配置されます。
COPY . .

# 6. ポートの公開
EXPOSE 8000

# 7. コンテナ起動時に実行されるコマンド
# `main:app` は、`/app` ディレクトリにある `main.py` ファイル内の
# `app` という名前のFastAPIアプリケーションインスタンスを指します。
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "main:app", "-k", "uvicorn.workers.UvicornWorker"]