import json

# 元のファイルを読み込む
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 必要な項目だけ抽出
extracted = {
    "講義概要": data["sections"].get("講義概要"),
    "授業科目の内容・目的・方法・到達目標": data["sections"].get("授業科目の内容・目的・方法・到達目標")
}

# 新しいファイルに保存
with open("sections_extracted.json", "w", encoding="utf-8") as f:
    json.dump(extracted, f, ensure_ascii=False, indent=2)