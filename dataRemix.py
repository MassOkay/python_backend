import json

# 元のファイルを読み込む
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(type(data))  # <class 'list'>

# 例：最初の要素からsectionsを抽出
sections = data[0]["sections"]

extracted = {
    "講義概要": sections.get("講義概要"),
    "授業科目の内容・目的・方法・到達目標": sections.get("授業科目の内容・目的・方法・到達目標")
}

# 新しいファイルに保存
with open("sections_extracted.json", "w", encoding="utf-8") as f:
    json.dump(extracted, f, ensure_ascii=False, indent=2)