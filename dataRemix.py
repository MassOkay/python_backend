import json

# 元のファイルを読み込む
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = []
for item in data:
    title = item.get("title", "")
    if "研究会" in title:
        continue  # スキップ
    sections = item.get("sections", {})
    results.append({
        "title": title,
        "講義概要": sections.get("講義概要"),
        "授業科目の内容・目的・方法・到達目標": sections.get("授業科目の内容・目的・方法・到達目標")
    })

# 新しいファイルに保存
with open("sections_extracted.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)