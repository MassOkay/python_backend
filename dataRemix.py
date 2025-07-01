import json
import sys
from typing import List, Dict, Any

# --- 設定 ---
INPUT_FILENAME = "data.json"
OUTPUT_FILENAME = "sections_extracted.json"
FILTER_KEYWORD = ["研究会", "情報基礎"]

ENGLISH_THRESHOLD = 0.4  # 英語の割合がこの値以上の場合に除外

def is_mostly_english(text: str, threshold: float) -> bool:
    """
    テキスト内のASCII文字の割合を計算し、しきい値以上であればTrueを返す。
    主に英語で記述されているかを判定するために使用する。
    """
    if not text:
        return False

    # 空白文字を除いたテキストで判定する
    clean_text = "".join(text.split())
    if not clean_text:
        return False

    total_chars = len(clean_text)
    ascii_chars = sum(1 for char in clean_text if char.isascii())

    return (ascii_chars / total_chars) >= threshold

def process_json_data(input_path: str, output_path: str) -> None:
    """
    JSONファイルを読み込み、特定のキーワードを含む項目や、内容が主に英語で記述されている項目を除外して、
    必要なセクションだけを抽出した新しいJSONファイルを作成する。
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        print(f"エラー: 入力ファイル '{input_path}' が見つかりません。", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"エラー: ファイル '{input_path}' は不正なJSON形式です。", file=sys.stderr)
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    keyword_filtered_items: List[str] = []
    english_filtered_items: List[str] = []

    for item in data:
        title = item.get("title", "")
        if any(keyword in title for keyword in FILTER_KEYWORD):
            keyword_filtered_items.append(title)
            continue  # スキップ
        

        sections = item.get("sections", {})
        # Noneの場合は空文字に変換
        overview = sections.get("講義概要") or ""
        details = sections.get("授業科目の内容・目的・方法・到達目標") or ""

        combined_text = overview + "\n" + details

        # 英語の割合が高いデータを除外する
        if is_mostly_english(combined_text, ENGLISH_THRESHOLD):
            english_filtered_items.append(title)
            continue

        results.append({
            "title": title,
            "講義概要": overview,
            "授業科目の内容・目的・方法・到達目標": details
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("処理が完了しました。")
    print("-" * 20)
    print(f"'{FILTER_KEYWORD}' を含むため除外されたデータ ({len(keyword_filtered_items)} 件):")
    for t in keyword_filtered_items:
        print(f"  - {t}")
    print("-" * 20)
    print(f"英語の割合が高いため除外されたデータ ({len(english_filtered_items)} 件):")
    for t in english_filtered_items:
        print(f"  - {t}")
    print("-" * 20)
    print(f"-> '{output_path}' に {len(results)} 件のデータを保存しました。")

if __name__ == "__main__":
    process_json_data(INPUT_FILENAME, OUTPUT_FILENAME)