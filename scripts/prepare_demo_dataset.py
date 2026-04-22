from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
ARTICLE_PATH = RAW_DIR / "finance_article.jsonl"
OUTPUT_PATH = RAW_DIR / "finance_docs.jsonl"

MAX_DOCS = 20
MAX_PER_CATEGORY = 5
MIN_TEXT_LENGTH = 180

CATEGORY_RULES = {
    "macro": ["非农", "就业", "美联储", "降准", "降息", "通胀", "财政", "pmi", "gdp"],
    "commodity": ["原油", "布伦特", "黄金", "天然气", "油价", "LNG", "霍尔木兹"],
    "banking": ["银行", "农行", "工行", "建行", "中行", "贷款", "担保", "融资"],
    "equity": ["A50", "阿里巴巴", "百度", "台积电", "特斯拉", "英伟达", "亚马逊"],
    "policy": ["人大", "政协", "双城记", "金融赋能", "监管", "ipo", "审核", "证监会", "央行"],
    "company": ["营收", "巨亏", "增长", "业绩", "指引", "工厂", "订单", "需求", "市场业务"],
}


def parse_date(raw_time: str) -> str:
    try:
        return datetime.fromisoformat(raw_time).date().isoformat()
    except ValueError:
        return raw_time[:10]


def infer_category(title: str, text: str) -> str:
    combined = f"{title}\n{text}"
    for category, keywords in CATEGORY_RULES.items():
        if any(keyword in combined for keyword in keywords):
            return category
    return "general"


def normalize_record(payload: dict, index: int) -> dict | None:
    title = str(payload.get("title", "")).strip()
    text = str(payload.get("text", "")).strip()
    raw_id = str(payload.get("id", "")).strip()
    source = str(payload.get("source", "")).strip() or "unknown"
    raw_time = str(payload.get("time", "")).strip()
    if not (title and text and raw_id and raw_time):
        return None
    if len(text) < MIN_TEXT_LENGTH:
        return None

    category = infer_category(title, text)
    return {
        "doc_id": f"demo_{index:04d}",
        "title": title,
        "source": source,
        "date": parse_date(raw_time),
        "category": category,
        "content": text,
        "_raw_id": raw_id,
    }


def build_demo_dataset() -> list[dict]:
    selected: list[dict] = []
    category_counts: defaultdict[str, int] = defaultdict(int)
    seen_titles: set[str] = set()

    with ARTICLE_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            record = normalize_record(payload, index=line_number)
            if record is None:
                continue
            if record["title"] in seen_titles:
                continue

            category = record["category"]
            if category == "general":
                continue
            if category_counts[category] >= MAX_PER_CATEGORY:
                continue

            selected.append(record)
            category_counts[category] += 1
            seen_titles.add(record["title"])
            if len(selected) >= MAX_DOCS:
                break

    cleaned: list[dict] = []
    for item in selected:
        cleaned.append(
            {
                "doc_id": item["doc_id"],
                "title": item["title"],
                "source": item["source"],
                "date": item["date"],
                "category": item["category"],
                "content": item["content"],
            }
        )
    return cleaned


def main() -> None:
    docs = build_demo_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Wrote {len(docs)} demo docs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
