from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_QA_PATH = BASE_DIR / "data" / "raw" / "finance_QA.jsonl"
OUTPUT_PATH = BASE_DIR / "data" / "eval" / "eval_cases.json"

MAX_CASES = 30
MIN_QUESTION_LENGTH = 20
MIN_ANSWER_LENGTH = 40


def extract_turns(payload: dict) -> tuple[str, str] | None:
    conversations = payload.get("conversations")
    if not isinstance(conversations, list):
        return None

    question = ""
    answer = ""
    for item in conversations:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if role == "user" and not question:
            question = content
        elif role == "assistant" and not answer:
            answer = content

    if len(question) < MIN_QUESTION_LENGTH or len(answer) < MIN_ANSWER_LENGTH:
        return None
    return question, answer


def load_cases() -> list[dict]:
    rows: list[tuple[str, str]] = []
    with RAW_QA_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            turns = extract_turns(payload)
            if turns is not None:
                rows.append(turns)

    if not rows:
        return []

    step = max(len(rows) // MAX_CASES, 1)
    selected: list[dict] = []
    seen_questions: set[str] = set()
    for index in range(0, len(rows), step):
        question, answer = rows[index]
        if question in seen_questions:
            continue
        selected.append(
            {
                "qid": f"q{len(selected) + 1}",
                "question": question,
                "task_type": "qa",
                "reference_answer": answer,
                "source_dataset": "finance_QA.jsonl",
            }
        )
        seen_questions.add(question)
        if len(selected) >= MAX_CASES:
            break
    return selected


def main() -> None:
    cases = load_cases()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} eval cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
