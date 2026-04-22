import json
from pathlib import Path
from typing import Iterable

from src.data.schemas import ChunkRecord, FinanceDocument


def load_finance_docs(path: Path) -> list[FinanceDocument]:
    documents: list[FinanceDocument] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        documents.append(FinanceDocument.from_dict(payload))
    return documents


def load_chunks(path: Path) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        chunks.append(ChunkRecord.from_dict(payload))
    return chunks


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

