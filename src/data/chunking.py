from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE
from src.data.loader import load_finance_docs, write_jsonl
from src.data.schemas import ChunkRecord, FinanceDocument


@dataclass
class ChunkingConfig:
    chunk_size: int = CHUNK_SIZE
    overlap: int = CHUNK_OVERLAP

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap cannot be negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")


def split_text(text: str, config: ChunkingConfig) -> list[str]:
    clean_text = text.strip()
    if not clean_text:
        return []

    chunks: list[str] = []
    step = config.chunk_size - config.overlap
    start = 0
    while start < len(clean_text):
        end = start + config.chunk_size
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean_text):
            break
        start += step
    return chunks


def chunk_document(document: FinanceDocument, config: ChunkingConfig) -> list[ChunkRecord]:
    text_chunks = split_text(document.content, config)
    return [
        ChunkRecord(
            chunk_id=f"{document.doc_id}_chunk_{index}",
            doc_id=document.doc_id,
            title=document.title,
            source=document.source,
            date=document.date,
            chunk_index=index,
            text=text,
        )
        for index, text in enumerate(text_chunks)
    ]


def build_chunks_file(
    raw_docs_path: Path,
    output_path: Path,
    config: ChunkingConfig | None = None,
) -> int:
    active_config = config or ChunkingConfig()
    documents = load_finance_docs(raw_docs_path)
    chunks: list[ChunkRecord] = []
    for document in documents:
        chunks.extend(chunk_document(document, active_config))
    write_jsonl((chunk.to_dict() for chunk in chunks), output_path)
    return len(chunks)
