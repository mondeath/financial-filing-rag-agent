from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class FinanceDocument:
    doc_id: str
    title: str
    source: str
    date: str
    category: str
    content: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinanceDocument":
        required_fields = ("doc_id", "title", "source", "date", "category", "content")
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise ValueError(f"Missing document fields: {', '.join(missing)}")

        normalized = {}
        for field in required_fields:
            value = data[field]
            if not isinstance(value, str):
                raise ValueError(f"Document field '{field}' must be a string")
            stripped = value.strip()
            if not stripped:
                raise ValueError(f"Document field '{field}' cannot be empty")
            normalized[field] = stripped
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    date: str
    chunk_index: int
    text: str
    company: str = ""
    doc_type: str = ""
    section: str = ""
    primary_topic: str = ""
    secondary_topic: str = ""
    chunk_type: str = "text"
    quality: str = "medium"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        if "id" in data and "chunk_id" not in data:
            return cls.from_jpm_10k_dict(data)

        required_fields = (
            "chunk_id",
            "doc_id",
            "title",
            "source",
            "date",
            "chunk_index",
            "text",
        )
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise ValueError(f"Missing chunk fields: {', '.join(missing)}")
        if not isinstance(data["chunk_index"], int):
            raise ValueError("Chunk field 'chunk_index' must be an integer")
        text = data["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Chunk field 'text' must be a non-empty string")

        return cls(
            chunk_id=str(data["chunk_id"]).strip(),
            doc_id=str(data["doc_id"]).strip(),
            title=str(data["title"]).strip(),
            source=str(data["source"]).strip(),
            date=str(data["date"]).strip(),
            chunk_index=data["chunk_index"],
            text=text.strip(),
            company=str(data.get("company", "")).strip(),
            doc_type=str(data.get("doc_type", "")).strip(),
            section=str(data.get("section", "")).strip(),
            primary_topic=str(data.get("primary_topic", "")).strip(),
            secondary_topic=str(data.get("secondary_topic", "")).strip(),
            chunk_type=str(data.get("chunk_type", "text")).strip() or "text",
            quality=str(data.get("quality", "medium")).strip() or "medium",
        )

    @classmethod
    def from_jpm_10k_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        required_fields = (
            "id",
            "company",
            "doc_type",
            "filing_date",
            "section",
            "primary_topic",
            "secondary_topic",
            "chunk_type",
            "quality",
            "title",
            "text",
            "source",
        )
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise ValueError(f"Missing JPM chunk fields: {', '.join(missing)}")

        text = str(data["text"]).strip()
        if not text:
            raise ValueError("JPM chunk field 'text' must be non-empty")

        chunk_id = str(data["id"]).strip()
        chunk_index = _parse_chunk_index(chunk_id)
        company = str(data["company"]).strip()
        doc_type = str(data["doc_type"]).strip()
        section = str(data["section"]).strip()
        filing_date = str(data["filing_date"]).strip()
        return cls(
            chunk_id=chunk_id,
            doc_id=f"{company} {doc_type}".strip(),
            title=str(data["title"]).strip(),
            source=str(data["source"]).strip(),
            date=filing_date,
            chunk_index=chunk_index,
            text=text,
            company=company,
            doc_type=doc_type,
            section=section,
            primary_topic=str(data["primary_topic"]).strip(),
            secondary_topic=str(data["secondary_topic"]).strip(),
            chunk_type=str(data["chunk_type"]).strip(),
            quality=str(data["quality"]).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_chunk_index(chunk_id: str) -> int:
    suffix = chunk_id.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return 0
