import json
import math
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.data.loader import load_chunks
from src.data.schemas import ChunkRecord
from src.retrieval.embeddings import EmbeddingModel

try:
    import faiss  # type: ignore
except ModuleNotFoundError:
    faiss = None

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:
    np = None


@dataclass
class IndexBuildResult:
    backend: str
    chunk_count: int


@dataclass
class IndexSearchResult:
    chunk: ChunkRecord
    score: float


class VectorIndex:
    def search(self, query_vector: list[float], top_k: int) -> list[IndexSearchResult]:
        raise NotImplementedError

    def save(self, output_path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, input_path: Path) -> "VectorIndex":
        meta = json.loads((input_path.with_suffix(".meta.json")).read_text(encoding="utf-8"))
        backend = meta["backend"]
        if backend == "faiss":
            return FaissVectorIndex.load(input_path)
        if backend == "simple":
            return SimpleVectorIndex.load(input_path)
        raise ValueError(f"Unsupported backend: {backend}")


class SimpleVectorIndex(VectorIndex):
    def __init__(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        self.chunks = chunks
        self.vectors = vectors

    def search(self, query_vector: list[float], top_k: int) -> list[IndexSearchResult]:
        if len(self.chunks) != len(self.vectors):
            raise ValueError("Chunk and vector counts must match")
        scored = [
            IndexSearchResult(chunk=chunk, score=_dot_product(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_chunks_db(self.chunks, _chunks_db_path(output_path))
        payload = {"vectors": self.vectors}
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle)
        meta = {"backend": "simple", "chunk_store": "sqlite"}
        output_path.with_suffix(".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, input_path: Path) -> "SimpleVectorIndex":
        with input_path.open("rb") as handle:
            payload = pickle.load(handle)
        chunks_db_path = _chunks_db_path(input_path)
        chunks = (
            _load_chunks_db(chunks_db_path)
            if chunks_db_path.exists()
            else payload["chunks"]
        )
        return cls(chunks=chunks, vectors=payload["vectors"])


class FaissVectorIndex(VectorIndex):
    def __init__(self, chunks: list[ChunkRecord], index: "faiss.IndexFlatIP") -> None:
        if faiss is None:
            raise RuntimeError("faiss is not installed")
        self.chunks = chunks
        self.index = index

    def search(self, query_vector: list[float], top_k: int) -> list[IndexSearchResult]:
        if np is None:
            raise RuntimeError("numpy is required for FAISS search")
        query_array = np.asarray([query_vector], dtype="float32")
        distances, positions = self.index.search(query_array, top_k)
        results: list[IndexSearchResult] = []
        for score, position in zip(distances[0], positions[0]):
            if position < 0:
                continue
            results.append(
                IndexSearchResult(chunk=self.chunks[position], score=float(score))
            )
        return results

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(output_path))
        _write_chunks_db(self.chunks, _chunks_db_path(output_path))
        meta = {"backend": "faiss", "chunk_store": "sqlite"}
        output_path.with_suffix(".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, input_path: Path) -> "FaissVectorIndex":
        index = faiss.read_index(str(input_path))
        chunks_db_path = _chunks_db_path(input_path)
        if chunks_db_path.exists():
            chunks = _load_chunks_db(chunks_db_path)
        else:
            chunks_path = input_path.with_suffix(".chunks.pkl")
            with chunks_path.open("rb") as handle:
                chunks = pickle.load(handle)
        return cls(chunks=chunks, index=index)


def build_index_from_chunks(
    chunks_path: Path,
    output_path: Path,
    embedding_model: EmbeddingModel,
) -> IndexBuildResult:
    chunks = load_chunks(chunks_path)
    vectors = embedding_model.embed_texts([_chunk_embedding_text(chunk) for chunk in chunks])
    if faiss is not None:
        index = _build_faiss_index(chunks=chunks, vectors=vectors)
        backend = "faiss"
    else:
        index = SimpleVectorIndex(chunks=chunks, vectors=vectors)
        backend = "simple"
    index.save(output_path)
    return IndexBuildResult(backend=backend, chunk_count=len(chunks))


def _chunk_embedding_text(chunk: ChunkRecord) -> str:
    metadata = [
        chunk.title,
        chunk.section,
        chunk.primary_topic,
        chunk.secondary_topic,
        chunk.chunk_type,
        chunk.quality,
    ]
    return "\n".join(part for part in metadata + [chunk.text] if part)


def _build_faiss_index(
    chunks: list[ChunkRecord], vectors: list[list[float]]
) -> FaissVectorIndex:
    if faiss is None:
        raise RuntimeError("faiss is not installed")
    if np is None:
        raise RuntimeError("numpy is required for FAISS indexing")
    dimension = len(vectors[0]) if vectors else 0
    index = faiss.IndexFlatIP(dimension)
    index.add(np.asarray(vectors, dtype="float32"))
    return FaissVectorIndex(chunks=chunks, index=index)


def _dot_product(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vector dimensions must match")
    return math.fsum(a * b for a, b in zip(left, right))


def _chunks_db_path(index_path: Path) -> Path:
    return index_path.with_suffix(".chunks.sqlite3")


def _write_chunks_db(chunks: list[ChunkRecord], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE IF EXISTS chunks")
        connection.execute(
            """
            CREATE TABLE chunks (
                position INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL UNIQUE,
                doc_id TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                date TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                company TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                section TEXT NOT NULL,
                primary_topic TEXT NOT NULL,
                secondary_topic TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                quality TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO chunks (
                position,
                chunk_id,
                doc_id,
                title,
                source,
                date,
                chunk_index,
                text,
                company,
                doc_type,
                section,
                primary_topic,
                secondary_topic,
                chunk_type,
                quality
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    position,
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.title,
                    chunk.source,
                    chunk.date,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.company,
                    chunk.doc_type,
                    chunk.section,
                    chunk.primary_topic,
                    chunk.secondary_topic,
                    chunk.chunk_type,
                    chunk.quality,
                )
                for position, chunk in enumerate(chunks)
            ],
        )


def _load_chunks_db(db_path: Path) -> list[ChunkRecord]:
    with sqlite3.connect(db_path) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(chunks)").fetchall()]
        if "company" in columns:
            rows = connection.execute(
                """
                SELECT
                    chunk_id,
                    doc_id,
                    title,
                    source,
                    date,
                    chunk_index,
                    text,
                    company,
                    doc_type,
                    section,
                    primary_topic,
                    secondary_topic,
                    chunk_type,
                    quality
                FROM chunks
                ORDER BY position
                """
            ).fetchall()
            return [
                ChunkRecord(
                    chunk_id=row[0],
                    doc_id=row[1],
                    title=row[2],
                    source=row[3],
                    date=row[4],
                    chunk_index=row[5],
                    text=row[6],
                    company=row[7],
                    doc_type=row[8],
                    section=row[9],
                    primary_topic=row[10],
                    secondary_topic=row[11],
                    chunk_type=row[12],
                    quality=row[13],
                )
                for row in rows
            ]
        rows = connection.execute(
            """
            SELECT chunk_id, doc_id, title, source, date, chunk_index, text
            FROM chunks
            ORDER BY position
            """
        ).fetchall()
    return [
        ChunkRecord(
            chunk_id=row[0],
            doc_id=row[1],
            title=row[2],
            source=row[3],
            date=row[4],
            chunk_index=row[5],
            text=row[6],
        )
        for row in rows
    ]
