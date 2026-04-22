from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from config import DEFAULT_TOP_K
from src.data.schemas import ChunkRecord
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.index import IndexSearchResult, VectorIndex


@dataclass
class RetrievedChunk:
    chunk: ChunkRecord
    score: float
    embedding_score: float = 0.0


@dataclass
class QueryProfile:
    query_type: Literal["risk", "business", "performance", "general"]
    sections: list[str]
    topics: list[str]


class Retriever:
    def __init__(self, index: VectorIndex, embedding_model: EmbeddingModel) -> None:
        self.index = index
        self.embedding_model = embedding_model

    @classmethod
    def load(cls, index_path: Path, embedding_model: EmbeddingModel) -> "Retriever":
        index = VectorIndex.load(index_path)
        return cls(index=index, embedding_model=embedding_model)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        query_vector = self.embedding_model.embed_text(query)
        candidate_k = max(top_k * 8, top_k)
        matches: list[IndexSearchResult] = self.index.search(query_vector, candidate_k)
        profile = classify_query(query)
        reranked = [
            RetrievedChunk(
                chunk=item.chunk,
                score=_final_score(item, profile),
                embedding_score=item.score,
            )
            for item in matches
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]


def classify_query(query: str) -> QueryProfile:
    normalized = query.lower()

    if any(k in normalized for k in ["risk", "regulation", "compliance", "litigation"]):
        return QueryProfile(
            query_type="risk",
            sections=["Item 1A Risk Factors", "Item 7 MD&A"],
            topics=["regulatory_risk", "financial_risk"],
        )

    if any(k in normalized for k in ["segment", "business", "subsidiary", "competitive", "competition"]):
        return QueryProfile(
            query_type="business",
            sections=["Item 1 Business"],
            topics=["business_segment", "company_overview", "competition"],
        )

    if any(k in normalized for k in ["revenue", "expense", "balance", "cash flow", "liquidity"]):
        return QueryProfile(
            query_type="performance",
            sections=["Item 7 MD&A"],
            topics=["performance_analysis"],
        )

    return QueryProfile(
        query_type="general",
        sections=["Item 7 MD&A", "Item 1 Business", "Item 1A Risk Factors"],
        topics=[],
    )


def _final_score(item: IndexSearchResult, profile: QueryProfile) -> float:
    section_bonus = _section_bonus(item.chunk.section, profile)
    topic_bonus = _topic_bonus(item.chunk, profile)
    quality_bonus = _quality_bonus(item.chunk.quality)
    return (
        0.70 * item.score
        + 0.15 * section_bonus
        + 0.10 * topic_bonus
        + 0.05 * quality_bonus
    )


def _section_bonus(section: str, profile: QueryProfile) -> float:
    if profile.query_type == "risk":
        weights = {
            "Item 1A Risk Factors": 0.20,
            "Item 7 MD&A": 0.15,
            "Item 1 Business": 0.05,
        }
        return weights.get(section, 0.0)

    if profile.query_type == "business":
        weights = {
            "Item 1 Business": 0.20,
            "Item 7 MD&A": 0.10,
            "Item 1A Risk Factors": 0.05,
        }
        return weights.get(section, 0.0)

    if section in profile.sections:
        return 0.15
    return 0.0


def _topic_bonus(chunk: ChunkRecord, profile: QueryProfile) -> float:
    bonus = 0.0
    if chunk.primary_topic in profile.topics:
        bonus += 0.10
    if chunk.secondary_topic in profile.topics:
        bonus += 0.05
    return bonus


def _quality_bonus(quality: str) -> float:
    normalized = quality.lower()
    if normalized == "high":
        return 0.05
    if normalized == "medium":
        return 0.02
    return 0.0
