import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from config import DEFAULT_TOP_K
from src.data.schemas import ChunkRecord
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.index import IndexSearchResult, VectorIndex, load_index_metadata


QueryType = Literal[
    "business_segment",
    "business_structure",
    "business_competition",
    "risk_regulatory",
    "risk_liquidity",
    "risk_market",
    "risk_credit",
    "risk_capital",
    "risk_technology",
    "risk_ai",
    "risk_governance",
    "performance_general",
    "general",
]

EMBEDDING_WEIGHT = 0.70
SECTION_WEIGHT = 0.15
TOPIC_WEIGHT = 0.10
QUALITY_WEIGHT = 0.05


@dataclass
class RerankBreakdown:
    query_type: QueryType
    embedding_score: float
    section_bonus: float
    topic_bonus: float
    quality_bonus: float
    final_score: float


@dataclass
class RetrievedChunk:
    chunk: ChunkRecord
    score: float
    embedding_score: float = 0.0
    rerank: RerankBreakdown | None = None


@dataclass
class QueryProfile:
    query_type: QueryType
    sections: list[str]
    topics: list[str]
    title_keywords: list[str]


class Retriever:
    def __init__(self, index: VectorIndex, embedding_model: EmbeddingModel) -> None:
        self.index = index
        self.embedding_model = embedding_model

    @classmethod
    def load(cls, index_path: Path, embedding_model: EmbeddingModel) -> "Retriever":
        metadata = load_index_metadata(index_path)
        stored_embedding = metadata.get("embedding", {})
        stored_dimension = stored_embedding.get("dimension")
        if isinstance(stored_dimension, int) and stored_dimension != embedding_model.dimension:
            raise ValueError(
                "Embedding dimension mismatch between index and query model: "
                f"index={stored_dimension}, query_model={embedding_model.dimension}. "
                "Rebuild the index with the current embedding configuration."
            )
        index = VectorIndex.load(index_path)
        return cls(index=index, embedding_model=embedding_model)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        query_vector = self.embedding_model.embed_text(query)
        candidate_k = max(top_k * 16, top_k, 32)
        matches: list[IndexSearchResult] = self.index.search(query_vector, candidate_k)
        profile = classify_query(query)
        reranked: list[RetrievedChunk] = []
        for item in matches:
            breakdown = score_candidate(item, profile)
            reranked.append(
                RetrievedChunk(
                    chunk=item.chunk,
                    score=breakdown.final_score,
                    embedding_score=item.score,
                    rerank=breakdown,
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]


def classify_query(query: str) -> QueryProfile:
    normalized = query.lower()

    if _has_any_keyword(normalized, ["ai", "artificial intelligence", "generative", "agentic"]):
        return QueryProfile(
            query_type="risk_ai",
            sections=["Item 1A Risk Factors"],
            topics=["cyber_risk", "technology_risk", "operational_risk"],
            title_keywords=["ai", "advanced technologies", "cybersecurity", "people risk"],
        )

    if any(k in normalized for k in ["cyber", "cybersecurity", "technology", "technological"]):
        return QueryProfile(
            query_type="risk_technology",
            sections=["Item 1A Risk Factors"],
            topics=["cyber_risk", "technology_risk", "operational_risk"],
            title_keywords=["cybersecurity", "technology", "advanced technologies"],
        )

    if any(k in normalized for k in ["liquidity", "funding", "contingency funding"]):
        return QueryProfile(
            query_type="risk_liquidity",
            sections=["Item 7 MD&A"],
            topics=["liquidity_risk"],
            title_keywords=["liquidity", "funding", "contingency"],
        )

    if any(k in normalized for k in ["market risk", "value-at-risk", "interest rate", "foreign exchange"]):
        return QueryProfile(
            query_type="risk_market",
            sections=["Item 7 MD&A", "Item 1A Risk Factors"],
            topics=["market_risk"],
            title_keywords=["market risk", "value-at-risk"],
        )

    if any(k in normalized for k in ["credit risk", "credit and investment", "default"]):
        return QueryProfile(
            query_type="risk_credit",
            sections=["Item 7 MD&A"],
            topics=["credit_risk"],
            title_keywords=["credit risk", "credit and investment"],
        )

    if any(k in normalized for k in ["capital", "capital requirement", "capital regulation"]):
        return QueryProfile(
            query_type="risk_capital",
            sections=["Item 7 MD&A", "Item 1 Business", "Item 1A Risk Factors"],
            topics=["capital_regulation", "regulatory_risk", "financial_risk"],
            title_keywords=["capital", "regulatory capital"],
        )

    if any(k in normalized for k in ["risk governance", "governance", "risk type", "risk categories"]):
        return QueryProfile(
            query_type="risk_governance",
            sections=["Item 7 MD&A"],
            topics=["financial_risk", "risk_governance"],
            title_keywords=["risk governance", "risk management", "risk types"],
        )

    if any(k in normalized for k in ["regulation", "regulatory", "compliance", "litigation"]):
        return QueryProfile(
            query_type="risk_regulatory",
            sections=["Item 1A Risk Factors", "Item 1 Business", "Item 7 MD&A"],
            topics=["regulatory_risk", "litigation_and_enforcement"],
            title_keywords=["legal", "regulatory", "regulation", "compliance"],
        )

    if any(k in normalized for k in ["competitive", "competition", "competitor"]):
        return QueryProfile(
            query_type="business_competition",
            sections=["Item 1 Business", "Item 1A Risk Factors"],
            topics=["competition", "competitive_environment"],
            title_keywords=["competitive", "competition"],
        )

    if any(k in normalized for k in ["subsidiary", "subsidiaries", "operating structure", "bank and non-bank"]):
        return QueryProfile(
            query_type="business_structure",
            sections=["Item 1 Business"],
            topics=["business_structure", "bank_subsidiaries", "international_structure"],
            title_keywords=["subsidiaries", "operating structure", "bank"],
        )

    if any(k in normalized for k in ["segment", "business segment", "reportable"]):
        return QueryProfile(
            query_type="business_segment",
            sections=["Item 1 Business", "Item 7 MD&A"],
            topics=["business_segment", "segment_overview", "company_overview"],
            title_keywords=["business segments", "segment"],
        )

    if any(k in normalized for k in ["revenue", "expense", "balance", "cash flow", "income", "earnings"]):
        return QueryProfile(
            query_type="performance_general",
            sections=["Item 7 MD&A"],
            topics=["performance_analysis", "balance_sheet_trends"],
            title_keywords=["revenue", "expense", "performance", "balance"],
        )

    return QueryProfile(
        query_type="general",
        sections=["Item 7 MD&A", "Item 1 Business", "Item 1A Risk Factors"],
        topics=[],
        title_keywords=[],
    )


def score_candidate(item: IndexSearchResult, profile: QueryProfile) -> RerankBreakdown:
    section_bonus = _section_bonus(item.chunk.section, profile)
    topic_bonus = _topic_bonus(item.chunk, profile)
    quality_bonus = _quality_bonus(item.chunk.quality)
    final_score = (
        EMBEDDING_WEIGHT * item.score
        + section_bonus
        + topic_bonus
        + quality_bonus
    )
    return RerankBreakdown(
        query_type=profile.query_type,
        embedding_score=item.score,
        section_bonus=section_bonus,
        topic_bonus=topic_bonus,
        quality_bonus=quality_bonus,
        final_score=final_score,
    )


def _section_bonus(section: str, profile: QueryProfile) -> float:
    if not section or section not in profile.sections:
        return 0.0
    rank = profile.sections.index(section)
    normalized = max(0.0, 1.0 - rank * 0.25)
    return SECTION_WEIGHT * normalized


def _topic_bonus(chunk: ChunkRecord, profile: QueryProfile) -> float:
    raw_bonus = 0.0
    if chunk.secondary_topic in profile.topics:
        raw_bonus += 0.70
    if chunk.primary_topic in profile.topics:
        raw_bonus += 0.50
    title = chunk.title.lower()
    if any(keyword in title for keyword in profile.title_keywords):
        raw_bonus += 0.15
    if raw_bonus <= 0.0:
        return 0.0
    return TOPIC_WEIGHT * min(raw_bonus, 1.0)


def _quality_bonus(quality: str) -> float:
    normalized = quality.lower()
    if normalized == "high":
        return QUALITY_WEIGHT
    if normalized == "medium":
        return QUALITY_WEIGHT * 0.4
    return 0.0


def _has_any_keyword(text: str, keywords: list[str]) -> bool:
    for keyword in keywords:
        if " " in keyword or "-" in keyword:
            if keyword in text:
                return True
            continue
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return True
    return False
