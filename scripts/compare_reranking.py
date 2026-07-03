from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import INDEX_STORE_PATH, JPM_10K_EVAL_CASES_PATH
from src.eval.evaluator import EvalCase, load_eval_cases
from src.retrieval.embeddings import build_embedding_model
from src.retrieval.index import IndexSearchResult, VectorIndex
from src.retrieval.retriever import RetrievedChunk, Retriever, classify_query


@dataclass
class CaseComparison:
    qid: str
    question: str
    query_type: str
    expected_sections: list[str]
    expected_topics: list[str]
    embedding_top1: str
    reranked_top1: str
    embedding_top1_hit: bool
    reranked_top1_hit: bool
    embedding_topk_hit: bool
    reranked_topk_hit: bool
    embedding_topic_top1_hit: bool
    reranked_topic_top1_hit: bool
    embedding_topic_topk_hit: bool
    reranked_topic_topk_hit: bool


@dataclass
class Aggregate:
    case_count: int
    embedding_top1_section_hit_rate: float
    reranked_top1_section_hit_rate: float
    embedding_topk_section_hit_rate: float
    reranked_topk_section_hit_rate: float
    embedding_top1_topic_hit_rate: float
    reranked_top1_topic_hit_rate: float
    embedding_topk_topic_hit_rate: float
    reranked_topk_topic_hit_rate: float


def run_comparison(
    cases: list[EvalCase],
    retriever: Retriever,
    top_k: int,
) -> list[CaseComparison]:
    comparisons: list[CaseComparison] = []
    for case in cases:
        embedding_results = _embedding_only_results(
            index=retriever.index,
            embedding_model=retriever.embedding_model,
            question=case.question,
            top_k=top_k,
        )
        reranked_results = retriever.retrieve(case.question, top_k=top_k)
        profile = classify_query(case.question)
        comparisons.append(
            CaseComparison(
                qid=case.qid,
                question=case.question,
                query_type=profile.query_type,
                expected_sections=case.expected_sections,
                expected_topics=case.expected_topics,
                embedding_top1=_title(embedding_results),
                reranked_top1=_title(reranked_results),
                embedding_top1_hit=_top1_section_hit(embedding_results, case),
                reranked_top1_hit=_top1_section_hit(reranked_results, case),
                embedding_topk_hit=_topk_section_hit(embedding_results, case),
                reranked_topk_hit=_topk_section_hit(reranked_results, case),
                embedding_topic_top1_hit=_top1_topic_hit(embedding_results, case),
                reranked_topic_top1_hit=_top1_topic_hit(reranked_results, case),
                embedding_topic_topk_hit=_topk_topic_hit(embedding_results, case),
                reranked_topic_topk_hit=_topk_topic_hit(reranked_results, case),
            )
        )
    return comparisons


def build_report(comparisons: list[CaseComparison]) -> str:
    aggregate = _aggregate(comparisons)
    lines = [
        "# Reranking Ablation Report",
        "",
        "This report compares embedding-only retrieval against the V1.1 risk-aware metadata reranker.",
        "",
        "## Summary",
        "",
        f"- cases: {aggregate.case_count}",
        f"- embedding_top1_section_hit_rate: {aggregate.embedding_top1_section_hit_rate:.3f}",
        f"- reranked_top1_section_hit_rate: {aggregate.reranked_top1_section_hit_rate:.3f}",
        f"- embedding_topk_section_hit_rate: {aggregate.embedding_topk_section_hit_rate:.3f}",
        f"- reranked_topk_section_hit_rate: {aggregate.reranked_topk_section_hit_rate:.3f}",
        f"- embedding_top1_topic_hit_rate: {aggregate.embedding_top1_topic_hit_rate:.3f}",
        f"- reranked_top1_topic_hit_rate: {aggregate.reranked_top1_topic_hit_rate:.3f}",
        f"- embedding_topk_topic_hit_rate: {aggregate.embedding_topk_topic_hit_rate:.3f}",
        f"- reranked_topk_topic_hit_rate: {aggregate.reranked_topk_topic_hit_rate:.3f}",
        "",
        "## Per-Case Results",
        "",
        "| Case | Query Type | Expected Sections | Expected Topics | Embedding Top-1 | Reranked Top-1 | Section Top-1 | Topic Top-1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in comparisons:
        lines.append(
            "| "
            f"{item.qid} | "
            f"{item.query_type} | "
            f"{', '.join(item.expected_sections) or 'n/a'} | "
            f"{', '.join(item.expected_topics) or 'n/a'} | "
            f"{item.embedding_top1} | "
            f"{item.reranked_top1} | "
            f"{_hit_pair(item.embedding_top1_hit, item.reranked_top1_hit)} | "
            f"{_hit_pair(item.embedding_topic_top1_hit, item.reranked_topic_top1_hit)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Section hit rates show whether reranking moves candidates into the expected SEC filing area.",
            "- Topic hit rates show whether reranking improves alignment with curated chunk labels.",
            "- A top-k hit with a top-1 miss means the answer may still be recoverable, but ranking or generation can choose a weaker passage.",
            "- This is a small curated JPM 10-K evaluation slice; the goal is to validate whether the reranker uses filing structure and curated metadata effectively, not to claim broad benchmark-level generalization.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _embedding_only_results(index: VectorIndex, embedding_model, question: str, top_k: int) -> list[RetrievedChunk]:
    query_vector = embedding_model.embed_text(question)
    matches: list[IndexSearchResult] = index.search(query_vector, top_k)
    return [
        RetrievedChunk(
            chunk=item.chunk,
            score=item.score,
            embedding_score=item.score,
        )
        for item in matches
    ]


def _aggregate(comparisons: list[CaseComparison]) -> Aggregate:
    if not comparisons:
        return Aggregate(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    count = len(comparisons)
    return Aggregate(
        case_count=count,
        embedding_top1_section_hit_rate=_rate(item.embedding_top1_hit for item in comparisons),
        reranked_top1_section_hit_rate=_rate(item.reranked_top1_hit for item in comparisons),
        embedding_topk_section_hit_rate=_rate(item.embedding_topk_hit for item in comparisons),
        reranked_topk_section_hit_rate=_rate(item.reranked_topk_hit for item in comparisons),
        embedding_top1_topic_hit_rate=_rate(item.embedding_topic_top1_hit for item in comparisons),
        reranked_top1_topic_hit_rate=_rate(item.reranked_topic_top1_hit for item in comparisons),
        embedding_topk_topic_hit_rate=_rate(item.embedding_topic_topk_hit for item in comparisons),
        reranked_topk_topic_hit_rate=_rate(item.reranked_topic_topk_hit for item in comparisons),
    )


def _rate(values) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(1 for value in materialized if value) / len(materialized)


def _title(results: list[RetrievedChunk]) -> str:
    if not results:
        return "None"
    return results[0].chunk.title


def _top1_section_hit(results: list[RetrievedChunk], case: EvalCase) -> bool:
    return bool(results) and _section_hit(results[0], case)


def _topk_section_hit(results: list[RetrievedChunk], case: EvalCase) -> bool:
    return any(_section_hit(item, case) for item in results)


def _section_hit(item: RetrievedChunk, case: EvalCase) -> bool:
    if not case.expected_sections:
        return False
    return item.chunk.section in case.expected_sections


def _top1_topic_hit(results: list[RetrievedChunk], case: EvalCase) -> bool:
    return bool(results) and _topic_hit(results[0], case)


def _topk_topic_hit(results: list[RetrievedChunk], case: EvalCase) -> bool:
    return any(_topic_hit(item, case) for item in results)


def _topic_hit(item: RetrievedChunk, case: EvalCase) -> bool:
    if not case.expected_topics:
        return False
    return (
        item.chunk.primary_topic in case.expected_topics
        or item.chunk.secondary_topic in case.expected_topics
    )


def _hit_pair(embedding_hit: bool, reranked_hit: bool) -> str:
    return f"{_yes_no(embedding_hit)} -> {_yes_no(reranked_hit)}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare embedding-only retrieval with risk-aware reranking.")
    parser.add_argument("--cases", type=Path, default=JPM_10K_EVAL_CASES_PATH)
    parser.add_argument("--index", type=Path, default=INDEX_STORE_PATH)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports") / "reranking_ablation.md")
    args = parser.parse_args()

    cases = load_eval_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    embedding_model = build_embedding_model()
    retriever = Retriever.load(index_path=args.index, embedding_model=embedding_model)
    comparisons = run_comparison(cases=cases, retriever=retriever, top_k=args.top_k)
    report = build_report(comparisons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote reranking ablation report to {args.output}")


if __name__ == "__main__":
    main()
