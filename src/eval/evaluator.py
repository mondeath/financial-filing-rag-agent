import json
from dataclasses import dataclass
from pathlib import Path

from src.pipeline.rag_pipeline import RAGPipeline, RAGResponse


@dataclass
class EvalCase:
    qid: str
    question: str
    task_type: str
    reference_answer: str
    source_dataset: str


@dataclass
class EvalResult:
    case: EvalCase
    response: RAGResponse
    reference_answer: str
    lexical_overlap: float


def load_eval_cases(path: Path) -> list[EvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for item in payload:
        cases.append(
            EvalCase(
                qid=str(item["qid"]),
                question=str(item["question"]),
                task_type=str(item.get("task_type", "qa")),
                reference_answer=str(item.get("reference_answer", "")),
                source_dataset=str(item.get("source_dataset", "")),
            )
        )
    return cases


def run_eval_cases(
    pipeline: RAGPipeline,
    cases: list[EvalCase],
    top_k: int,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for case in cases:
        response = pipeline.answer_question(case.question, top_k=top_k)
        results.append(
            EvalResult(
                case=case,
                response=response,
                reference_answer=case.reference_answer,
                lexical_overlap=_lexical_overlap(response.answer, case.reference_answer),
            )
        )
    return results


def build_eval_report(results: list[EvalResult]) -> str:
    if not results:
        return "No eval results."

    answered_count = sum(1 for result in results if result.response.answer != "insufficient information")
    avg_overlap = sum(result.lexical_overlap for result in results) / len(results)
    avg_sources = sum(len(result.response.sources) for result in results) / len(results)
    avg_evidence = sum(len(result.response.evidence) for result in results) / len(results)

    sections = [
        "Eval Summary:",
        f"- cases: {len(results)}",
        f"- answered: {answered_count}",
        f"- insufficient_information: {len(results) - answered_count}",
        f"- avg_lexical_overlap_vs_reference: {avg_overlap:.3f}",
        f"- avg_sources_per_answer: {avg_sources:.2f}",
        f"- avg_evidence_items_per_answer: {avg_evidence:.2f}",
        "",
    ]

    for result in results:
        sections.extend(
            [
                f"[{result.case.qid}] {result.case.question}",
                f"System Answer: {result.response.answer}",
                f"Reference Answer: {result.reference_answer}",
                f"Sources: {'; '.join(result.response.sources) if result.response.sources else 'None'}",
                f"Evidence: {' | '.join(result.response.evidence) if result.response.evidence else 'None'}",
                f"Lexical Overlap: {result.lexical_overlap:.3f}",
                "Retrieved Chunks:",
                _format_retrieved_chunks(result.response.retrieved_chunks),
                "Manual Review:",
                "- retrieval_relevance: TODO",
                "- groundedness: TODO",
                "- hallucination: TODO",
                "- completeness: TODO",
                "",
            ]
        )
    return "\n".join(sections).strip()


def _format_retrieved_chunks(retrieved_chunks) -> str:
    if not retrieved_chunks:
        return "- None"

    lines: list[str] = []
    for index, item in enumerate(retrieved_chunks, 1):
        chunk = item.chunk
        lines.append(
            (
                f"- {index}. score={item.score:.4f}, "
                f"embedding={item.embedding_score:.4f}, "
                f"section={chunk.section or 'n/a'}, "
                f"primary_topic={chunk.primary_topic or 'n/a'}, "
                f"secondary_topic={chunk.secondary_topic or 'n/a'}, "
                f"quality={chunk.quality or 'n/a'}, "
                f"title={chunk.title}"
            )
        )
    return "\n".join(lines)


def _lexical_overlap(system_answer: str, reference_answer: str) -> float:
    left = set(_tokenize(system_answer))
    right = set(_tokenize(reference_answer))
    if not left or not right:
        return 0.0
    return len(left & right) / len(right)


def _tokenize(text: str) -> list[str]:
    import re

    normalized = text.lower()
    raw_tokens = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", normalized)
    tokens: list[str] = []
    for token in raw_tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.extend(list(token))
            if len(token) >= 2:
                tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            tokens.append(token)
    return [token for token in tokens if token.strip()]
