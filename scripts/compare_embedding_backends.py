from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import JPM_10K_CHUNKS_PATH, JPM_10K_EVAL_CASES_PATH
from src.eval.evaluator import (
    EvalResult,
    EvalSummary,
    build_eval_summary,
    load_eval_cases,
    run_eval_cases,
)
from src.llm.generator import GroundedExtractiveGenerator
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.embeddings import (
    EmbeddingError,
    HashingEmbeddingModel,
    build_remote_embedding_model_from_env,
)
from src.retrieval.index import build_index_from_chunks
from src.retrieval.retriever import Retriever


@dataclass
class ExperimentResult:
    backend_label: str
    summary: EvalSummary | None
    results: list[EvalResult]
    notes: list[str]
    status: str


def run_backend_eval(
    backend_label: str,
    chunks_path: Path,
    cases_path: Path,
    top_k: int,
    limit: int | None,
    embedding_model,
) -> ExperimentResult:
    cases = load_eval_cases(cases_path)
    if limit is not None:
        cases = cases[:limit]

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "index_store"
        build_index_from_chunks(
            chunks_path=chunks_path,
            output_path=index_path,
            embedding_model=embedding_model,
        )
        retriever = Retriever.load(index_path=index_path, embedding_model=embedding_model)
        pipeline = RAGPipeline(
            retriever=retriever,
            generator=GroundedExtractiveGenerator(),
        )
        results = run_eval_cases(pipeline=pipeline, cases=cases, top_k=top_k)

    return ExperimentResult(
        backend_label=backend_label,
        summary=build_eval_summary(results),
        results=results,
        notes=[],
        status="ok",
    )


def build_comparison_report(experiments: list[ExperimentResult]) -> str:
    lines = [
        "# Embedding Comparison Report",
        "",
        "This report compares the JPM 10-K RAG pipeline across embedding backends while keeping the corpus, reranking logic, and local grounded generator fixed.",
        "",
        "## Summary Table",
        "",
        "| Backend | Status | Cases | Answered | Avg Lexical Overlap | Avg Sources | Avg Evidence | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for experiment in experiments:
        if experiment.summary is None:
            lines.append(
                f"| {experiment.backend_label} | {experiment.status} | - | - | - | - | - | {'; '.join(experiment.notes) or 'n/a'} |"
            )
            continue
        summary = experiment.summary
        lines.append(
            f"| {experiment.backend_label} | {experiment.status} | "
            f"{summary.case_count} | {summary.answered_count} | "
            f"{summary.avg_lexical_overlap:.3f} | "
            f"{summary.avg_sources_per_answer:.2f} | "
            f"{summary.avg_evidence_items_per_answer:.2f} | "
            f"{'; '.join(experiment.notes) or 'n/a'} |"
        )

    lines.extend(
        [
            "",
            "## Per-Case Snapshot",
            "",
        ]
    )

    case_ids = []
    for experiment in experiments:
        case_ids.extend(result.case.qid for result in experiment.results)
    ordered_case_ids = list(dict.fromkeys(case_ids))

    for qid in ordered_case_ids:
        lines.append(f"### {qid}")
        lines.append("")
        for experiment in experiments:
            match = next((result for result in experiment.results if result.case.qid == qid), None)
            if match is None:
                lines.append(f"- `{experiment.backend_label}`: no result")
                continue
            top_title = (
                match.response.retrieved_chunks[0].chunk.title
                if match.response.retrieved_chunks
                else "None"
            )
            lines.append(
                f"- `{experiment.backend_label}`: overlap={match.lexical_overlap:.3f}, "
                f"top_title={top_title}, "
                f"answer={match.response.answer}"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- This comparison is most useful when the real embedding backend is available and both runs succeed on the same eval slice.",
            "- The lexical overlap metric is lightweight and should be read together with retrieved chunk titles and answer groundedness.",
            "- If the real embedding backend is unavailable, the hashing baseline still provides a reproducible local reference point.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hashing and configured embedding backends.")
    parser.add_argument("--chunks", type=Path, default=JPM_10K_CHUNKS_PATH)
    parser.add_argument("--cases", type=Path, default=JPM_10K_EVAL_CASES_PATH)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports") / "embedding_comparison.md",
    )
    args = parser.parse_args()

    experiments: list[ExperimentResult] = []

    experiments.append(
        run_backend_eval(
            backend_label="hashing-baseline",
            chunks_path=args.chunks,
            cases_path=args.cases,
            top_k=args.top_k,
            limit=args.limit,
            embedding_model=HashingEmbeddingModel(),
        )
    )

    remote_model = build_remote_embedding_model_from_env()
    if remote_model is None:
        experiments.append(
            ExperimentResult(
                backend_label="configured-remote-embedding",
                summary=None,
                results=[],
                notes=["No remote embedding configuration found in environment."],
                status="skipped",
            )
        )
    else:
        try:
            experiments.append(
                run_backend_eval(
                    backend_label=remote_model.describe(),
                    chunks_path=args.chunks,
                    cases_path=args.cases,
                    top_k=args.top_k,
                    limit=args.limit,
                    embedding_model=remote_model,
                )
            )
        except (EmbeddingError, ValueError) as exc:
            experiments.append(
                ExperimentResult(
                    backend_label=remote_model.describe(),
                    summary=None,
                    results=[],
                    notes=[str(exc)],
                    status="failed",
                )
            )

    report = build_comparison_report(experiments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote embedding comparison report to {args.output}")


if __name__ == "__main__":
    main()
