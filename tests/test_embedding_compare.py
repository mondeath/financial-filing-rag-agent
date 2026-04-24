import unittest

from src.eval.evaluator import EvalCase, EvalResult, EvalSummary
from src.pipeline.rag_pipeline import RAGResponse
from src.retrieval.retriever import RetrievedChunk
from src.data.schemas import ChunkRecord
from scripts.compare_embedding_backends import (
    ExperimentResult,
    build_comparison_report,
)


class EmbeddingCompareTests(unittest.TestCase):
    def test_build_comparison_report_includes_summary_table(self) -> None:
        chunk = ChunkRecord(
            chunk_id="doc_1_chunk_0",
            doc_id="doc_1",
            title="Business segments",
            source="demo",
            date="2026-02-13",
            chunk_index=0,
            text="The Firm has three reportable business segments.",
            section="Item 1 Business",
            primary_topic="business_segment",
            secondary_topic="segment_overview",
            quality="high",
        )
        result = EvalResult(
            case=EvalCase(
                qid="q1",
                question="What are the business segments?",
                task_type="qa",
                reference_answer="The Firm has three reportable business segments.",
                source_dataset="demo",
            ),
            response=RAGResponse(
                answer="The Firm has three reportable business segments.",
                sources=["Business segments (demo, 2026-02-13)"],
                evidence=["The Firm has three reportable business segments."],
                retrieved_chunks=[
                    RetrievedChunk(chunk=chunk, score=0.8, embedding_score=0.7)
                ],
                prompt="prompt",
            ),
            reference_answer="The Firm has three reportable business segments.",
            lexical_overlap=1.0,
        )
        experiments = [
            ExperimentResult(
                backend_label="hashing-baseline",
                summary=EvalSummary(
                    case_count=1,
                    answered_count=1,
                    insufficient_information_count=0,
                    avg_lexical_overlap=1.0,
                    avg_sources_per_answer=1.0,
                    avg_evidence_items_per_answer=1.0,
                ),
                results=[result],
                notes=[],
                status="ok",
            ),
            ExperimentResult(
                backend_label="configured-local-bge",
                summary=None,
                results=[],
                notes=["No local BGE configuration found in environment."],
                status="skipped",
            ),
            ExperimentResult(
                backend_label="configured-remote-embedding",
                summary=None,
                results=[],
                notes=["No remote embedding configuration found in environment."],
                status="skipped",
            ),
        ]

        report = build_comparison_report(experiments)
        self.assertIn("# Embedding Comparison Report", report)
        self.assertIn("| Backend | Status | Cases | Answered | Avg Lexical Overlap |", report)
        self.assertIn("hashing-baseline", report)
        self.assertIn("configured-local-bge", report)
        self.assertIn("configured-remote-embedding", report)
        self.assertIn("### q1", report)


if __name__ == "__main__":
    unittest.main()
