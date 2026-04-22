import json
import tempfile
import unittest
from pathlib import Path

from src.eval.evaluator import build_eval_report, load_eval_cases, run_eval_cases
from src.llm.base import AnswerGenerator
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.retriever import RetrievedChunk
from src.data.schemas import ChunkRecord


class StubAnswerGenerator(AnswerGenerator):
    def generate(self, question: str, chunks: list[ChunkRecord], prompt: str) -> str:
        del question, chunks, prompt
        return (
            "Answer:\n系统根据上下文生成了回答。\n\n"
            "Sources:\n- 测试标题 (Test Source, 2026-01-01)\n\n"
            "Evidence:\n1. 这是测试证据。\n"
        )


class StubRetriever:
    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        del query, top_k
        return [
            RetrievedChunk(
                chunk=ChunkRecord(
                    chunk_id="doc_1_chunk_0",
                    doc_id="doc_1",
                    title="测试标题",
                    source="Test Source",
                    date="2026-01-01",
                    chunk_index=0,
                    text="这是测试证据。",
                    section="Item 1 Business",
                    primary_topic="company_overview",
                    secondary_topic="company_profile",
                    quality="high",
                ),
                score=1.0,
                embedding_score=0.9,
            )
        ]


class EvalTests(unittest.TestCase):
    def test_load_eval_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eval_cases.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "qid": "q1",
                            "question": "问题",
                            "task_type": "qa",
                            "reference_answer": "参考答案",
                            "source_dataset": "finance_QA.jsonl",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            cases = load_eval_cases(path)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].qid, "q1")

    def test_run_eval_cases_and_report(self) -> None:
        pipeline = RAGPipeline(
            retriever=StubRetriever(),
            generator=StubAnswerGenerator(),
        )
        cases = [
            load_eval_cases_from_dict(
                {
                    "qid": "q1",
                    "question": "测试问题",
                    "task_type": "qa",
                    "reference_answer": "这是参考答案",
                    "source_dataset": "finance_QA.jsonl",
                }
            )
        ]
        results = run_eval_cases(pipeline=pipeline, cases=cases, top_k=4)
        report = build_eval_report(results)
        self.assertEqual(len(results), 1)
        self.assertIn("Eval Summary:", report)
        self.assertIn("Manual Review:", report)
        self.assertIn("System Answer:", report)
        self.assertIn("Retrieved Chunks:", report)
        self.assertIn("section=Item 1 Business", report)
        self.assertIn("embedding=0.9000", report)


def load_eval_cases_from_dict(payload: dict):
    from src.eval.evaluator import EvalCase

    return EvalCase(
        qid=payload["qid"],
        question=payload["question"],
        task_type=payload["task_type"],
        reference_answer=payload["reference_answer"],
        source_dataset=payload["source_dataset"],
    )


if __name__ == "__main__":
    unittest.main()
