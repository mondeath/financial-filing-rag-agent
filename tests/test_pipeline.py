import json
import tempfile
import unittest
from pathlib import Path

from src.data.chunking import build_chunks_file
from src.llm.base import AnswerGenerator
from src.llm.generator import GroundedExtractiveGenerator
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.embeddings import HashingEmbeddingModel
from src.retrieval.index import build_index_from_chunks
from src.retrieval.retriever import Retriever


class StubStructuredGenerator(AnswerGenerator):
    def generate(self, question: str, chunks, prompt: str) -> str:
        del question, chunks, prompt
        return (
            "Answer:\n"
            "降准释放了长期流动性，并支持实体经济融资。\n\n"
            "Sources:\n"
            "- 降准政策点评 (Research, 2026-04-10)\n\n"
            "Evidence:\n"
            "1. 央行宣布降准，释放长期流动性。\n"
            "2. 政策目标是降低银行资金成本并支持实体经济融资。\n"
        )


class PipelineTests(unittest.TestCase):
    def test_answer_question_returns_grounded_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "finance_docs.jsonl"
            chunks_path = Path(tmpdir) / "chunks.jsonl"
            index_path = Path(tmpdir) / "index_store"

            docs = [
                {
                    "doc_id": "doc_policy",
                    "title": "降准政策点评",
                    "source": "Research",
                    "date": "2026-04-10",
                    "category": "macro",
                    "content": (
                        "央行宣布降准，释放长期流动性。"
                        "政策目标是降低银行资金成本并支持实体经济融资。"
                    ),
                }
            ]
            raw_path.write_text(
                "\n".join(json.dumps(doc, ensure_ascii=False) for doc in docs) + "\n",
                encoding="utf-8",
            )

            build_chunks_file(raw_docs_path=raw_path, output_path=chunks_path)
            build_index_from_chunks(
                chunks_path=chunks_path,
                output_path=index_path,
                embedding_model=HashingEmbeddingModel(),
            )

            retriever = Retriever.load(
                index_path=index_path,
                embedding_model=HashingEmbeddingModel(),
            )
            pipeline = RAGPipeline(
                retriever=retriever,
                generator=GroundedExtractiveGenerator(),
            )

            response = pipeline.answer_question("降准有什么作用？")

            self.assertIn("流动性", response.answer)
            self.assertEqual(len(response.sources), 1)
            self.assertGreaterEqual(len(response.evidence), 1)
            self.assertIn("Question", response.prompt)

    def test_pipeline_can_parse_structured_llm_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "finance_docs.jsonl"
            chunks_path = Path(tmpdir) / "chunks.jsonl"
            index_path = Path(tmpdir) / "index_store"

            docs = [
                {
                    "doc_id": "doc_policy",
                    "title": "降准政策点评",
                    "source": "Research",
                    "date": "2026-04-10",
                    "category": "macro",
                    "content": (
                        "央行宣布降准，释放长期流动性。"
                        "政策目标是降低银行资金成本并支持实体经济融资。"
                    ),
                }
            ]
            raw_path.write_text(
                "\n".join(json.dumps(doc, ensure_ascii=False) for doc in docs) + "\n",
                encoding="utf-8",
            )

            build_chunks_file(raw_docs_path=raw_path, output_path=chunks_path)
            build_index_from_chunks(
                chunks_path=chunks_path,
                output_path=index_path,
                embedding_model=HashingEmbeddingModel(),
            )

            retriever = Retriever.load(
                index_path=index_path,
                embedding_model=HashingEmbeddingModel(),
            )
            pipeline = RAGPipeline(
                retriever=retriever,
                generator=StubStructuredGenerator(),
            )

            response = pipeline.answer_question("降准有什么作用？")

            self.assertIn("长期流动性", response.answer)
            self.assertEqual(response.sources, ["降准政策点评 (Research, 2026-04-10)"])
            self.assertEqual(
                response.evidence,
                [
                    "央行宣布降准，释放长期流动性。",
                    "政策目标是降低银行资金成本并支持实体经济融资。",
                ],
            )

    def test_answer_question_returns_insufficient_information_for_irrelevant_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "finance_docs.jsonl"
            chunks_path = Path(tmpdir) / "chunks.jsonl"
            index_path = Path(tmpdir) / "index_store"

            docs = [
                {
                    "doc_id": "doc_equity",
                    "title": "半导体板块复盘",
                    "source": "Research",
                    "date": "2026-04-11",
                    "category": "equity",
                    "content": "半导体景气度回升，库存去化改善。",
                }
            ]
            raw_path.write_text(
                "\n".join(json.dumps(doc, ensure_ascii=False) for doc in docs) + "\n",
                encoding="utf-8",
            )

            build_chunks_file(raw_docs_path=raw_path, output_path=chunks_path)
            build_index_from_chunks(
                chunks_path=chunks_path,
                output_path=index_path,
                embedding_model=HashingEmbeddingModel(),
            )

            retriever = Retriever.load(
                index_path=index_path,
                embedding_model=HashingEmbeddingModel(),
            )
            pipeline = RAGPipeline(
                retriever=retriever,
                generator=GroundedExtractiveGenerator(),
            )

            response = pipeline.answer_question("美国非农数据对美元有什么影响？")

            self.assertEqual(response.answer, "insufficient information")
            self.assertEqual(response.sources, [])
            self.assertEqual(response.evidence, [])


if __name__ == "__main__":
    unittest.main()
