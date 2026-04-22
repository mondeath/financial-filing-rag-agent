import json
import tempfile
import unittest
from pathlib import Path

from src.data.chunking import build_chunks_file
from src.retrieval.embeddings import HashingEmbeddingModel
from src.retrieval.index import build_index_from_chunks
from src.retrieval.retriever import Retriever


class RetrievalTests(unittest.TestCase):
    def test_retriever_returns_relevant_chunk_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "finance_docs.jsonl"
            chunks_path = Path(tmpdir) / "chunks.jsonl"
            index_path = Path(tmpdir) / "index_store"

            docs = [
                {
                    "doc_id": "doc_macro",
                    "title": "降准政策点评",
                    "source": "Research",
                    "date": "2026-04-10",
                    "category": "macro",
                    "content": "央行宣布降准，释放长期流动性，支持实体经济融资。",
                },
                {
                    "doc_id": "doc_equity",
                    "title": "半导体板块复盘",
                    "source": "Research",
                    "date": "2026-04-11",
                    "category": "equity",
                    "content": "半导体景气度回升，库存去化改善。",
                },
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
            self.assertTrue(index_path.with_suffix(".chunks.sqlite3").exists())
            retriever = Retriever.load(
                index_path=index_path,
                embedding_model=HashingEmbeddingModel(),
            )

            results = retriever.retrieve("降准释放了什么", top_k=2)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].chunk.doc_id, "doc_macro")
            self.assertIn("流动性", results[0].chunk.text)


if __name__ == "__main__":
    unittest.main()
