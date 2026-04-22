import json
import tempfile
import unittest
from pathlib import Path

from src.data.chunking import ChunkingConfig, build_chunks_file, split_text
from src.data.loader import load_chunks


class ChunkingTests(unittest.TestCase):
    def test_split_text_uses_overlap(self) -> None:
        text = "a" * 900
        chunks = split_text(text, ChunkingConfig(chunk_size=500, overlap=80))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 500)
        self.assertEqual(chunks[0][-80:], chunks[1][:80])

    def test_build_chunks_file_outputs_spec_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "finance_docs.jsonl"
            output_path = Path(tmpdir) / "chunks.jsonl"
            doc = {
                "doc_id": "doc_0001",
                "title": "央行降准释放长期流动性",
                "source": "Sina Finance",
                "date": "2026-04-10",
                "category": "macro",
                "content": "货币政策" * 120,
            }
            raw_path.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")

            count = build_chunks_file(raw_docs_path=raw_path, output_path=output_path)

            chunks = load_chunks(output_path)
            self.assertEqual(count, len(chunks))
            self.assertEqual(chunks[0].chunk_id, "doc_0001_chunk_0")
            self.assertEqual(chunks[0].doc_id, "doc_0001")
            self.assertEqual(chunks[0].title, doc["title"])


if __name__ == "__main__":
    unittest.main()
