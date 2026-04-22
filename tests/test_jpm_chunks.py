import json
import tempfile
import unittest
from pathlib import Path

from src.data.loader import load_chunks
from src.data.schemas import ChunkRecord
from src.retrieval.embeddings import HashingEmbeddingModel
from src.retrieval.index import IndexSearchResult, SimpleVectorIndex
from src.retrieval.retriever import Retriever, classify_query


class JPMChunkTests(unittest.TestCase):
    def test_load_jpm_10k_chunk_schema(self) -> None:
        payload = {
            "id": "jpm_2025_10k_item1_0003",
            "company": "JPMorgan Chase & Co.",
            "doc_type": "10-K",
            "filing_date": "2026-02-13",
            "section": "Item 1 Business",
            "primary_topic": "business_segment",
            "secondary_topic": "segment_overview",
            "chunk_type": "text",
            "quality": "high",
            "title": "Business segments",
            "text": "The Firm has three reportable business segments.",
            "source": "data/raw/jpm-20251231.htm",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chunks.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            chunks = load_chunks(path)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "jpm_2025_10k_item1_0003")
        self.assertEqual(chunks[0].section, "Item 1 Business")
        self.assertEqual(chunks[0].primary_topic, "business_segment")
        self.assertEqual(chunks[0].quality, "high")

    def test_business_query_profile(self) -> None:
        profile = classify_query("What are the main business segments?")
        self.assertEqual(profile.query_type, "business")
        self.assertIn("Item 1 Business", profile.sections)
        self.assertIn("business_segment", profile.topics)

    def test_rerank_prefers_matching_business_metadata(self) -> None:
        chunks = [
            ChunkRecord(
                chunk_id="risk",
                doc_id="jpm",
                title="Cybersecurity risk",
                source="source",
                date="2026-02-13",
                chunk_index=1,
                text="Business operations may be disrupted by technology risk.",
                section="Item 1A Risk Factors",
                primary_topic="operational_risk",
                quality="medium",
            ),
            ChunkRecord(
                chunk_id="segment",
                doc_id="jpm",
                title="Business segments",
                source="source",
                date="2026-02-13",
                chunk_index=2,
                text="The Firm has three reportable business segments.",
                section="Item 1 Business",
                primary_topic="business_segment",
                quality="high",
            ),
        ]
        vectors = HashingEmbeddingModel().embed_texts([chunk.text for chunk in chunks])
        retriever = Retriever(
            index=SimpleVectorIndex(chunks=chunks, vectors=vectors),
            embedding_model=HashingEmbeddingModel(),
        )

        results = retriever.retrieve("What are the main business segments?", top_k=2)

        self.assertEqual(results[0].chunk.chunk_id, "segment")


if __name__ == "__main__":
    unittest.main()
