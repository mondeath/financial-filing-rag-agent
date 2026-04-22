import unittest

from src.data.schemas import ChunkRecord
from src.pipeline.debug import format_prompt_debug, format_retrieval_debug
from src.pipeline.rag_pipeline import RAGResponse
from src.retrieval.retriever import RetrievedChunk


class DebugFormattingTests(unittest.TestCase):
    def test_format_retrieval_debug(self) -> None:
        response = RAGResponse(
            answer="answer",
            sources=[],
            evidence=[],
            retrieved_chunks=[
                RetrievedChunk(
                    chunk=ChunkRecord(
                        chunk_id="c1",
                        doc_id="jpm",
                        title="Business segments",
                        source="source",
                        date="2026-02-13",
                        chunk_index=1,
                        text="text",
                        section="Item 1 Business",
                        primary_topic="business_segment",
                        secondary_topic="segment_overview",
                        quality="high",
                    ),
                    score=0.5,
                    embedding_score=0.4,
                )
            ],
            prompt="prompt text",
        )

        debug = format_retrieval_debug(response)

        self.assertIn("Retrieved Chunks:", debug)
        self.assertIn("score=0.5000", debug)
        self.assertIn("embedding=0.4000", debug)
        self.assertIn("section=Item 1 Business", debug)

    def test_format_prompt_debug(self) -> None:
        response = RAGResponse(
            answer="answer",
            sources=[],
            evidence=[],
            retrieved_chunks=[],
            prompt="prompt text",
        )

        self.assertEqual(format_prompt_debug(response), "Prompt:\nprompt text")


if __name__ == "__main__":
    unittest.main()
