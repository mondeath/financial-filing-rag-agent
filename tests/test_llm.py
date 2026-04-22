import os
import unittest
from unittest import mock

from src.data.schemas import ChunkRecord
from src.llm.generator import (
    FallbackAnswerGenerator,
    GroundedExtractiveGenerator,
    build_answer_generator,
)
from src.llm.prompting import build_prompt


class LLMTests(unittest.TestCase):
    def test_build_answer_generator_falls_back_without_key(self) -> None:
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=False):
            generator = build_answer_generator()
        self.assertIsInstance(generator, GroundedExtractiveGenerator)

    def test_build_answer_generator_uses_fallback_wrapper_with_key(self) -> None:
        env = {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "demo-model",
            "OPENAI_BASE_URL": "https://example.com/v1",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            generator = build_answer_generator()
        self.assertIsInstance(generator, FallbackAnswerGenerator)

    def test_prompt_contains_strict_output_contract(self) -> None:
        chunks = [
            ChunkRecord(
                chunk_id="doc_1_chunk_0",
                doc_id="doc_1",
                title="降准政策点评",
                source="Research",
                date="2026-04-10",
                chunk_index=0,
                text="央行宣布降准，释放长期流动性。",
            )
        ]
        prompt = build_prompt("降准有什么影响？", chunks)
        self.assertIn("Answer:", prompt)
        self.assertIn("Sources:", prompt)
        self.assertIn("Evidence:", prompt)
        self.assertIn("Do not hallucinate", prompt)


if __name__ == "__main__":
    unittest.main()
