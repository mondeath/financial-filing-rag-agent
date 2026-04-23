import os
import unittest
from unittest import mock
from urllib import error

from src.data.schemas import ChunkRecord
from src.llm.generator import (
    FallbackAnswerGenerator,
    GroundedExtractiveGenerator,
    build_answer_generator,
)
from src.llm.prompting import build_prompt
from src.retrieval.embeddings import (
    EmbeddingError,
    FallbackEmbeddingModel,
    HashingEmbeddingModel,
    OpenAICompatibleEmbeddingConfig,
    OpenAICompatibleEmbeddingModel,
    build_embedding_model,
)
from src.retrieval.index import build_index_from_chunks
from src.retrieval.retriever import Retriever


class _FakeHTTPResponse:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload.encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


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

    def test_build_embedding_model_falls_back_without_key(self) -> None:
        with mock.patch.dict(os.environ, {"EMBEDDING_PROVIDER": "openai"}, clear=False):
            model = build_embedding_model()
        self.assertIsInstance(model, HashingEmbeddingModel)

    def test_build_embedding_model_uses_fallback_wrapper_with_key(self) -> None:
        env = {
            "EMBEDDING_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_EMBEDDING_MODEL": "demo-embedding-model",
            "OPENAI_BASE_URL": "https://example.com/v1",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            model = build_embedding_model()
        self.assertIsInstance(model, FallbackEmbeddingModel)

    def test_build_embedding_model_uses_embedding_specific_env_vars(self) -> None:
        env = {
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_API_KEY": "embed-key",
            "EMBEDDING_BASE_URL": "https://embeddings.example.com/v1",
            "OPENAI_EMBEDDING_MODEL": "demo-embedding-model",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            model = build_embedding_model()
        self.assertIsInstance(model, FallbackEmbeddingModel)
        self.assertEqual(model.primary.config.api_key, "embed-key")
        self.assertEqual(model.primary.config.base_url, "https://embeddings.example.com/v1")

    def test_openai_embedding_model_batches_inputs(self) -> None:
        model = OpenAICompatibleEmbeddingModel(
            OpenAICompatibleEmbeddingConfig(
                api_key="test-key",
                base_url="https://example.com/v1",
                model="demo-model",
                dimension=3,
                batch_size=2,
            )
        )
        responses = [
            _FakeHTTPResponse('{"data":[{"embedding":[1,0,0]},{"embedding":[0,1,0]}]}'),
            _FakeHTTPResponse('{"data":[{"embedding":[0,0,1]}]}'),
        ]
        with mock.patch("src.retrieval.embeddings.request.urlopen", side_effect=responses) as urlopen:
            vectors = model.embed_texts(["a", "b", "c"])
        self.assertEqual(vectors, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(urlopen.call_count, 2)

    def test_fallback_embedding_model_uses_hashing_on_remote_failure(self) -> None:
        env = {
            "EMBEDDING_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_EMBEDDING_MODEL": "demo-embedding-model",
            "OPENAI_BASE_URL": "https://example.com/v1",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            model = build_embedding_model()
        with mock.patch(
            "src.retrieval.embeddings.request.urlopen",
            side_effect=error.URLError("network down"),
        ):
            vector = model.embed_text("jpmorgan")
        self.assertEqual(len(vector), model.dimension)
        self.assertTrue(any(value != 0.0 for value in vector))

    def test_retriever_rejects_dimension_mismatch(self) -> None:
        with self.subTest("build index"):
            import tempfile
            from pathlib import Path

            from src.data.loader import write_jsonl

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                chunks_path = tmp_path / "chunks.jsonl"
                index_path = tmp_path / "index_store"
                write_jsonl(
                    [
                        {
                            "chunk_id": "doc_1_chunk_0",
                            "doc_id": "doc_1",
                            "title": "Business segments",
                            "source": "demo",
                            "date": "2026-02-13",
                            "chunk_index": 0,
                            "text": "JPMorgan Chase has three reportable business segments.",
                        }
                    ],
                    chunks_path,
                )
                build_index_from_chunks(
                    chunks_path=chunks_path,
                    output_path=index_path,
                    embedding_model=HashingEmbeddingModel(dimension=8),
                )
                with self.assertRaisesRegex(ValueError, "Embedding dimension mismatch"):
                    Retriever.load(
                        index_path=index_path,
                        embedding_model=HashingEmbeddingModel(dimension=16),
                    )


if __name__ == "__main__":
    unittest.main()
