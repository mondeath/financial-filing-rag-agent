import hashlib
import importlib
import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request

from config import EMBEDDING_DIMENSION


class EmbeddingModel(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def metadata(self) -> dict[str, str | int]:
        return {
            "provider": self.__class__.__name__,
            "dimension": self.dimension,
        }

    def describe(self) -> str:
        metadata = self.metadata()
        provider = metadata.get("provider", self.__class__.__name__)
        dimension = metadata.get("dimension", self.dimension)
        model = metadata.get("model")
        if model:
            return f"{provider}:{model} (dim={dimension})"
        return f"{provider} (dim={dimension})"


class HashingEmbeddingModel(EmbeddingModel):
    """A deterministic stdlib-only embedding baseline."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def metadata(self) -> dict[str, str | int]:
        return {
            "provider": "hashing",
            "model": "hashing-baseline",
            "dimension": self.dimension,
        }


class EmbeddingError(RuntimeError):
    pass


@dataclass(slots=True)
class LocalSentenceTransformerConfig:
    model_name: str
    dimension: int
    device: str = "cpu"
    normalize: bool = True
    batch_size: int = 32


class LocalSentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, config: LocalSentenceTransformerConfig) -> None:
        self.config = config
        try:
            sentence_transformer_cls = _load_sentence_transformer_class()
            self._model = sentence_transformer_cls(config.model_name, device=config.device)
        except Exception as exc:
            raise EmbeddingError(f"Failed to load local embedding model: {exc}") from exc

    @property
    def dimension(self) -> int:
        return self.config.dimension

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"Local embedding inference failed: {exc}") from exc
        return [[float(value) for value in row] for row in vectors.tolist()]

    def metadata(self) -> dict[str, str | int]:
        model_name = self.config.model_name
        if Path(model_name).exists():
            model_name = Path(model_name).name
        return {
            "provider": "local",
            "model": model_name,
            "dimension": self.dimension,
            "device": self.config.device,
        }


@dataclass(slots=True)
class OpenAICompatibleEmbeddingConfig:
    api_key: str
    base_url: str
    model: str
    dimension: int = EMBEDDING_DIMENSION
    timeout: float = 60.0
    batch_size: int = 32


class OpenAICompatibleEmbeddingModel(EmbeddingModel):
    """Minimal OpenAI-style embeddings client."""

    def __init__(self, config: OpenAICompatibleEmbeddingConfig) -> None:
        self.config = config

    @property
    def dimension(self) -> int:
        return self.config.dimension

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        batch_size = max(1, self.config.batch_size)
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset : offset + batch_size]
            payload = self._request_embeddings(batch)
            vectors.extend(_extract_embeddings(payload))
        return vectors

    def metadata(self) -> dict[str, str | int]:
        return {
            "provider": "openai",
            "model": self.config.model,
            "dimension": self.dimension,
            "base_url": self.config.base_url,
        }

    def _request_embeddings(self, inputs: str | list[str]) -> dict:
        endpoint = self.config.base_url.rstrip("/") + "/embeddings"
        payload = {
            "model": self.config.model,
            "input": inputs,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.config.timeout) as response:
                raw_payload = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise EmbeddingError(f"Embedding request failed: HTTP {exc.code} {details}") from exc
        except error.URLError as exc:
            raise EmbeddingError(f"Embedding request failed: {exc.reason}") from exc

        try:
            return json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise EmbeddingError("Embedding response is not valid JSON") from exc


class FallbackEmbeddingModel(EmbeddingModel):
    def __init__(self, primary: EmbeddingModel, fallback: EmbeddingModel) -> None:
        self.primary = primary
        self.fallback = fallback

    @property
    def dimension(self) -> int:
        return self.primary.dimension

    def embed_text(self, text: str) -> list[float]:
        try:
            return self.primary.embed_text(text)
        except EmbeddingError:
            return self.fallback.embed_text(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.primary.embed_texts(texts)
        except EmbeddingError:
            return self.fallback.embed_texts(texts)

    def metadata(self) -> dict[str, str | int]:
        metadata = dict(self.primary.metadata())
        fallback_provider = self.fallback.metadata().get("provider", self.fallback.__class__.__name__)
        metadata["fallback_provider"] = str(fallback_provider)
        return metadata


def build_embedding_model() -> EmbeddingModel:
    provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if provider == "bge":
        try:
            primary = build_local_bge_embedding_model_from_env()
        except EmbeddingError:
            return HashingEmbeddingModel()
        if primary is None:
            return HashingEmbeddingModel()
        fallback = HashingEmbeddingModel(dimension=primary.dimension)
        return FallbackEmbeddingModel(primary=primary, fallback=fallback)
    if provider != "openai":
        return HashingEmbeddingModel()

    primary = build_remote_embedding_model_from_env()
    if primary is None:
        return HashingEmbeddingModel()
    fallback = HashingEmbeddingModel(dimension=primary.dimension)
    return FallbackEmbeddingModel(primary=primary, fallback=fallback)


def build_remote_embedding_model_from_env() -> OpenAICompatibleEmbeddingModel | None:
    api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        return None

    base_url = os.getenv(
        "EMBEDDING_BASE_URL",
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    ).strip()
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
    timeout = float(os.getenv("EMBEDDING_TIMEOUT", os.getenv("OPENAI_TIMEOUT", "60")))
    dimension = int(os.getenv("OPENAI_EMBEDDING_DIMENSION", str(EMBEDDING_DIMENSION)))
    batch_size = int(os.getenv("OPENAI_EMBEDDING_BATCH_SIZE", "32"))
    return OpenAICompatibleEmbeddingModel(
        OpenAICompatibleEmbeddingConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            dimension=dimension,
            timeout=timeout,
            batch_size=batch_size,
        )
    )


def build_local_bge_embedding_model_from_env() -> LocalSentenceTransformerEmbeddingModel | None:
    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip()
    if not model_name:
        return None
    dimension = int(os.getenv("LOCAL_EMBEDDING_DIMENSION", "384"))
    batch_size = int(os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", "32"))
    device = os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu").strip() or "cpu"
    normalize = _as_bool(os.getenv("LOCAL_EMBEDDING_NORMALIZE", "true"))
    return LocalSentenceTransformerEmbeddingModel(
        LocalSentenceTransformerConfig(
            model_name=model_name,
            dimension=dimension,
            device=device,
            normalize=normalize,
            batch_size=batch_size,
        )
    )


def _extract_embedding(payload: dict) -> list[float]:
    return _extract_embeddings(payload)[0]


def _extract_embeddings(payload: dict) -> list[list[float]]:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise EmbeddingError("Embedding response does not contain data")
    vectors: list[list[float]] = []
    for item in data:
        if not isinstance(item, dict):
            raise EmbeddingError("Embedding response item has invalid format")
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingError("Embedding response does not contain embedding values")
        try:
            vectors.append([float(value) for value in embedding])
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Embedding response contains non-numeric values") from exc
    return vectors


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    buffer: list[str] = []
    for char in text.lower():
        if char.isalnum():
            buffer.append(char)
        else:
            if buffer:
                tokens.append("".join(buffer))
                buffer.clear()
    if buffer:
        tokens.append("".join(buffer))
    return tokens


def _load_sentence_transformer_class():
    try:
        module = importlib.import_module("sentence_transformers")
    except ModuleNotFoundError as exc:
        raise EmbeddingError(
            "sentence-transformers is not installed. Install sentence-transformers, transformers, and torch to use LOCAL_EMBEDDING_MODEL."
        ) from exc
    sentence_transformer_cls = getattr(module, "SentenceTransformer", None)
    if sentence_transformer_cls is None:
        raise EmbeddingError("sentence_transformers.SentenceTransformer is unavailable")
    return sentence_transformer_cls


def _as_bool(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off"}
