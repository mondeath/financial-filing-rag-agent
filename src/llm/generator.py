import json
import os
import re
from dataclasses import dataclass
from urllib import error, request

from config import MAX_EVIDENCE_ITEMS
from src.data.schemas import ChunkRecord
from src.llm.base import AnswerGenerator


class GroundedExtractiveGenerator(AnswerGenerator):
    """A local fallback generator that only uses retrieved context."""

    min_sentence_score = 0.12

    def generate(self, question: str, chunks: list[ChunkRecord], prompt: str) -> str:
        del prompt
        if not chunks:
            return "insufficient information"

        candidate_sentences = _collect_candidate_sentences(question, chunks)
        if not candidate_sentences:
            return "insufficient information"

        ranked = sorted(candidate_sentences, key=lambda item: item["score"], reverse=True)
        best_score = float(ranked[0]["score"])
        score_floor = max(self.min_sentence_score, best_score * 0.80)
        top_items = _dedupe_by_source(
            [item for item in ranked if float(item["score"]) >= score_floor]
        )[:MAX_EVIDENCE_ITEMS]
        if not top_items:
            return "insufficient information"

        answer_sentences = [str(item["sentence"]) for item in top_items]
        sources = _format_sources(top_items)
        evidence = [str(item["sentence"]) for item in top_items]
        return _format_structured_answer(
            answer=" ".join(answer_sentences),
            sources=sources,
            evidence=evidence,
        )


class LLMGenerationError(RuntimeError):
    pass


@dataclass
class OpenAICompatibleConfig:
    api_key: str
    base_url: str
    model: str
    timeout: float = 60.0


class OpenAICompatibleGenerator(AnswerGenerator):
    """Minimal OpenAI-style chat completions client."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config

    def generate(self, question: str, chunks: list[ChunkRecord], prompt: str) -> str:
        del question, chunks
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
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
            raise LLMGenerationError(f"LLM request failed: HTTP {exc.code} {details}") from exc
        except error.URLError as exc:
            raise LLMGenerationError(f"LLM request failed: {exc.reason}") from exc

        try:
            parsed = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise LLMGenerationError("LLM response is not valid JSON") from exc
        return _extract_openai_text(parsed)


class FallbackAnswerGenerator(AnswerGenerator):
    def __init__(self, primary: AnswerGenerator, fallback: AnswerGenerator) -> None:
        self.primary = primary
        self.fallback = fallback

    def generate(self, question: str, chunks: list[ChunkRecord], prompt: str) -> str:
        try:
            return self.primary.generate(question, chunks, prompt)
        except LLMGenerationError:
            return self.fallback.generate(question, chunks, prompt)


def build_answer_generator() -> AnswerGenerator:
    local_generator = GroundedExtractiveGenerator()
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider != "openai":
        return local_generator

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return local_generator

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
    real_generator = OpenAICompatibleGenerator(
        OpenAICompatibleConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
    )
    return FallbackAnswerGenerator(primary=real_generator, fallback=local_generator)


def _extract_openai_text(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMGenerationError("LLM response does not contain choices")

    first = choices[0]
    if not isinstance(first, dict):
        raise LLMGenerationError("LLM response choice has invalid format")

    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text.strip())
            if text_parts:
                return "\n".join(text_parts).strip()

    text = first.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    raise LLMGenerationError("LLM response does not contain text content")


def _collect_candidate_sentences(
    question: str, chunks: list[ChunkRecord]
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    question_terms = set(_tokenize(question))
    for chunk_rank, chunk in enumerate(chunks):
        metadata_terms = set(
            _tokenize(
                " ".join(
                    [
                        chunk.title,
                        chunk.section,
                        chunk.primary_topic,
                        chunk.secondary_topic,
                    ]
                )
            )
        )
        for sentence in _split_sentences(chunk.text):
            results.append(
                {
                    "sentence": sentence,
                    "score": _sentence_score(sentence, question_terms, metadata_terms)
                    + _rank_bonus(chunk_rank),
                    "chunk": chunk,
                }
            )
    return results


def _sentence_score(
    sentence: str,
    question_terms: set[str],
    metadata_terms: set[str],
) -> float:
    sentence_terms = set(_tokenize(sentence))
    useful_query_terms = question_terms - _stopwords()
    if not useful_query_terms:
        return 0.0
    sentence_overlap = len(sentence_terms & useful_query_terms)
    metadata_overlap = len(metadata_terms & useful_query_terms)
    if sentence_overlap == 0 and metadata_overlap == 0:
        return 0.0
    score = (
        sentence_overlap / max(len(useful_query_terms), 1)
        + 0.25 * metadata_overlap / max(len(useful_query_terms), 1)
    )
    return max(0.0, score - _length_penalty(sentence))


def _length_penalty(sentence: str) -> float:
    extra_chars = max(len(sentence) - 700, 0)
    return min(extra_chars * 0.0004, 0.25)


def _rank_bonus(chunk_rank: int) -> float:
    return max(0.0, 0.05 - chunk_rank * 0.01)


def _format_sources(items: list[dict[str, object]]) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for item in items:
        chunk = item["chunk"]
        if not isinstance(chunk, ChunkRecord):
            continue
        source = f"{chunk.title} ({chunk.source}, {chunk.date})"
        if source not in seen:
            sources.append(source)
            seen.add(source)
    return sources


def _dedupe_by_source(items: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in items:
        chunk = item["chunk"]
        if not isinstance(chunk, ChunkRecord):
            continue
        source_key = f"{chunk.title}|{chunk.source}|{chunk.date}"
        if source_key in seen:
            continue
        deduped.append(item)
        seen.add(source_key)
    return deduped


def _format_structured_answer(
    answer: str,
    sources: list[str],
    evidence: list[str],
) -> str:
    sources_block = "\n".join(f"- {source}" for source in sources) or "- None"
    evidence_block = "\n".join(
        f"{index}. {item}" for index, item in enumerate(evidence, 1)
    )
    return (
        f"Answer:\n{answer}\n\n"
        f"Sources:\n{sources_block}\n\n"
        f"Evidence:\n{evidence_block}"
    )


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def _tokenize(text: str) -> list[str]:
    normalized = text.lower()
    raw_tokens = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", normalized)
    tokens: list[str] = []
    for token in raw_tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.extend(list(token))
            if len(token) >= 2:
                tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            tokens.append(token)
    return [token for token in tokens if token.strip()]


def _stopwords() -> set[str]:
    return {
        "a",
        "about",
        "and",
        "are",
        "as",
        "at",
        "chase",
        "describe",
        "does",
        "for",
        "how",
        "in",
        "is",
        "its",
        "jpmorgan",
        "main",
        "of",
        "or",
        "say",
        "says",
        "the",
        "to",
        "what",
        "with",
    }
