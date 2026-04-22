from __future__ import annotations

import re
from dataclasses import dataclass

from config import DEFAULT_TOP_K, MAX_CONTEXT_CHUNKS, MAX_EVIDENCE_ITEMS
from src.data.schemas import ChunkRecord
from src.llm.base import AnswerGenerator
from src.llm.prompting import build_prompt
from src.retrieval.retriever import RetrievedChunk, Retriever


@dataclass
class RAGResponse:
    answer: str
    sources: list[str]
    evidence: list[str]
    retrieved_chunks: list[RetrievedChunk]
    prompt: str

    def to_display_string(self) -> str:
        sources_block = "\n".join(f"- {source}" for source in self.sources) or "- None"
        evidence_block = (
            "\n".join(f"{index}. {item}" for index, item in enumerate(self.evidence, 1))
            or "1. insufficient information"
        )
        return (
            f"Answer:\n{self.answer}\n\n"
            f"Sources:\n{sources_block}\n\n"
            f"Evidence:\n{evidence_block}"
        )


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: AnswerGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    def answer_question(self, question: str, top_k: int = DEFAULT_TOP_K) -> RAGResponse:
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        usable_chunks = _select_usable_chunks(retrieved)
        prompt = build_prompt(question, usable_chunks)

        if not usable_chunks:
            return RAGResponse(
                answer="insufficient information",
                sources=[],
                evidence=[],
                retrieved_chunks=retrieved,
                prompt=prompt,
            )

        raw_output = self.generator.generate(question, usable_chunks, prompt)
        answer, sources, evidence = _normalize_generation_output(raw_output, usable_chunks)
        if answer == "insufficient information":
            return RAGResponse(
                answer=answer,
                sources=[],
                evidence=[],
                retrieved_chunks=retrieved,
                prompt=prompt,
            )
        return RAGResponse(
            answer=answer,
            sources=sources,
            evidence=evidence,
            retrieved_chunks=retrieved,
            prompt=prompt,
        )


def _select_usable_chunks(retrieved: list[RetrievedChunk]) -> list[ChunkRecord]:
    if not retrieved:
        return []

    best_score = retrieved[0].score
    threshold = best_score * 0.78 if best_score > 0 else best_score
    selected: list[ChunkRecord] = []
    seen_ids: set[str] = set()
    for item in retrieved:
        if item.score < threshold and selected:
            break
        if item.chunk.chunk_id in seen_ids:
            continue
        selected.append(item.chunk)
        seen_ids.add(item.chunk.chunk_id)
        if len(selected) >= MAX_CONTEXT_CHUNKS:
            break
    return selected


def _normalize_generation_output(
    raw_output: str, chunks: list[ChunkRecord]
) -> tuple[str, list[str], list[str]]:
    parsed = _parse_structured_output(raw_output)
    if parsed is None:
        answer = raw_output.strip() or "insufficient information"
        if answer == "insufficient information":
            return answer, [], []
        return answer, _collect_sources(chunks), _build_evidence(chunks, answer)

    answer, parsed_sources, parsed_evidence = parsed
    normalized_answer = answer.strip() or "insufficient information"
    if normalized_answer == "insufficient information":
        return normalized_answer, [], []

    sources = _filter_sources(parsed_sources, chunks) or _collect_sources(chunks)
    evidence = _filter_evidence(parsed_evidence, chunks) or _build_evidence(chunks, normalized_answer)
    return normalized_answer, sources, evidence


def _parse_structured_output(raw_output: str) -> tuple[str, list[str], list[str]] | None:
    pattern = re.compile(
        r"Answer:\s*(?P<answer>.*?)\s*Sources:\s*(?P<sources>.*?)\s*Evidence:\s*(?P<evidence>.*)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(raw_output.strip())
    if not match:
        return None
    answer = match.group("answer").strip()
    sources = _parse_bullets(match.group("sources"))
    evidence = _parse_numbered_items(match.group("evidence"))
    return answer, sources, evidence


def _parse_bullets(block: str) -> list[str]:
    items: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items


def _parse_numbered_items(block: str) -> list[str]:
    items: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        match = re.match(r"\d+\.\s+(.*)", stripped)
        if match:
            items.append(match.group(1).strip())
    return items


def _collect_sources(chunks: list[ChunkRecord]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk in chunks:
        source = f"{chunk.title} ({chunk.source}, {chunk.date})"
        if source not in seen:
            seen.add(source)
            ordered.append(source)
    return ordered


def _filter_sources(candidate_sources: list[str], chunks: list[ChunkRecord]) -> list[str]:
    valid_sources = _collect_sources(chunks)
    valid_titles = {chunk.title for chunk in chunks}
    accepted: list[str] = []
    for source in candidate_sources:
        if source in valid_sources or any(title in source for title in valid_titles):
            accepted.append(source)
    return accepted


def _filter_evidence(candidate_evidence: list[str], chunks: list[ChunkRecord]) -> list[str]:
    accepted: list[str] = []
    for item in candidate_evidence:
        if _evidence_matches_chunks(item, chunks):
            accepted.append(item)
        if len(accepted) >= MAX_EVIDENCE_ITEMS:
            break
    return accepted


def _evidence_matches_chunks(evidence: str, chunks: list[ChunkRecord]) -> bool:
    evidence_terms = set(_tokenize(evidence))
    if not evidence_terms:
        return False
    for chunk in chunks:
        if evidence in chunk.text:
            return True
        for sentence in _split_sentences(chunk.text):
            sentence_terms = set(_tokenize(sentence))
            if len(evidence_terms & sentence_terms) >= 2:
                return True
    return False


def _build_evidence(chunks: list[ChunkRecord], answer: str) -> list[str]:
    if answer == "insufficient information":
        return []

    evidence: list[str] = []
    answer_terms = set(_tokenize(answer))
    for chunk in chunks:
        for sentence in _split_sentences(chunk.text):
            sentence_terms = set(_tokenize(sentence))
            if answer_terms & sentence_terms:
                evidence.append(sentence)
                if len(evidence) >= MAX_EVIDENCE_ITEMS:
                    return evidence
    return evidence


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
