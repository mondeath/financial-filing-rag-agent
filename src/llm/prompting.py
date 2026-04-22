from src.data.schemas import ChunkRecord


SYSTEM_INSTRUCTION = (
    "You are a rigorous financial RAG research assistant. "
    "You must answer ONLY using the provided context chunks. "
    "Do not use outside knowledge. Do not hallucinate. "
    "If the context is insufficient, output 'insufficient information' in the Answer section. "
    "You must produce three sections exactly: Answer, Sources, Evidence. "
    "Sources must cite the supporting chunk source entries. "
    "Evidence must quote or closely restate supporting text that is directly aligned with the retrieved chunks. "
    "Be concise, factual, and structured."
)

OUTPUT_FORMAT = """Use exactly this format:

Answer:
<concise grounded answer, or 'insufficient information'>

Sources:
- <title> (<source>, <date>)

Evidence:
1. <evidence sentence aligned with a context chunk>
2. <optional second evidence sentence>
"""


def build_context_block(chunks: list[ChunkRecord]) -> str:
    sections: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        sections.append(
            "\n".join(
                [
                    f"[Chunk {index}]",
                    f"Chunk ID: {chunk.chunk_id}",
                    f"Title: {chunk.title}",
                    f"Source: {chunk.source}",
                    f"Date: {chunk.date}",
                    chunk.text,
                ]
            )
        )
    return "\n\n".join(sections)


def build_prompt(question: str, chunks: list[ChunkRecord]) -> str:
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"{OUTPUT_FORMAT}\n\n"
        f"[Question]\n{question}\n\n"
        f"[Context]\n{build_context_block(chunks)}"
    )
