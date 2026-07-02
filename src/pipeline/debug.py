from src.pipeline.rag_pipeline import RAGResponse


def format_retrieval_debug(response: RAGResponse) -> str:
    if not response.retrieved_chunks:
        return "Retrieved Chunks:\n- None"

    lines = ["Retrieved Chunks:"]
    for index, item in enumerate(response.retrieved_chunks, 1):
        chunk = item.chunk
        query_type = item.rerank.query_type if item.rerank else "n/a"
        section_bonus = item.rerank.section_bonus if item.rerank else 0.0
        topic_bonus = item.rerank.topic_bonus if item.rerank else 0.0
        quality_bonus = item.rerank.quality_bonus if item.rerank else 0.0
        lines.append(
            (
                f"- {index}. score={item.score:.4f}, "
                f"embedding={item.embedding_score:.4f}, "
                f"query_type={query_type}, "
                f"section_bonus={section_bonus:.4f}, "
                f"topic_bonus={topic_bonus:.4f}, "
                f"quality_bonus={quality_bonus:.4f}, "
                f"section={chunk.section or 'n/a'}, "
                f"primary_topic={chunk.primary_topic or 'n/a'}, "
                f"secondary_topic={chunk.secondary_topic or 'n/a'}, "
                f"quality={chunk.quality or 'n/a'}, "
                f"title={chunk.title}"
            )
        )
    return "\n".join(lines)


def format_prompt_debug(response: RAGResponse) -> str:
    return f"Prompt:\n{response.prompt}"
