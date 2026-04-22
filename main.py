import argparse
from pathlib import Path

from config import CHUNKS_PATH, EVAL_CASES_PATH, INDEX_STORE_PATH, RAW_DOCS_PATH
from src.data.chunking import ChunkingConfig, build_chunks_file
from src.eval.evaluator import build_eval_report, load_eval_cases, run_eval_cases
from src.llm.generator import build_answer_generator
from src.pipeline.debug import format_prompt_debug, format_retrieval_debug
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.embeddings import HashingEmbeddingModel
from src.retrieval.index import IndexBuildResult, build_index_from_chunks
from src.retrieval.retriever import Retriever


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def handle_build_chunks(_: argparse.Namespace) -> None:
    _ensure_parent(CHUNKS_PATH)
    count = build_chunks_file(
        raw_docs_path=RAW_DOCS_PATH,
        output_path=CHUNKS_PATH,
        config=ChunkingConfig(),
    )
    print(f"Built {count} chunks at {CHUNKS_PATH}")


def handle_build_index(args: argparse.Namespace) -> None:
    _ensure_parent(INDEX_STORE_PATH)
    result: IndexBuildResult = build_index_from_chunks(
        chunks_path=args.chunks,
        output_path=INDEX_STORE_PATH,
        embedding_model=HashingEmbeddingModel(),
    )
    print(
        f"Built index backend={result.backend} chunks={result.chunk_count} "
        f"path={INDEX_STORE_PATH}"
    )


def handle_ask(args: argparse.Namespace) -> None:
    retriever = Retriever.load(
        index_path=INDEX_STORE_PATH,
        embedding_model=HashingEmbeddingModel(),
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=build_answer_generator(),
    )
    response = pipeline.answer_question(args.question, top_k=args.top_k)
    print(response.to_display_string())
    if args.debug:
        print()
        print(format_retrieval_debug(response))
    if args.show_prompt:
        print()
        print(format_prompt_debug(response))


def handle_eval(args: argparse.Namespace) -> None:
    retriever = Retriever.load(
        index_path=INDEX_STORE_PATH,
        embedding_model=HashingEmbeddingModel(),
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=build_answer_generator(),
    )
    cases = load_eval_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    results = run_eval_cases(pipeline=pipeline, cases=cases, top_k=args.top_k)
    print(build_eval_report(results))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Financial RAG Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_chunks = subparsers.add_parser("build-chunks")
    build_chunks.set_defaults(func=handle_build_chunks)

    build_index = subparsers.add_parser("build-index")
    build_index.add_argument(
        "--chunks",
        type=Path,
        default=CHUNKS_PATH,
        help="Path to chunk JSONL file used to build the retrieval index.",
    )
    build_index.set_defaults(func=handle_build_index)

    ask = subparsers.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=4)
    ask.add_argument(
        "--debug",
        action="store_true",
        help="Show retrieved chunks with rerank metadata.",
    )
    ask.add_argument(
        "--show-prompt",
        action="store_true",
        help="Show the final prompt passed to the answer generator.",
    )
    ask.set_defaults(func=handle_ask)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument(
        "--cases",
        type=Path,
        default=EVAL_CASES_PATH,
        help="Path to eval cases JSON file.",
    )
    evaluate.add_argument("--top-k", type=int, default=4)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.set_defaults(func=handle_eval)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
