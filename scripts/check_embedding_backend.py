from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.embeddings import (  # noqa: E402
    FallbackEmbeddingModel,
    build_embedding_model,
    build_local_bge_embedding_model_from_env,
    build_remote_embedding_model_from_env,
)


def main() -> None:
    model = build_embedding_model()
    print("Selected embedding backend:")
    print(f"- {model.describe()}")

    if isinstance(model, FallbackEmbeddingModel):
        print("- mode: remote embedding with local hashing fallback")
    else:
        print("- mode: local hashing baseline")

    remote_model = build_remote_embedding_model_from_env()
    if remote_model is None:
        print("- strict remote config: missing")
        print("  Set EMBEDDING_PROVIDER, EMBEDDING_API_KEY, EMBEDDING_BASE_URL, and OPENAI_EMBEDDING_MODEL.")
    else:
        print("- strict remote config: present")
        print(f"- remote target: {remote_model.describe()}")

    try:
        local_bge_model = build_local_bge_embedding_model_from_env()
    except Exception as exc:
        print("- strict local BGE config: invalid")
        print(f"  {exc}")
    else:
        if local_bge_model is None:
            print("- strict local BGE config: missing")
            print("  Set EMBEDDING_PROVIDER=bge and optionally LOCAL_EMBEDDING_MODEL.")
        else:
            print("- strict local BGE config: present")
            print(f"- local target: {local_bge_model.describe()}")


if __name__ == "__main__":
    main()
