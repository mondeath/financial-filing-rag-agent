from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
INDEX_DIR = BASE_DIR / "index"

RAW_DOCS_PATH = RAW_DIR / "finance_docs.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
JPM_10K_CHUNKS_PATH = PROCESSED_DIR / "jpm_2025_10k_chunks.jsonl"
INDEX_META_PATH = INDEX_DIR / "index_meta.json"
INDEX_STORE_PATH = INDEX_DIR / "index_store"
EVAL_CASES_PATH = EVAL_DIR / "eval_cases.json"
JPM_10K_EVAL_CASES_PATH = EVAL_DIR / "jpm_10k_eval_cases.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 4
EMBEDDING_DIMENSION = 256
MAX_CONTEXT_CHUNKS = 4
MAX_EVIDENCE_ITEMS = 2
