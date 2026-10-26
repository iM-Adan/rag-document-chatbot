"""Central configuration, driven by environment variables (or a local .env file)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # Paths
    data_dir: str = os.getenv("DATA_DIR", "data")
    chroma_dir: str = os.getenv("CHROMA_DIR", "chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")

    # Models
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    llm_model: str = os.getenv("LLM_MODEL", "google/flan-t5-base")
    llm_task: str = os.getenv("LLM_TASK", "text2text-generation")
    max_new_tokens: int = _int("MAX_NEW_TOKENS", 256)
    device: int = _int("DEVICE", -1)  # -1 = CPU, 0 = first GPU

    # Chunking
    chunk_size: int = _int("CHUNK_SIZE", 800)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 150)
    min_page_chars: int = _int("MIN_PAGE_CHARS", 40)

    # Retrieval
    top_k: int = _int("TOP_K", 4)
    fetch_k: int = _int("FETCH_K", 8)
    num_query_variants: int = _int("NUM_QUERY_VARIANTS", 3)
    dedup_threshold: float = _float("DEDUP_THRESHOLD", 0.85)
    max_context_chars: int = _int("MAX_CONTEXT_CHARS", 2400)


settings = Settings()
