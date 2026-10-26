"""Document ingestion: PDF extraction (with fallback), chunking and indexing.

Run with:  python -m app.ingest --data-dir data --reset
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings
from app.utils import clean_text, content_hash
from app.vectorstore import get_vectorstore, reset_vectorstore

logger = logging.getLogger("rag.ingest")


def load_pdf(path: Path, min_chars: int = settings.min_page_chars) -> List[Document]:
    """Extract text page by page.

    pypdf is fast and used first. When a page yields too little text (layout-heavy
    pages, tables, odd encodings), pdfplumber is used for that page only so no
    page is silently dropped from the index.
    """
    reader = PdfReader(str(path))
    plumber = None
    docs: List[Document] = []
    fallback_pages = 0
    try:
        for idx, page in enumerate(reader.pages):
            try:
                text = clean_text(page.extract_text() or "")
            except Exception:  # corrupt page object
                text = ""
            extractor = "pypdf"

            if len(text) < min_chars:
                if plumber is None:
                    plumber = pdfplumber.open(str(path))
                try:
                    alt = clean_text(plumber.pages[idx].extract_text() or "")
                except Exception:
                    alt = ""
                if len(alt) > len(text):
                    text, extractor = alt, "pdfplumber"
                    fallback_pages += 1

            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.name, "page": idx + 1, "extractor": extractor},
                    )
                )
    finally:
        if plumber is not None:
            plumber.close()

    logger.info("%s: %d pages extracted (%d via pdfplumber fallback)", path.name, len(docs), fallback_pages)
    return docs


def load_text(path: Path) -> List[Document]:
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    return [Document(page_content=text, metadata={"source": path.name, "page": 1, "extractor": "text"})] if text else []


def load_directory(data_dir: str) -> List[Document]:
    root = Path(data_dir)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"})
    if not files:
        raise FileNotFoundError(f"No .pdf/.txt/.md files found in '{root}'")
    docs: List[Document] = []
    for path in files:
        docs.extend(load_pdf(path) if path.suffix.lower() == ".pdf" else load_text(path))
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def build_index(data_dir: str = settings.data_dir, reset: bool = False, batch_size: int = 128) -> int:
    docs = load_directory(data_dir)
    chunks = split_documents(docs)

    # Content-hash IDs make re-ingestion idempotent and drop exact duplicate chunks.
    unique, seen = [], set()
    for chunk in chunks:
        key = f"{chunk.metadata['source']}:{content_hash(chunk.page_content)}"
        if key not in seen:
            seen.add(key)
            unique.append((key, chunk))

    store = reset_vectorstore() if reset else get_vectorstore()
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        store.add_documents([c for _, c in batch], ids=[k for k, _ in batch])
        logger.info("Indexed %d/%d chunks", min(start + batch_size, len(unique)), len(unique))

    logger.info("Done: %d pages -> %d chunks (%d unique)", len(docs), len(chunks), len(unique))
    return len(unique)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build the vector index from local documents.")
    parser.add_argument("--data-dir", default=settings.data_dir)
    parser.add_argument("--reset", action="store_true", help="Drop the existing collection first")
    args = parser.parse_args()
    build_index(args.data_dir, reset=args.reset)


if __name__ == "__main__":
    main()
