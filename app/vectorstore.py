"""Embeddings (MiniLM) and the persistent ChromaDB vector store."""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def reset_vectorstore() -> Chroma:
    store = get_vectorstore()
    store.delete_collection()
    return get_vectorstore()


def count_chunks(store: Chroma) -> int:
    return store._collection.count()
