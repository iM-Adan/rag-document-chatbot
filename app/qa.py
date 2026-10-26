"""RetrievalQA chain on top of a local Hugging Face model."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

from app.config import settings
from app.retriever import MultiQueryRetriever
from app.vectorstore import count_chunks, get_vectorstore

NO_ANSWER = "I don't know based on the provided documents."

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Answer the question using only the context below. "
        f'If the answer is not in the context, reply "{NO_ANSWER}"\n\n'
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)


class IndexEmpty(RuntimeError):
    """Raised when a question is asked before any documents were ingested."""


def build_llm() -> HuggingFacePipeline:
    pipeline_kwargs: Dict[str, Any] = {
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": False,
    }
    if settings.llm_task == "text-generation":
        pipeline_kwargs["return_full_text"] = False
    return HuggingFacePipeline.from_model_id(
        model_id=settings.llm_model,
        task=settings.llm_task,
        device=settings.device,
        pipeline_kwargs=pipeline_kwargs,
    )


class RAGService:
    def __init__(self) -> None:
        self.llm = build_llm()
        self.store = get_vectorstore()

    def chunk_count(self) -> int:
        return count_chunks(self.store)

    def ask(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        if self.chunk_count() == 0:
            raise IndexEmpty("The vector index is empty. Run ingestion first.")

        retriever = MultiQueryRetriever(
            vectorstore=self.store,
            llm=self.llm,
            k=top_k or settings.top_k,
        )
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
        result = chain.invoke({"query": question})

        sources: List[Dict[str, Any]] = [
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "snippet": d.page_content[:240],
            }
            for d in result.get("source_documents", [])
        ]
        return {"answer": result["result"].strip(), "sources": sources}
