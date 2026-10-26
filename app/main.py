"""FastAPI application exposing the RAG pipeline."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.qa import IndexEmpty, RAGService
from app.schemas import AskRequest, AskResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["rag"] = RAGService()  # load models once at startup
    yield
    state.clear()


app = FastAPI(
    title="RAG Chatbot",
    description="Question answering over your own documents using local retrieval-augmented generation.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", indexed_chunks=state["rag"].chunk_count())


# Sync endpoint on purpose: FastAPI runs it in a worker thread, so CPU-bound
# model inference does not block the event loop.
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        return AskResponse(**state["rag"].ask(req.question, req.top_k))
    except IndexEmpty as exc:
        raise HTTPException(status_code=503, detail=str(exc))
