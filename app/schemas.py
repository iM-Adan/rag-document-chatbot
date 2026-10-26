from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Number of chunks to use as context")


class Source(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
