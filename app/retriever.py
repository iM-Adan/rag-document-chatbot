"""Multi-query retriever with rank fusion, de-duplication and context budgeting."""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import settings
from app.utils import (
    deduplicate,
    keyword_query,
    parse_queries,
    reciprocal_rank_fusion,
    trim_to_budget,
)

logger = logging.getLogger("rag.retriever")

REWRITE_PROMPT = (
    "Rewrite the question below in {n} different ways that keep the same meaning "
    "but use different wording. Write one rewrite per line and nothing else.\n\n"
    "Question: {question}\n\nRewrites:"
)


class MultiQueryRetriever(BaseRetriever):
    """Retrieve with several query variants, fuse the rankings, then clean the result.

    Steps:
      1. Build query variants (LLM paraphrases + a keyword-only variant).
      2. Run a similarity search per variant.
      3. Merge the ranked lists with Reciprocal Rank Fusion.
      4. Remove exact/near duplicate chunks.
      5. Keep the top-k chunks and trim them to a character budget for the LLM.
    """

    vectorstore: Any
    llm: Any = None
    k: int = settings.top_k
    fetch_k: int = settings.fetch_k
    num_variants: int = settings.num_query_variants
    dedup_threshold: float = settings.dedup_threshold
    max_context_chars: int = settings.max_context_chars

    def _generate_queries(self, question: str) -> List[str]:
        variants: List[str] = []
        if self.llm is not None and self.num_variants > 0:
            try:
                raw = self.llm.invoke(REWRITE_PROMPT.format(n=self.num_variants, question=question))
                variants = parse_queries(str(raw), question, self.num_variants)
            except Exception as exc:  # never let query rewriting break retrieval
                logger.warning("Query rewriting failed: %s", exc)

        keywords = keyword_query(question)
        if keywords and keywords != question.lower() and keywords not in variants:
            variants.append(keywords)

        return [question] + variants

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        queries = self._generate_queries(query)
        logger.info("Retrieving with %d queries: %s", len(queries), queries)

        ranked_lists = [
            self.vectorstore.similarity_search(q, k=self.fetch_k) for q in queries
        ]
        fused = reciprocal_rank_fusion(ranked_lists)
        unique = deduplicate(fused, threshold=self.dedup_threshold)[: self.k]
        return trim_to_budget(unique, self.max_context_chars)
