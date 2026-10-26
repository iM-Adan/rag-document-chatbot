"""Small, dependency-light helpers for text cleanup, dedup, fusion and context budgeting."""
from __future__ import annotations

import hashlib
import re
from typing import List, Sequence

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "in",
    "on", "for", "and", "or", "with", "what", "which", "who", "whom", "how", "why",
    "when", "where", "does", "do", "did", "this", "that", "these", "those", "about",
    "can", "could", "should", "would", "me", "my", "i", "you", "it", "its", "as",
    "at", "by", "from", "tell",
}


def clean_text(text: str) -> str:
    """Repair hyphenation across line breaks and collapse whitespace."""
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return _WS.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def content_hash(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def _tokens(text: str) -> set:
    return set(_TOKEN.findall(normalize_text(text)))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _document_origin(doc) -> tuple:
    metadata = doc.metadata
    return metadata.get("source"), metadata.get("page")


def deduplicate(docs: Sequence, threshold: float = 0.85) -> List:
    """Drop exact duplicates and near-duplicates (token-set Jaccard >= threshold).

    Duplicate text from different source pages remains separate so citations are
    not lost. Within one source page, order is preserved and the highest-ranked
    copy wins.
    """
    kept, seen_hashes, kept_docs, kept_tokens = [], set(), [], []
    for doc in docs:
        origin = _document_origin(doc)
        digest = (origin, content_hash(doc.page_content))
        if digest in seen_hashes:
            continue
        toks = _tokens(doc.page_content)
        if any(
            _document_origin(other) == origin and jaccard(toks, other_tokens) >= threshold
            for other, other_tokens in zip(kept_docs, kept_tokens)
        ):
            continue
        seen_hashes.add(digest)
        kept_docs.append(doc)
        kept_tokens.append(toks)
        kept.append(doc)
    return kept


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence], k: int = 60) -> List:
    """Merge several ranked result lists into one using Reciprocal Rank Fusion."""
    scores: dict = {}
    first_seen: dict = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = (_document_origin(doc), content_hash(doc.page_content))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            first_seen.setdefault(key, doc)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [first_seen[key] for key in ordered]


def trim_to_budget(docs: Sequence, max_chars: int, min_tail: int = 200) -> List:
    """Keep documents in order until the character budget is used up.

    The last document is truncated if enough budget remains to make it useful.
    """
    out, used = [], 0
    for doc in docs:
        remaining = max_chars - used
        if remaining <= 0:
            break
        text = doc.page_content
        if len(text) <= remaining:
            out.append(doc)
            used += len(text)
        elif remaining >= min_tail:
            clone = doc.model_copy(update={"page_content": text[:remaining].rstrip()})
            out.append(clone)
            break
        else:
            break
    return out


def keyword_query(question: str) -> str:
    """Strip stopwords to produce a keyword-style variant of a question."""
    words = [w for w in _TOKEN.findall(question.lower()) if w not in STOPWORDS]
    return " ".join(words)


def parse_queries(raw: str, original: str, limit: int) -> List[str]:
    """Parse LLM output (one rewrite per line) into a clean, unique list of queries."""
    seen = {normalize_text(original)}
    queries: List[str] = []
    for line in raw.splitlines():
        line = _BULLET.sub("", line).strip().strip('"')
        key = normalize_text(line)
        if len(line) < 5 or key in seen:
            continue
        seen.add(key)
        queries.append(line)
        if len(queries) >= limit:
            break
    return queries
