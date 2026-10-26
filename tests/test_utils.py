from langchain_core.documents import Document

from app.utils import (
    clean_text,
    deduplicate,
    keyword_query,
    parse_queries,
    reciprocal_rank_fusion,
    trim_to_budget,
)


def doc(text):
    return Document(page_content=text)


def test_clean_text_repairs_hyphenation_and_whitespace():
    assert clean_text("infor-\nmation   retrieval\n\nworks") == "information retrieval works"


def test_deduplicate_removes_exact_and_near_duplicates():
    docs = [
        doc("The quick brown fox jumps over the lazy dog near the river bank"),
        doc("the quick brown fox jumps over the lazy dog near the river bank"),
        doc("The quick brown fox jumps over the lazy dog near the river"),
        doc("Completely different content about vector databases"),
    ]
    out = deduplicate(docs, threshold=0.85)
    assert len(out) == 2
    assert out[0] is docs[0]


def test_deduplicate_preserves_identical_text_from_distinct_sources():
    first = Document(page_content="shared text", metadata={"source": "first.txt", "page": 1})
    second = Document(page_content="shared text", metadata={"source": "second.txt", "page": 1})

    assert deduplicate([first, second]) == [first, second]


def test_rrf_promotes_documents_found_by_multiple_queries():
    a, b, c = doc("alpha"), doc("bravo"), doc("charlie")
    fused = reciprocal_rank_fusion([[a, b], [c, b]])
    assert fused[0].page_content == "bravo"
    assert {d.page_content for d in fused} == {"alpha", "bravo", "charlie"}


def test_rrf_preserves_identical_text_from_distinct_sources():
    first = Document(page_content="shared text", metadata={"source": "first.txt", "page": 1})
    second = Document(page_content="shared text", metadata={"source": "second.txt", "page": 1})

    fused = reciprocal_rank_fusion([[first], [second]])

    assert fused == [first, second]


def test_trim_to_budget_truncates_last_document():
    docs = [doc("x" * 500), doc("y" * 500)]
    out = trim_to_budget(docs, max_chars=700)
    assert sum(len(d.page_content) for d in out) == 700
    assert len(docs[1].page_content) == 500  # original untouched


def test_trim_to_budget_drops_tiny_leftover():
    out = trim_to_budget([doc("x" * 500), doc("y" * 500)], max_chars=550)
    assert len(out) == 1


def test_keyword_query_removes_stopwords():
    assert keyword_query("What is the refund policy for orders?") == "refund policy orders"


def test_parse_queries_cleans_bullets_and_duplicates():
    raw = "1. How do refunds work?\n- How do refunds work?\n- Explain the refund process\n\nok"
    out = parse_queries(raw, "How do refunds work?", limit=3)
    assert out == ["Explain the refund process"]
