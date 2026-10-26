from pathlib import Path

from app import ingest
from langchain_core.documents import Document


def test_load_directory_uses_relative_source_paths_for_same_named_files(tmp_path):
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    (tmp_path / "first" / "guide.txt").write_text("First guide", encoding="utf-8")
    (tmp_path / "second" / "guide.txt").write_text("Second guide", encoding="utf-8")

    documents = ingest.load_directory(str(tmp_path))

    assert [document.metadata["source"] for document in documents] == [
        "first/guide.txt",
        "second/guide.txt",
    ]


def test_load_pdf_warns_when_a_page_has_no_extractable_text(tmp_path, monkeypatch, caplog):
    class EmptyPage:
        def extract_text(self):
            return ""

    class FakePdfReader:
        def __init__(self, _path):
            self.pages = [EmptyPage()]

    class FakePlumber:
        pages = [EmptyPage()]

        def close(self):
            pass

    monkeypatch.setattr(ingest, "PdfReader", FakePdfReader)
    monkeypatch.setattr(ingest.pdfplumber, "open", lambda _path: FakePlumber())

    documents = ingest.load_pdf(Path(tmp_path / "scan.pdf"), min_chars=1)

    assert documents == []
    assert "No text extracted from scan.pdf page 1; page skipped" in caplog.text
    assert "may require OCR" in caplog.text


def test_build_index_removes_stale_ids_after_adding_current_chunks(monkeypatch):
    class FakeStore:
        def __init__(self):
            self.ids = ["removed.txt:old-hash", "current.txt:old-hash"]
            self.deleted = []

        def get(self, include):
            assert include == ["metadatas"]
            return {"ids": list(self.ids)}

        def add_documents(self, documents, ids):
            assert len(documents) == len(ids)
            self.ids.extend(ids)

        def delete(self, ids):
            self.deleted.extend(ids)
            self.ids = [item for item in self.ids if item not in ids]

    store = FakeStore()
    document = Document(page_content="current content", metadata={"source": "current.txt"})
    monkeypatch.setattr(ingest, "load_directory", lambda _directory: [document])
    monkeypatch.setattr(ingest, "split_documents", lambda loaded: loaded)
    monkeypatch.setattr(ingest, "get_vectorstore", lambda: store)

    count = ingest.build_index(data_dir="unused")

    expected_id = f"current.txt:1:{ingest.content_hash('current content')}"
    assert count == 1
    assert set(store.deleted) == {"removed.txt:old-hash", "current.txt:old-hash"}
    assert store.ids == [expected_id]


def test_build_index_keeps_identical_chunks_from_different_pages(monkeypatch):
    class FakeStore:
        def __init__(self):
            self.ids = []

        def get(self, include):
            return {"ids": list(self.ids)}

        def add_documents(self, documents, ids):
            self.ids.extend(ids)

    store = FakeStore()
    documents = [
        Document(page_content="repeated page text", metadata={"source": "guide.pdf", "page": page})
        for page in (1, 2)
    ]
    monkeypatch.setattr(ingest, "load_directory", lambda _directory: documents)
    monkeypatch.setattr(ingest, "split_documents", lambda loaded: loaded)
    monkeypatch.setattr(ingest, "get_vectorstore", lambda: store)

    count = ingest.build_index(data_dir="unused")

    digest = ingest.content_hash("repeated page text")
    assert count == 2
    assert set(store.ids) == {f"guide.pdf:1:{digest}", f"guide.pdf:2:{digest}"}


def test_build_index_does_not_delete_existing_index_when_no_chunks(monkeypatch):
    class FakeStore:
        def get(self, include):
            return {"ids": ["existing-id"]}

    store = FakeStore()
    monkeypatch.setattr(ingest, "load_directory", lambda _directory: [])
    monkeypatch.setattr(ingest, "split_documents", lambda loaded: loaded)
    monkeypatch.setattr(ingest, "get_vectorstore", lambda: store)

    try:
        ingest.build_index(data_dir="unused")
    except ValueError as exc:
        assert "refusing to remove the existing index" in str(exc)
    else:
        raise AssertionError("Expected ingestion to fail rather than clear an empty index")
