from pathlib import Path

from app import ingest


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
