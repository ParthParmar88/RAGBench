"""Ingestion idempotency and chunk provenance.

The database and both API calls are faked; what's under test is the decision
logic — when to skip, when to replace, and what provenance gets attached.
"""
import pytest

from app.services import embeddings, ingestion, pdf_extractor, vector_store
from app.services.ingestion import EmptyDocumentError, ingest_pdf
from app.services.pdf_extractor import PageSpan

PDF = b"%PDF-fake-bytes"
OTHER_PDF = b"%PDF-different-bytes"


class FakeSession:
    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = 1


class FakeDocument:
    """Stands in for a previously ingested row."""

    def __init__(self, title, sha, chunk_count=7, page_count=3):
        self.id = 1
        self.title = title
        self.content_sha256 = sha
        self.chunk_count = chunk_count
        self.page_count = page_count


@pytest.fixture
def env(monkeypatch):
    """Fake the PDF, the embedding API, and the store. Records what happened."""
    log = {"embed_calls": 0, "deleted": [], "stored": None}

    def install(text="hello world", spans=None, existing=None):
        monkeypatch.setattr(
            pdf_extractor, "extract_document",
            lambda _: (text, spans if spans is not None else [PageSpan(1, 0, len(text))]),
        )

        def fake_embed(texts):
            log["embed_calls"] += 1
            return [[0.0] * 4 for _ in texts]

        monkeypatch.setattr(embeddings, "embed_texts", fake_embed)
        monkeypatch.setattr(vector_store, "get_document", lambda db, title: existing)
        monkeypatch.setattr(
            vector_store, "delete_source",
            lambda db, title: log["deleted"].append(title) or 1,
        )

        def fake_store(db, document, chunks, vectors):
            log["stored"] = chunks
            return len(chunks)

        monkeypatch.setattr(vector_store, "add_knowledge_units", fake_store)
        return log

    return install


def test_new_document_is_created(env):
    log = env()
    result = ingest_pdf(FakeSession(), "paper.pdf", PDF)
    assert result.status == "created"
    assert result.chunks_added == 1
    assert log["embed_calls"] == 1


def test_identical_reupload_is_a_no_op(env):
    """The core fix: same name + same bytes must not re-embed or duplicate.

    Before this, a second upload appended a full copy of every chunk, so the
    retriever could return the same passage twice under different ids.
    """
    sha = ingestion._sha256(PDF)
    log = env(existing=FakeDocument("paper.pdf", sha, chunk_count=7))

    result = ingest_pdf(FakeSession(), "paper.pdf", PDF)

    assert result.status == "unchanged"
    assert result.chunks_added == 7  # reports what's already stored
    assert log["embed_calls"] == 0  # no spend
    assert log["deleted"] == []  # nothing destroyed
    assert log["stored"] is None  # nothing written


def test_changed_bytes_replace_the_old_chunks(env):
    """Same filename, different content: drop the stale chunks, don't append."""
    log = env(existing=FakeDocument("paper.pdf", ingestion._sha256(OTHER_PDF)))

    result = ingest_pdf(FakeSession(), "paper.pdf", PDF)

    assert result.status == "replaced"
    assert log["deleted"] == ["paper.pdf"]
    assert log["embed_calls"] == 1


def test_checksum_distinguishes_content_not_filename(env):
    """Identical bytes under a *new* name are still a fresh ingest."""
    log = env(existing=None)
    result = ingest_pdf(FakeSession(), "copy.pdf", PDF)
    assert result.status == "created"
    assert log["embed_calls"] == 1


def test_pdf_without_extractable_text_is_rejected(env):
    env(text="   ", spans=[])
    with pytest.raises(EmptyDocumentError):
        ingest_pdf(FakeSession(), "scanned.pdf", PDF)


def test_chunks_carry_sequential_indices(env):
    text = "a" * 3000  # > CHUNK_SIZE, so several chunks
    log = env(text=text, spans=[PageSpan(1, 0, 3000)])

    ingest_pdf(FakeSession(), "long.pdf", PDF)

    stored = log["stored"]
    assert len(stored) > 1
    assert [c.chunk_index for c in stored] == list(range(len(stored)))


def test_chunks_carry_the_page_they_start_on(env):
    page_one, page_two = "a" * 1500, "b" * 1500
    text = page_one + "\n\n" + page_two
    spans = [PageSpan(1, 0, 1500), PageSpan(2, 1502, 3002)]
    log = env(text=text, spans=spans)

    ingest_pdf(FakeSession(), "two-page.pdf", PDF)

    pages = [c.page for c in log["stored"]]
    assert pages[0] == 1
    assert 2 in pages
    assert all(p in (1, 2) for p in pages)


def test_page_count_reflects_pages_with_text(env):
    log = env(text="a\n\nb", spans=[PageSpan(1, 0, 1), PageSpan(3, 3, 4)])
    result = ingest_pdf(FakeSession(), "gappy.pdf", PDF)
    assert result.page_count == 2
