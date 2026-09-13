"""Tests for sentence-aware chunking with overlap."""

from sponsor_scout.pipeline import chunk_text, split_sentences

LONG = (
    "Nimbus Data Labs is hiring an AI Engineer for our Sydney ML platform team. "
    "We are an approved sponsor and can support a subclass 482 visa. "
    "You will work on RAG pipelines, vector search, and LLM evaluation. "
    "Three years of Python experience is required. "
    "We offer relocation support for offshore candidates."
)


def test_short_text_is_one_chunk():
    assert chunk_text("A single short sentence.", max_words=60) == [
        "A single short sentence."
    ]


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_long_text_splits_into_multiple_chunks():
    chunks = chunk_text(LONG, max_words=20, overlap_words=5)
    assert len(chunks) > 1


def test_consecutive_chunks_overlap():
    """The whole point of overlap: a sentence spanning a chunk boundary is
    still retrievable, because the tail of chunk N opens chunk N+1."""
    chunks = chunk_text(LONG, max_words=20, overlap_words=5)
    for first, second in zip(chunks, chunks[1:]):
        tail = first.split()[-5:]
        assert second.split()[: len(tail)] == tail


def test_zero_overlap_produces_disjoint_chunks():
    chunks = chunk_text(LONG, max_words=20, overlap_words=0)
    joined = " ".join(chunks).split()
    assert len(joined) == len(LONG.split())


def test_no_content_is_lost():
    """Every word of the source appears somewhere in the chunks."""
    chunks = chunk_text(LONG, max_words=20, overlap_words=5)
    blob = " ".join(chunks)
    for word in LONG.split():
        assert word in blob


def test_sentence_splitter_handles_terminators():
    assert split_sentences("One. Two! Three? Four") == [
        "One.",
        "Two!",
        "Three?",
        "Four",
    ]
