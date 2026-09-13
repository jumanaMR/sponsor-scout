"""Tests for NewPostingWatcher — the diff step behind a job alert.

The behaviour that matters here is idempotency: a scheduled job that runs
every six hours must not email you the same twelve postings every time.
"""

import json

import pytest

from sponsor_scout.pipeline import NewPostingWatcher, SponsorshipRAG
from sponsor_scout.sample_data import SAMPLE_POSTINGS

NEW_POSTING = {
    "id": "syn-999",
    "company": "Pinehollow AI",
    "country": "Australia",
    "role_title": "AI Engineer",
    "source": "test",
    "date_posted": "2026-09-13",
    "text": (
        "Pinehollow AI is an approved 482 sponsor hiring an offshore AI "
        "engineer for our Perth office, with relocation support."
    ),
}


def build(postings):
    index = SponsorshipRAG(n_components=50)
    index.add_documents(postings, chunk_size=60, overlap=15)
    index.build_index()
    return index


@pytest.fixture
def state_path(tmp_path):
    return str(tmp_path / "seen.json")


def test_first_check_reports_everything(state_path):
    watcher = NewPostingWatcher(state_path=state_path)
    new = watcher.check(build(SAMPLE_POSTINGS), SAMPLE_POSTINGS)
    assert len(new) == len(SAMPLE_POSTINGS)


def test_second_check_reports_nothing(state_path):
    rag = build(SAMPLE_POSTINGS)
    watcher = NewPostingWatcher(state_path=state_path)
    watcher.check(rag, SAMPLE_POSTINGS)
    assert watcher.check(rag, SAMPLE_POSTINGS).empty


def test_only_genuinely_new_postings_are_reported(state_path):
    watcher = NewPostingWatcher(state_path=state_path)
    watcher.check(build(SAMPLE_POSTINGS), SAMPLE_POSTINGS)

    corpus = SAMPLE_POSTINGS + [NEW_POSTING]
    new = watcher.check(build(corpus), corpus)
    assert list(new["doc_id"]) == ["syn-999"]
    assert list(new["company"]) == ["Pinehollow AI"]


def test_state_survives_a_new_process(state_path):
    """A scheduled job is a fresh process every run — seen state has to
    come off disk, not out of memory."""
    rag = build(SAMPLE_POSTINGS)
    NewPostingWatcher(state_path=state_path).check(rag, SAMPLE_POSTINGS)

    reloaded = NewPostingWatcher(state_path=state_path)
    assert reloaded.check(rag, SAMPLE_POSTINGS).empty

    with open(state_path) as handle:
        assert len(json.load(handle)["seen_ids"]) == len(SAMPLE_POSTINGS)


def test_mark_seen_false_is_a_dry_run(state_path):
    rag = build(SAMPLE_POSTINGS)
    watcher = NewPostingWatcher(state_path=state_path)
    first = watcher.check(rag, SAMPLE_POSTINGS, mark_seen=False)
    second = watcher.check(rag, SAMPLE_POSTINGS, mark_seen=False)
    assert len(first) == len(second) == len(SAMPLE_POSTINGS)


def test_digest_is_readable_and_says_when_nothing_is_new(state_path):
    rag = build(SAMPLE_POSTINGS)
    watcher = NewPostingWatcher(state_path=state_path)

    digest = watcher.notify_digest(rag, SAMPLE_POSTINGS)
    assert "new posting" in digest
    assert "Nimbus Data Labs" in digest

    assert watcher.notify_digest(rag, SAMPLE_POSTINGS) == (
        "No new postings since your last check."
    )


def test_digest_scores_against_the_cv_when_one_is_loaded(state_path, tmp_path):
    rag = build(SAMPLE_POSTINGS)
    rag.load_cv("AI engineer with RAG and vector search experience.")
    watcher = NewPostingWatcher(state_path=state_path)
    new = watcher.check(rag, SAMPLE_POSTINGS)
    assert new["match_score"].notna().all()
