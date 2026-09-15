"""Tests for the index, retrieval, recommender and CV matching.

These assert behaviour that must hold regardless of the embedding
backend — so they keep passing when TF-IDF+SVD is swapped for a neural
model (see the roadmap in the README).
"""

import pytest

from sponsor_scout.pipeline import SponsorshipRAG
from sponsor_scout.sample_data import SAMPLE_POSTINGS

CV = (
    "AI and Automation Engineer. RAG pipelines, vector search, chunking, LLM "
    "evaluation, Python, n8n workflow automation. Seeking a sponsored AI "
    "engineering role in Australia or the Netherlands."
)


@pytest.fixture
def rag():
    index = SponsorshipRAG(n_components=50)
    index.add_documents(SAMPLE_POSTINGS, chunk_size=60, overlap=15)
    index.build_index()
    return index


def test_index_covers_every_posting(rag):
    assert {c.doc_id for c in rag.chunks} == {d["id"] for d in SAMPLE_POSTINGS}
    assert rag.vectors.shape[0] == len(rag.chunks)


@pytest.mark.parametrize("n_postings", [1, 2, 3])
def test_a_tiny_corpus_still_builds(n_postings):
    """REGRESSION: with fewer than ~5 chunks, max_df=0.95 rounds down below
    min_df=1 and sklearn raises. A near-empty corpus is a real state — a
    freshly-seeded store, or a filter that matched one company — and it must
    not 500 the search endpoint."""
    index = SponsorshipRAG(n_components=50)
    index.add_documents(SAMPLE_POSTINGS[:n_postings])
    index.build_index()
    assert index.vectors.shape[0] >= 1
    assert len(index.query("visa sponsorship", top_k=1)) == 1


def test_query_before_build_is_an_error():
    empty = SponsorshipRAG()
    empty.add_documents(SAMPLE_POSTINGS[:2])
    with pytest.raises(ValueError):
        empty.query("anything")


def test_query_returns_scores_in_descending_order(rag):
    results = rag.query("visa sponsorship for AI engineers", top_k=6)
    assert len(results) == 6
    assert list(results["score"]) == sorted(results["score"], reverse=True)


def test_query_surfaces_the_obviously_relevant_posting(rag):
    results = rag.query("subclass 482 sponsorship Sydney AI engineer", top_k=3)
    assert "Nimbus Data Labs" in set(results["company"])


def test_recommend_requires_some_signal(rag):
    with pytest.raises(ValueError):
        rag.recommend_next()


def test_recommend_excludes_what_you_already_viewed(rag):
    rag.record_view("syn-001")
    rag.record_view("syn-003")
    recs = rag.recommend_next(top_k=20)
    assert "syn-001" not in set(recs["doc_id"])
    assert "syn-003" not in set(recs["doc_id"])


def test_sponsorship_boost_sinks_non_sponsoring_postings(rag):
    """A posting that clearly won't sponsor must not outrank one that
    will, on content similarity alone."""
    rag.record_view("syn-001")
    recs = rag.recommend_next(top_k=20)
    sponsoring = recs[recs["sponsorship_signal"] == "likely sponsoring"]["score"]
    refusing = recs[recs["sponsorship_signal"] == "likely NOT sponsoring"]["score"]
    assert sponsoring.min() > refusing.max()


def test_recommendations_explain_themselves(rag):
    rag.record_view("syn-001")
    recs = rag.recommend_next(top_k=5)
    assert recs["because_you_viewed"].notna().all()
    assert (recs["because_you_viewed"] == "Nimbus Data Labs").all()


def test_cv_match_requires_a_cv(rag):
    with pytest.raises(ValueError):
        rag.cv_match_report()


def test_cv_match_ranks_every_posting(rag):
    rag.load_cv(CV)
    report = rag.cv_match_report()
    assert len(report) == len(SAMPLE_POSTINGS)
    assert list(report["score"]) == sorted(report["score"], reverse=True)


def test_cv_only_cold_start_needs_no_view_history(rag):
    """cv_weight=1.0 is the cold-start path: recommend from the CV alone."""
    rag.load_cv(CV)
    recs = rag.recommend_next(cv_weight=1.0, top_k=5)
    assert len(recs) == 5
    assert (recs["because_you_viewed"] == "your CV").all()


def test_cv_weight_without_a_cv_is_rejected(rag):
    rag.record_view("syn-001")
    with pytest.raises(ValueError):
        rag.recommend_next(cv_weight=0.5)


def test_blending_a_cv_changes_the_ranking(rag):
    rag.record_view("syn-002")  # a Melbourne ML role that doesn't sponsor
    rag.load_cv(CV)
    views_only = rag.recommend_next(cv_weight=0.0, top_k=8)
    blended = rag.recommend_next(cv_weight=0.9, top_k=8)
    assert list(views_only["doc_id"]) != list(blended["doc_id"])
