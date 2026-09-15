"""Agent and store tests.

The behaviours that matter for something running unattended on a timer:
new means new across runs, a broken source doesn't kill the run, and the
digest says something truthful when there's nothing to report.
"""

from __future__ import annotations

import pytest

from sponsor_scout.agent import JobAgent
from sponsor_scout.sources.base import Posting, SourceAdapter, SourceError
from sponsor_scout.store import PostingStore


def make_posting(pid: str, text: str, company: str = "Acme", country: str = "Australia", **kw) -> Posting:
    return Posting(
        id=pid, company=company, role_title="AI Engineer", country=country,
        location=country, source="test", source_url=f"https://example.test/{pid}",
        date_posted="2026-09-01", text=text, **kw,
    )


SPONSORS = "We are an approved sponsor and can support a subclass 482 visa for offshore applicants."
REFUSES = "We are unable to sponsor visas and require existing work rights."


class FakeSource(SourceAdapter):
    def __init__(self, postings, name="fake"):
        self._postings = postings
        self.name = name

    def describe(self):
        return self.name

    def fetch(self, limit=100):
        return list(self._postings)[:limit]


class BrokenSource(SourceAdapter):
    name = "broken"

    def fetch(self, limit=100):
        raise SourceError("broken", "HTTP 503")


class ExplodingSource(SourceAdapter):
    name = "exploding"

    def fetch(self, limit=100):
        raise ValueError("unexpected provider change")


@pytest.fixture
def store(tmp_path):
    return PostingStore(tmp_path / "test.db")


# --- the core loop ---------------------------------------------------------


def test_an_empty_store_is_still_the_store_it_was_given(tmp_path):
    """REGRESSION: JobAgent used `store or PostingStore()`. PostingStore
    defines __len__, so an EMPTY store is falsy and was silently replaced by
    the default database — the agent wrote to a file the caller never named,
    and 'nothing new' was reported because the rows went elsewhere. Silent,
    and only visible as a wrong number.
    """
    given = PostingStore(tmp_path / "explicit.db")
    assert len(given) == 0  # the falsy condition that triggered the bug
    agent = JobAgent(store=given, sources=[FakeSource([make_posting("a", SPONSORS)])])
    assert agent.store is given
    report = agent.run()
    assert report.stored_new == 1
    assert len(given) == 1


def test_run_classifies_and_stores(store):
    agent = JobAgent(store=store, sources=[FakeSource([
        make_posting("a", SPONSORS),
        make_posting("b", REFUSES, company="Nope"),
    ])])
    report = agent.run()
    assert report.fetched == 2
    assert report.stored_new == 2
    assert report.label_counts["likely sponsoring"] == 1
    assert report.label_counts["likely NOT sponsoring"] == 1


def test_second_run_reports_nothing_new(store):
    """The whole point of a scheduled agent: it must not re-alert you about
    postings it already reported."""
    source = FakeSource([make_posting("a", SPONSORS)])
    agent = JobAgent(store=store, sources=[source])
    assert agent.run().stored_new == 1
    second = agent.run()
    assert second.stored_new == 0
    assert second.fetched == 1  # still fetched, just not new
    assert "No new postings" in second.digest()


def test_only_genuinely_new_postings_are_reported(store):
    agent = JobAgent(store=store, sources=[FakeSource([make_posting("a", SPONSORS)])])
    agent.run()
    agent.sources = [FakeSource([make_posting("a", SPONSORS), make_posting("b", SPONSORS, company="New Co")])]
    report = agent.run()
    assert report.stored_new == 1
    assert report.new_postings[0]["company"] == "New Co"


def test_duplicate_ids_within_one_run_are_collapsed(store):
    """The same role often appears on two boards; it should count once."""
    agent = JobAgent(store=store, sources=[
        FakeSource([make_posting("same", SPONSORS)], name="one"),
        FakeSource([make_posting("same", SPONSORS)], name="two"),
    ])
    report = agent.run()
    assert report.fetched == 1
    assert report.stored_new == 1


# --- resilience ------------------------------------------------------------


def test_a_dead_source_does_not_kill_the_run(store):
    agent = JobAgent(store=store, sources=[BrokenSource(), FakeSource([make_posting("a", SPONSORS)])])
    report = agent.run()
    assert report.stored_new == 1
    assert report.sources_failed[0]["source"] == "broken"
    assert "503" in report.sources_failed[0]["error"]
    assert "broken" in report.digest()


def test_an_unexpected_exception_is_contained(store):
    """A provider silently changing its JSON shape raises something that
    isn't SourceError. The run must survive that too."""
    agent = JobAgent(store=store, sources=[ExplodingSource(), FakeSource([make_posting("a", SPONSORS)])])
    report = agent.run()
    assert report.stored_new == 1
    assert "ValueError" in report.sources_failed[0]["error"]


def test_all_sources_failing_is_reported_not_raised(store):
    report = JobAgent(store=store, sources=[BrokenSource(), ExplodingSource()]).run()
    assert report.stored_new == 0
    assert len(report.sources_failed) == 2


# --- provider disagreement -------------------------------------------------


def test_provider_flag_disagreeing_with_the_text_is_surfaced(store):
    """Where a source claims sponsorship but the posting says otherwise,
    that's the most useful row in the report — it's where an automated
    label is most likely wrong."""
    agent = JobAgent(store=store, sources=[FakeSource([
        make_posting("a", REFUSES, provider_sponsorship=True),
    ])])
    report = agent.run()
    assert len(report.disagreements) == 1
    assert report.disagreements[0]["provider_sponsorship"] is True
    assert "disagreed" in report.digest()


def test_agreement_is_not_flagged(store):
    agent = JobAgent(store=store, sources=[FakeSource([
        make_posting("a", SPONSORS, provider_sponsorship=True),
    ])])
    assert JobAgent(store=store, sources=agent.sources).run().disagreements == []


# --- digest ----------------------------------------------------------------


def test_digest_lists_sponsoring_roles_with_links(store):
    agent = JobAgent(store=store, sources=[FakeSource([make_posting("a", SPONSORS)])])
    digest = agent.run().digest()
    assert "likely sponsoring" in digest
    assert "https://example.test/a" in digest


def test_digest_is_honest_when_nothing_sponsors(store):
    agent = JobAgent(store=store, sources=[FakeSource([make_posting("a", REFUSES)])])
    assert "Nothing new with a clear sponsorship signal" in agent.run().digest()


# --- store -----------------------------------------------------------------


def test_store_filters_by_country_and_label(store):
    JobAgent(store=store, sources=[FakeSource([
        make_posting("a", SPONSORS, country="Australia"),
        make_posting("b", SPONSORS, company="Dutch Co", country="Netherlands"),
        make_posting("c", REFUSES, company="No Co", country="Australia"),
    ])]).run()

    assert len(store.all_postings(country="Australia")) == 2
    assert len(store.all_postings(label="likely sponsoring")) == 2
    assert len(store.all_postings(country="Australia", label="likely sponsoring")) == 1
    assert len(store.all_postings(country="All")) == 3


def test_store_stats_and_countries(store):
    JobAgent(store=store, sources=[FakeSource([
        make_posting("a", SPONSORS, country="Australia"),
        make_posting("b", SPONSORS, company="Dutch Co", country="Netherlands"),
    ])]).run()

    stats = store.stats()
    assert stats["total"] == 2
    assert stats["countries"] == 2
    assert stats["sponsoring"] == 2
    assert {c["country"] for c in store.countries()} == {"Australia", "Netherlands"}


def test_store_survives_a_new_process(tmp_path):
    """A scheduled run is a fresh process — state has to come off disk."""
    path = tmp_path / "persist.db"
    JobAgent(store=PostingStore(path), sources=[FakeSource([make_posting("a", SPONSORS)])]).run()
    reopened = JobAgent(store=PostingStore(path), sources=[FakeSource([make_posting("a", SPONSORS)])])
    assert reopened.run().stored_new == 0


def test_notification_state_round_trips(store):
    agent = JobAgent(store=store, sources=[FakeSource([make_posting("a", SPONSORS)])])
    agent.run()
    pending = store.unnotified()
    assert len(pending) == 1
    store.mark_notified([p["id"] for p in pending])
    assert store.unnotified() == []


def test_evidence_phrases_are_persisted(store):
    """The evidence is what lets you disagree with the label at a glance, so
    it has to survive the round trip to storage."""
    JobAgent(store=store, sources=[FakeSource([make_posting("a", SPONSORS)])]).run()
    row = store.all_postings()[0]
    assert "approved sponsor" in row["evidence"]
