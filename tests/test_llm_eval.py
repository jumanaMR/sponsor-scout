"""Tests for the LLM classifier interface, its stub implementation, and the
eval harness that compares a classifier against the keyword baseline."""

from __future__ import annotations

from typing import List

import pytest

from sponsor_scout.eval_llm import evaluate
from sponsor_scout.llm_classifier import (
    LLMClassification,
    LLMSponsorshipClassifier,
    StubLLMClassifier,
)
from sponsor_scout.pipeline import SPONSORSHIP_LABELS, sponsorship_signal
from sponsor_scout.sample_data import SAMPLE_POSTINGS

SPONSORS = "We are an approved sponsor and can support a subclass 482 visa for offshore applicants."
REFUSES = "We are unable to sponsor visas and require existing work rights."
SILENT = "We build warehouse robotics in Python and C++. Five years experience required."


def posting(pid: str, text: str, **kw) -> dict:
    row = {"id": pid, "company": kw.pop("company", "Acme"), "text": text}
    row.update(kw)
    return row


class MirrorClassifier(LLMSponsorshipClassifier):
    """Always agrees with the keyword baseline — the "perfect" case."""

    def classify(self, text: str) -> LLMClassification:
        return LLMClassification(label=sponsorship_signal(text)["label"], rationale="mirrors baseline")


class ConstantClassifier(LLMSponsorshipClassifier):
    """Always returns the same label, regardless of input — a classifier
    that's wrong whenever the baseline says anything else."""

    def __init__(self, label: str):
        self.label = label

    def classify(self, text: str) -> LLMClassification:
        return LLMClassification(label=self.label, rationale="constant")


class BatchingClassifier(LLMSponsorshipClassifier):
    """Overrides classify_many() instead of classify() — the seam a real
    API-backed implementation would use to batch requests."""

    def classify(self, text: str) -> LLMClassification:
        raise AssertionError("classify() should not be called when classify_many() is overridden")

    def classify_many(self, texts: List[str]) -> List[LLMClassification]:
        return [LLMClassification(label=sponsorship_signal(t)["label"]) for t in texts]


# --- LLMSponsorshipClassifier interface -------------------------------------


def test_default_classify_many_calls_classify_per_text():
    calls = []

    class Recorder(LLMSponsorshipClassifier):
        def classify(self, text):
            calls.append(text)
            return LLMClassification(label="likely sponsoring")

    Recorder().classify_many(["a", "b", "c"])
    assert calls == ["a", "b", "c"]


def test_overriding_classify_many_bypasses_classify():
    results = BatchingClassifier().classify_many([SPONSORS, REFUSES])
    assert [r.label for r in results] == ["likely sponsoring", "likely NOT sponsoring"]


# --- StubLLMClassifier -------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_label",
    [
        (SPONSORS, "likely sponsoring"),
        (REFUSES, "likely NOT sponsoring"),
        (SILENT, "no sponsorship language found"),
        ("Sponsorship status unclear; recommend contacting the recruiter.",
         "no clear sponsorship language - unclear, verify manually"),
    ],
)
def test_stub_classifier_returns_canonical_labels(text, expected_label):
    result = StubLLMClassifier().classify(text)
    assert result.label == expected_label
    assert result.label in SPONSORSHIP_LABELS


def test_stub_classifier_negation_is_not_a_positive_match():
    """The stub reuses sponsorship_signal()'s negation/word-boundary
    helpers rather than reimplementing naive substring matching — this
    pins that it actually benefits from that, not just imports it."""
    result = StubLLMClassifier().classify("We offer no relocation support for this role.")
    assert result.label != "likely sponsoring"


def test_stub_classifier_rationale_is_populated():
    result = StubLLMClassifier().classify(SPONSORS)
    assert result.rationale


# --- evaluate() ---------------------------------------------------------------


def test_evaluate_empty_postings_does_not_raise():
    result = evaluate([])
    assert result.n == 0
    assert result.accuracy == 0.0
    assert result.disagreements == []
    assert "No postings" in result.digest()


def test_perfect_agreement_has_no_disagreements_and_full_scores():
    postings = [posting("a", SPONSORS), posting("b", REFUSES), posting("c", SILENT)]
    result = evaluate(postings, classifier=MirrorClassifier())
    assert result.accuracy == 1.0
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.disagreements == []
    for label in ["likely sponsoring", "likely NOT sponsoring", "no sponsorship language found"]:
        assert result.precision[label] == 1.0
        assert result.recall[label] == 1.0


def test_constant_wrong_classifier_disagrees_on_everything_it_gets_wrong():
    postings = [posting("a", SPONSORS), posting("b", REFUSES)]
    # Constant classifier always says "likely NOT sponsoring" - right for
    # posting b, wrong for posting a.
    result = evaluate(postings, classifier=ConstantClassifier("likely NOT sponsoring"))
    assert result.accuracy == 0.5
    assert len(result.disagreements) == 1
    assert result.disagreements[0]["id"] == "a"
    assert result.disagreements[0]["baseline_label"] == "likely sponsoring"
    assert result.disagreements[0]["llm_label"] == "likely NOT sponsoring"


def test_disagreement_rows_carry_identifying_fields():
    postings = [posting("a", SPONSORS, company="Acme Corp")]
    result = evaluate(postings, classifier=ConstantClassifier("no sponsorship language found"))
    row = result.disagreements[0]
    assert row["company"] == "Acme Corp"
    assert row["id"] == "a"
    assert "llm_rationale" in row
    assert row["excerpt"] == SPONSORS[:200]


def test_confusion_matrix_shape_matches_label_count():
    postings = [posting("a", SPONSORS), posting("b", REFUSES)]
    result = evaluate(postings, classifier=MirrorClassifier())
    n = len(SPONSORSHIP_LABELS)
    assert len(result.confusion_matrix) == n
    assert all(len(row) == n for row in result.confusion_matrix)
    assert sum(sum(row) for row in result.confusion_matrix) == len(postings)


def test_default_classifier_is_the_stub_and_runs_on_sample_data():
    """No classifier passed - evaluate() should default to StubLLMClassifier
    and run without error over the real sample corpus."""
    result = evaluate(SAMPLE_POSTINGS)
    assert result.n == len(SAMPLE_POSTINGS)
    assert 0.0 <= result.accuracy <= 1.0
    assert set(result.labels) == set(SPONSORSHIP_LABELS)


def test_digest_reports_agreement_and_disagreements():
    postings = [posting("a", SPONSORS), posting("b", REFUSES)]
    result = evaluate(postings, classifier=ConstantClassifier("no sponsorship language found"))
    digest = result.digest()
    assert "Agreement with baseline" in digest
    assert "disagreement" in digest
    assert "Confusion matrix" in digest


def test_digest_is_honest_when_everything_agrees():
    postings = [posting("a", SPONSORS)]
    digest = evaluate(postings, classifier=MirrorClassifier()).digest()
    assert "No disagreements" in digest
