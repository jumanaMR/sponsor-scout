"""Tests for the sponsorship-signal heuristic.

Two of these are regression tests for bugs that actually shipped in an
early version and were caught by reading output rather than by the code
looking wrong. Both are the same failure mode — naive substring matching
against a phrase list — and both would have put a company that explicitly
refuses to sponsor in front of a candidate as a "maybe".
"""

import pytest

from sponsor_scout.pipeline import sponsorship_signal
from sponsor_scout.sample_data import SAMPLE_POSTINGS


def label(text: str) -> str:
    return sponsorship_signal(text)["label"]


# --- the two regressions ---------------------------------------------------


def test_negated_positive_phrase_is_not_a_positive_match():
    """REGRESSION: 'no mention of visa sponsorship' contains the positive
    phrase 'visa sponsorship', and was being read as a positive signal."""
    result = sponsorship_signal(
        "No mention of visa sponsorship in this posting; status unclear."
    )
    assert "visa sponsorship" not in result["positive_matches"]
    assert result["label"] != "likely sponsoring"


def test_unable_to_sponsor_is_not_a_mixed_signal():
    """REGRESSION: 'unable to sponsor' contains the positive phrase
    'able to sponsor' as a substring, so a clear refusal came back as
    'mixed signal - verify manually' instead of a clear no."""
    result = sponsorship_signal("We are unable to sponsor visas for this role.")
    assert result["positive_matches"] == []
    assert result["label"] == "likely NOT sponsoring"


def test_word_start_fix_did_not_break_suffixes():
    """The fix for the bug above checks the START of a word only, so a
    plural or other suffix must still match."""
    assert label("We are an approved sponsors list member.") == "likely sponsoring"


# --- clear cases -----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "We are an approved sponsor and can support a subclass 482 visa.",
        "This role is eligible for Skilled Worker visa sponsorship.",
        "We support the EU Blue Card process for qualified non-EU applicants.",
        "We can support a work permit plus LMIA sponsorship.",
    ],
)
def test_clear_positives(text):
    assert label(text) == "likely sponsoring"


@pytest.mark.parametrize(
    "text",
    [
        "Unfortunately we are not able to offer visa sponsorship at this time.",
        "We do not sponsor visas for this position.",
        "We are not currently registered as a sponsor for employment permits.",
        "Applicants must already have the right to work in the EU.",
    ],
)
def test_clear_negatives(text):
    assert label(text) == "likely NOT sponsoring"


def test_ambiguous_language_is_flagged_not_guessed():
    result = sponsorship_signal(
        "Sponsorship status unclear; recommend contacting the recruiter."
    )
    assert "unclear" in result["label"]


def test_silence_is_not_a_signal():
    result = sponsorship_signal(
        "We build warehouse robotics in Python and C++. Five years experience required."
    )
    assert result["label"] == "no sponsorship language found"
    assert result["positive_matches"] == []
    assert result["negative_matches"] == []


# --- the shipped sample corpus --------------------------------------------

EXPECTED = {
    "Nimbus Data Labs": "likely sponsoring",
    "Aurora Analytics Pty Ltd": "likely NOT sponsoring",
    "Harbor Robotics": "likely sponsoring",
    "Delta Cloud Systems": "likely sponsoring",
    "Nordwind Systeme GmbH": "likely sponsoring",
    "Thistle & Byte": "likely sponsoring",
    "Cliffside Softworks": "likely NOT sponsoring",
    "Maple Ridge AI": "likely sponsoring",
    "Silverline Cognition": "likely sponsoring",
    "Ferngully Data Co": "likely NOT sponsoring",
    "Brightcask Technologies": "likely sponsoring",
}


@pytest.mark.parametrize("posting", SAMPLE_POSTINGS, ids=lambda p: p["company"])
def test_sample_corpus_labels_are_stable(posting):
    """Pins the whole sample corpus so a change to the phrase lists can't
    silently reclassify postings."""
    expected = EXPECTED.get(posting["company"])
    if expected is None:  # Quillfeather is the deliberate 'unclear' case
        assert "unclear" in label(posting["text"])
    else:
        assert label(posting["text"]) == expected
