"""An LLM-based alternative to the keyword sponsorship heuristic, behind a
provider-agnostic interface.

`sponsorship_signal()` in pipeline.py is fast and has zero external
dependencies, but it is exact-phrase matching — it cannot read a paraphrase
it has no phrase for. A language model reads for the same underlying
question (does this posting offer visa sponsorship?) from context instead
of a fixed phrase list, which is the sort of case worth measuring rather
than assuming.

LLMSponsorshipClassifier is that interface. This repo has no
ANTHROPIC_API_KEY configured (see the placeholder in .env.example, already
anticipating this) and no network budget here to verify a real call end to
end, so the only implementation shipped is StubLLMClassifier — a
deterministic, offline stand-in used to build and test the eval harness in
sponsor_scout.eval against something real, without needing a key or a
network call in CI.

Wiring in a real provider later means writing one class:

    class AnthropicSponsorshipClassifier(LLMSponsorshipClassifier):
        def __init__(self, model="claude-sonnet-5"):
            import anthropic  # lazy import, like sentence-transformers in embeddings.py
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
            self._model = model

        def classify(self, text: str) -> LLMClassification:
            prompt = (
                "Read this job posting and decide whether it offers visa "
                "sponsorship. Answer with exactly one of these labels:\n"
                + "\n".join(SPONSORSHIP_LABELS) + f"\n\nPosting:\n{text}\n\nLabel:"
            )
            response = self._client.messages.create(
                model=self._model, max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            label = response.content[0].text.strip()
            return LLMClassification(label=label if label in SPONSORSHIP_LABELS
                                      else "no clear sponsorship language - unclear, verify manually",
                                      rationale=f"model: {self._model}")

    def classify_many() should override the default one-at-a-time loop to
    batch requests (or run them concurrently) instead — a real API call per
    posting one at a time is the throughput bottleneck a production version
    would need to fix first.

Everything else — the eval harness, the metrics, the disagreement report —
is unchanged: it only calls classify()/classify_many() through this
interface, never anything provider-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sponsor_scout.pipeline import _is_negated, _phrase_present

UNCLEAR_LABEL = "no clear sponsorship language - unclear, verify manually"
SILENT_LABEL = "no sponsorship language found"


@dataclass
class LLMClassification:
    label: str
    rationale: str = ""


class LLMSponsorshipClassifier:
    """Interface. `label` on every returned LLMClassification must be one
    of SPONSORSHIP_LABELS, so a classifier here is a drop-in comparison
    against sponsorship_signal() rather than a different label scheme the
    eval harness has to reconcile."""

    def classify(self, text: str) -> LLMClassification:
        raise NotImplementedError

    def classify_many(self, texts: List[str]) -> List[LLMClassification]:
        """Default: one call per text. A real API-backed implementation
        should override this to batch or parallelize requests — the
        eval harness calls this, not classify(), so that's the one seam
        that needs to change for throughput."""
        return [self.classify(t) for t in texts]


# ---------------------------------------------------------------------------
# Stub implementation — NOT a language model
# ---------------------------------------------------------------------------
#
# A deterministic rule-based stand-in, used only to exercise the eval
# harness end to end without a network call or an API key. It deliberately
# reads for a different, broader set of phrases than sponsorship_signal()'s
# exact list — closer to paraphrases a model would plausibly pick up on
# (e.g. "global mobility team", "welcome overseas applicants") that aren't
# in the keyword list at all — so the two produce genuine disagreements for
# the harness to surface, rather than trivially agreeing on everything.
# It reuses sponsorship_signal()'s word-boundary and negation-window checks
# (_phrase_present/_is_negated) rather than reimplementing naive substring
# matching, which is exactly the failure mode documented in
# tests/test_sponsorship_signal.py.

STUB_POSITIVE_HINTS = [
    "visa sponsorship", "sponsor a visa", "sponsor a work permit",
    "approved sponsor", "licensed sponsor", "registered sponsor",
    "subclass 482", "blue card", "skilled worker visa", "lmia",
    "global talent stream", "global mobility", "immigration support",
    "immigration team", "handle your visa", "handle the visa process",
    "assist with relocation", "relocation support", "relocation assistance",
    "welcome overseas applicants", "welcome international candidates",
    "open to overseas candidates", "open to international applicants",
    "work permit sponsorship",
]

STUB_NEGATIVE_HINTS = [
    "unable to sponsor", "not able to sponsor", "not able to offer visa",
    "do not sponsor", "cannot sponsor", "no sponsorship",
    "not currently registered as a sponsor", "no visa support",
    "no relocation support", "no relocation assistance",
    "must already have the right to work", "existing work rights required",
    "must have work authorization", "already authorized to work",
    "must hold valid work authorization", "local candidates only",
]

STUB_AMBIGUOUS_HINTS = [
    "unclear", "not specified", "tbd", "to be confirmed", "unsure",
    "contact recruiter", "case by case", "case-by-case",
]


class StubLLMClassifier(LLMSponsorshipClassifier):
    """See the module-level note above: a deterministic offline stand-in,
    not a real model call."""

    def classify(self, text: str) -> LLMClassification:
        lower = text.lower()
        pos_hits = [
            p for p in STUB_POSITIVE_HINTS
            if _phrase_present(lower, p) and not _is_negated(lower, p)
        ]
        neg_hits = [p for p in STUB_NEGATIVE_HINTS if _phrase_present(lower, p)]
        ambiguous_hits = [p for p in STUB_AMBIGUOUS_HINTS if _phrase_present(lower, p)]

        if pos_hits and not neg_hits:
            label = "likely sponsoring"
            rationale = f"reads as offering sponsorship: {', '.join(pos_hits)}"
        elif neg_hits and not pos_hits:
            label = "likely NOT sponsoring"
            rationale = f"reads as declining sponsorship: {', '.join(neg_hits)}"
        elif pos_hits and neg_hits:
            label = "mixed signal - verify manually"
            rationale = f"both signals present: +{pos_hits} / -{neg_hits}"
        elif ambiguous_hits:
            label = UNCLEAR_LABEL
            rationale = f"explicitly flagged as unclear: {', '.join(ambiguous_hits)}"
        else:
            label = SILENT_LABEL
            rationale = "no sponsorship-relevant language found"
        return LLMClassification(label=label, rationale=rationale)
