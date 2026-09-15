"""Eval harness: measure an LLMSponsorshipClassifier against the keyword
heuristic (sponsorship_signal()) that the rest of the pipeline uses today.

The keyword heuristic is the *baseline* here, not ground truth — the README
is explicit that it's a triage filter, not a classifier to trust blindly.
What this harness answers is narrower and more useful: where does a
candidate classifier agree with the heuristic, and — the interesting part —
exactly where and how does it disagree? Every disagreement is a posting
worth reading yourself, the same way agent.py already surfaces
provider-flag disagreements as the most informative row in a run.

    python -m sponsor_scout.eval_llm --sample
    python -m sponsor_scout.eval_llm --db data/postings.db --limit 500
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from sponsor_scout.llm_classifier import LLMSponsorshipClassifier, StubLLMClassifier
from sponsor_scout.pipeline import SPONSORSHIP_LABELS, sponsorship_signal
from sponsor_scout.store import PostingStore

# Short forms for the confusion-matrix header/rows — the real labels are
# full sentences and would make the matrix unreadable as columns.
_SHORT_LABEL = {
    "likely sponsoring": "sponsor",
    "likely NOT sponsoring": "not-sponsor",
    "mixed signal - verify manually": "mixed",
    "no clear sponsorship language - unclear, verify manually": "unclear",
    "no sponsorship language found": "silent",
}


@dataclass
class LLMEvalResult:
    n: int
    labels: List[str]
    accuracy: float
    macro_f1: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1: Dict[str, float]
    support: Dict[str, int]
    confusion_matrix: List[List[int]]  # rows = baseline (keyword heuristic), cols = classifier under test
    disagreements: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "labels": self.labels,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
            "confusion_matrix": self.confusion_matrix,
            "disagreements": self.disagreements,
        }

    def digest(self, max_disagreements: int = 15) -> str:
        if self.n == 0:
            return "No postings to evaluate."

        lines = [f"LLM classifier vs. keyword baseline — {self.n} posting(s)", ""]
        lines.append(f"Agreement with baseline: {self.accuracy:.1%}    Macro F1: {self.macro_f1:.3f}")
        lines.append("")
        lines.append(f"{'label':<58} {'prec':>6} {'rec':>6} {'f1':>6} {'support':>8}")
        for label in self.labels:
            lines.append(
                f"{label:<58} {self.precision[label]:>6.2f} {self.recall[label]:>6.2f} "
                f"{self.f1[label]:>6.2f} {self.support[label]:>8}"
            )

        lines.append("")
        lines.append("Confusion matrix (rows = keyword baseline, cols = classifier under test):")
        short = [_SHORT_LABEL.get(l, l) for l in self.labels]
        col_width = max(len(s) for s in short) + 2
        lines.append(" " * 14 + "".join(f"{s:>{col_width}}" for s in short))
        for label, row in zip(short, self.confusion_matrix):
            lines.append(f"{label:<14}" + "".join(f"{v:>{col_width}}" for v in row))

        if self.disagreements:
            lines.append("")
            lines.append(
                f"{len(self.disagreements)} disagreement(s) — where the classifier under test read "
                "a posting differently from the keyword baseline:"
            )
            for d in self.disagreements[:max_disagreements]:
                lines.append(
                    f"  - {d.get('company', '?')}: baseline='{d['baseline_label']}' "
                    f"vs classifier='{d['llm_label']}'"
                )
            if len(self.disagreements) > max_disagreements:
                lines.append(f"  … and {len(self.disagreements) - max_disagreements} more")
        else:
            lines.append("")
            lines.append("No disagreements — the classifier under test read every posting the same way as the baseline.")
        return "\n".join(lines)


def evaluate(
    postings: List[Dict[str, Any]],
    classifier: Optional[LLMSponsorshipClassifier] = None,
) -> LLMEvalResult:
    """Run `classifier` (default: StubLLMClassifier — see llm_classifier.py
    for why) over `postings` and compare its labels against
    sponsorship_signal()'s, treating the keyword heuristic as the
    reference/baseline label for precision, recall, F1 and the confusion
    matrix. `postings` need only have "text", and ideally "id"/"company"
    for readable disagreement rows — the shape sample_data.py and
    PostingStore.all_postings() both already produce.
    """
    classifier = classifier or StubLLMClassifier()
    labels = list(SPONSORSHIP_LABELS)

    if not postings:
        zeros = {label: 0.0 for label in labels}
        zero_support = {label: 0 for label in labels}
        return LLMEvalResult(
            n=0, labels=labels, accuracy=0.0, macro_f1=0.0,
            precision=zeros, recall=dict(zeros), f1=dict(zeros), support=zero_support,
            confusion_matrix=[[0] * len(labels) for _ in labels],
        )

    texts = [p.get("text", "") for p in postings]
    baseline_labels = [sponsorship_signal(t)["label"] for t in texts]
    classifications = classifier.classify_many(texts)
    llm_labels = [c.label for c in classifications]

    precision, recall, f1, support = precision_recall_fscore_support(
        baseline_labels, llm_labels, labels=labels, zero_division=0
    )
    cm = confusion_matrix(baseline_labels, llm_labels, labels=labels)
    accuracy = float(np.mean([b == p for b, p in zip(baseline_labels, llm_labels)]))
    # Macro-average only over labels that actually occur in this run - in
    # the baseline OR in what the classifier predicted, so a hallucinated
    # label the baseline never uses still counts against the score.
    # Averaging in the untouched canonical labels at f1=0 would instead
    # penalise the score for classes that were never in play in this eval
    # set at all, which is a property of the data, not the classifier.
    observed = set(baseline_labels) | set(llm_labels)
    present = np.array([label in observed for label in labels])
    macro_f1 = float(np.mean(f1[present])) if present.any() else 0.0

    disagreements = [
        {
            "id": posting.get("id"),
            "company": posting.get("company"),
            "baseline_label": baseline,
            "llm_label": predicted,
            "llm_rationale": classification.rationale,
            "excerpt": (posting.get("text") or "")[:200],
        }
        for posting, baseline, predicted, classification in zip(postings, baseline_labels, llm_labels, classifications)
        if baseline != predicted
    ]

    return LLMEvalResult(
        n=len(postings),
        labels=labels,
        accuracy=accuracy,
        macro_f1=macro_f1,
        precision=dict(zip(labels, precision.tolist())),
        recall=dict(zip(labels, recall.tolist())),
        f1=dict(zip(labels, f1.tolist())),
        support=dict(zip(labels, support.tolist())),
        confusion_matrix=cm.tolist(),
        disagreements=disagreements,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sponsor_scout.eval_llm", description=__doc__)
    parser.add_argument("--db", default="data/postings.db", help="SQLite store path")
    parser.add_argument("--limit", type=int, default=500, help="max postings to evaluate")
    parser.add_argument("--sample", action="store_true", help="evaluate the synthetic sample corpus instead of the store")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    if args.sample:
        from sponsor_scout.sample_data import SAMPLE_POSTINGS
        postings = SAMPLE_POSTINGS
    else:
        postings = PostingStore(args.db).all_postings(limit=args.limit)
        if not postings:
            print(f"No postings in {args.db} — run the agent first, or pass --sample.")
            return 1

    result = evaluate(postings)
    print(json.dumps(result.to_dict(), indent=2) if args.json else result.digest())
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
