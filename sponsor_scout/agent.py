"""The ingestion agent.

One run does the whole loop without supervision:

    poll every configured source
        -> normalise each posting to one shape
        -> classify its sponsorship language
        -> store it, keeping only what was never seen before
        -> report what's new

Designed to be run on a timer (cron, a systemd timer, GitHub Actions on a
schedule, or an EventBridge rule once it's on AWS) and to be safe when it
fails: one dead source doesn't abort the run, and re-running never
re-reports a posting it already told you about.

    python -m sponsor_scout.agent run --limit 50
    python -m sponsor_scout.agent doctor
    python -m sponsor_scout.agent report --country Australia
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sponsor_scout.pipeline import sponsorship_signal
from sponsor_scout.sources import SourceAdapter, SourceError, build_sources
from sponsor_scout.store import PostingStore, posting_to_record


@dataclass
class RunReport:
    """What one agent run did. Printed as a digest, and the shape a
    notification channel (email, Slack) would serialise."""

    started_at: str
    fetched: int = 0
    stored_new: int = 0
    sources_ok: List[str] = field(default_factory=list)
    sources_failed: List[Dict[str, str]] = field(default_factory=list)
    new_postings: List[Dict[str, Any]] = field(default_factory=list)
    label_counts: Dict[str, int] = field(default_factory=dict)
    disagreements: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "fetched": self.fetched,
            "stored_new": self.stored_new,
            "sources_ok": self.sources_ok,
            "sources_failed": self.sources_failed,
            "label_counts": self.label_counts,
            "new_postings": self.new_postings,
            "disagreements": self.disagreements,
        }

    def digest(self, max_rows: int = 15) -> str:
        """Human-readable summary — the text a scheduled run would email."""
        lines = [f"Sponsor Scout — run at {self.started_at}", ""]
        lines.append(
            f"{self.fetched} postings fetched from {len(self.sources_ok)} source(s); "
            f"{self.stored_new} new."
        )
        if self.sources_failed:
            lines.append("")
            lines.append("Sources that failed (run continued without them):")
            for failure in self.sources_failed:
                lines.append(f"  - {failure['source']}: {failure['error']}")

        sponsoring = [p for p in self.new_postings if p["sponsorship_label"] == "likely sponsoring"]
        if sponsoring:
            lines.append("")
            lines.append(f"New and likely sponsoring ({len(sponsoring)}):")
            for posting in sponsoring[:max_rows]:
                lines.append(
                    f"  - {posting['company']} — {posting['role_title']} "
                    f"({posting['country'] or 'location unclear'})"
                )
                if posting.get("source_url"):
                    lines.append(f"    {posting['source_url']}")
            if len(sponsoring) > max_rows:
                lines.append(f"  … and {len(sponsoring) - max_rows} more")
        elif self.stored_new:
            lines.append("")
            lines.append("Nothing new with a clear sponsorship signal this run.")

        if self.disagreements:
            lines.append("")
            lines.append(
                f"{len(self.disagreements)} posting(s) where the source's own visa flag "
                "disagreed with the text — worth reading yourself:"
            )
            for row in self.disagreements[:5]:
                lines.append(
                    f"  - {row['company']}: source says {row['provider_sponsorship']}, "
                    f"text reads as '{row['sponsorship_label']}'"
                )

        if not self.stored_new:
            lines.append("")
            lines.append("No new postings since the last run.")
        return "\n".join(lines)


class JobAgent:
    def __init__(
        self,
        store: Optional[PostingStore] = None,
        sources: Optional[List[SourceAdapter]] = None,
    ):
        # `store or PostingStore()` would be a bug: PostingStore defines
        # __len__, so an EMPTY store is falsy and would be silently swapped
        # for the default database — the agent would write somewhere the
        # caller never asked for. Always compare against None.
        self.store = store if store is not None else PostingStore()
        self.sources = sources if sources is not None else build_sources()

    def run(self, limit_per_source: int = 50) -> RunReport:
        report = RunReport(started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        records: List[Dict[str, Any]] = []
        seen_this_run: set[str] = set()

        for source in self.sources:
            label = source.describe()
            try:
                postings = source.fetch(limit=limit_per_source)
            except SourceError as exc:
                # A provider being down is routine, not fatal: a scheduled run
                # that aborts on the first 503 stops reporting altogether.
                report.sources_failed.append({"source": label, "error": str(exc)})
                continue
            except Exception as exc:  # noqa: BLE001 - defensive: never kill the run
                report.sources_failed.append({"source": label, "error": f"{type(exc).__name__}: {exc}"})
                continue

            report.sources_ok.append(label)
            for posting in postings:
                if posting.id in seen_this_run:
                    continue  # same role listed on two boards
                seen_this_run.add(posting.id)
                report.fetched += 1

                signal = sponsorship_signal(posting.text)
                evidence = signal["positive_matches"] + signal["negative_matches"]
                record = posting_to_record(
                    posting,
                    label=signal["label"],
                    evidence=evidence,
                    first_seen=report.started_at,
                )
                records.append(record)

                report.label_counts[signal["label"]] = report.label_counts.get(signal["label"], 0) + 1

                # Where a source publishes its own visa flag, a mismatch with
                # what the text actually says is the most informative thing in
                # the run — it's exactly where an automated label is wrong.
                claimed = posting.provider_sponsorship
                if claimed is True and signal["label"] == "likely NOT sponsoring":
                    report.disagreements.append(_disagreement(record, claimed))
                elif claimed is False and signal["label"] == "likely sponsoring":
                    report.disagreements.append(_disagreement(record, claimed))

        new_ids = self.store.upsert_many(records)
        report.stored_new = len(new_ids)
        by_id = {r["id"]: r for r in records}
        report.new_postings = [by_id[i] for i in new_ids if i in by_id]
        return report

    def doctor(self) -> List[Dict[str, Any]]:
        """Hit every source once and report whether its field mapping worked.

        This exists because the adapters were written in an environment with
        no outbound access to these APIs — so the parsing is unverified until
        it runs somewhere with a real network. Run this first; it tells you
        per source whether you got postings and whether the fields that
        matter (title, text, location) actually came through.
        """
        results = []
        for source in self.sources:
            entry: Dict[str, Any] = {"source": source.describe(), "ok": False}
            try:
                postings = source.fetch(limit=3)
            except SourceError as exc:
                entry["error"] = str(exc)
                results.append(entry)
                continue
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"{type(exc).__name__}: {exc}"
                results.append(entry)
                continue

            entry["count"] = len(postings)
            if not postings:
                entry["error"] = "reachable, but returned no postings"
                results.append(entry)
                continue

            sample = postings[0]
            entry["ok"] = True
            entry["sample"] = {
                "company": sample.company,
                "role_title": sample.role_title,
                "location": sample.location,
                "country": sample.country,
                "text_chars": len(sample.text),
                "has_url": bool(sample.source_url),
            }
            # The text length is the field that silently breaks: if an ATS
            # renames its description field, everything still "works" but
            # every posting classifies as 'no sponsorship language'.
            problems = []
            if not sample.role_title:
                problems.append("no role_title")
            if len(sample.text) < 200:
                problems.append(f"description looks too short ({len(sample.text)} chars) — field mapping may be wrong")
            if not sample.location:
                problems.append("no location, so country filtering won't work")
            if problems:
                entry["warnings"] = problems
            results.append(entry)
        return results


def _disagreement(record: Dict[str, Any], claimed: bool) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "company": record["company"],
        "role_title": record["role_title"],
        "provider_sponsorship": claimed,
        "sponsorship_label": record["sponsorship_label"],
        "source_url": record.get("source_url", ""),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_agent(args: argparse.Namespace) -> JobAgent:
    sources = build_sources(
        greenhouse=args.greenhouse.split(",") if args.greenhouse else None,
        lever=args.lever.split(",") if args.lever else None,
        ashby=args.ashby.split(",") if args.ashby else None,
        arbeitnow=not args.no_arbeitnow,
        arbeitnow_visa_only=args.visa_only,
        arbeitnow_pages=args.pages,
    )
    return JobAgent(store=PostingStore(args.db), sources=sources)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sponsor_scout.agent", description=__doc__)
    parser.add_argument("command", choices=["run", "doctor", "report"], help="what to do")
    parser.add_argument("--db", default="data/postings.db", help="SQLite store path")
    parser.add_argument("--limit", type=int, default=50, help="max postings per source")
    parser.add_argument("--greenhouse", help="comma-separated Greenhouse board tokens")
    parser.add_argument("--lever", help="comma-separated Lever handles")
    parser.add_argument("--ashby", help="comma-separated Ashby org names")
    parser.add_argument("--no-arbeitnow", action="store_true", help="skip the Arbeitnow source")
    parser.add_argument("--visa-only", action="store_true", help="Arbeitnow: only postings it flags as sponsoring")
    parser.add_argument("--pages", type=int, default=1, help="Arbeitnow: pages to pull")
    parser.add_argument("--country", help="report: filter by country")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    agent = _build_agent(args)

    if args.command == "doctor":
        results = agent.doctor()
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print("Source check\n")
            for entry in results:
                mark = "ok  " if entry["ok"] else "FAIL"
                print(f"[{mark}] {entry['source']}")
                if entry.get("error"):
                    print(f"        {entry['error']}")
                if entry.get("sample"):
                    s = entry["sample"]
                    print(f"        {entry['count']} postings; sample: {s['role_title']!r} "
                          f"@ {s['company']!r} ({s['location'] or 'no location'}) — {s['text_chars']} chars")
                for warning in entry.get("warnings", []):
                    print(f"        warning: {warning}")
        return 0 if all(e["ok"] for e in results) else 1

    if args.command == "run":
        report = agent.run(limit_per_source=args.limit)
        print(json.dumps(report.to_dict(), indent=2) if args.json else report.digest())
        return 0

    rows = agent.store.all_postings(country=args.country, limit=200)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    stats = agent.store.stats()
    print(f"{stats['total']} postings · {stats['companies']} companies · "
          f"{stats['countries']} countries · {stats['sponsoring']} likely sponsoring\n")
    for row in rows[:50]:
        print(f"  {row['sponsorship_label']:<34} {row['company'][:24]:<26} "
              f"{(row['country'] or '?')[:18]:<20} {row['role_title'][:40]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
