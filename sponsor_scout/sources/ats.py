"""Greenhouse, Lever and Ashby — public job-board endpoints.

This is the cheapest real data in the project: any company hosting its
careers page on one of these ATSs serves its own live openings as JSON, free
and without a key. You choose the companies, so coverage is worldwide and
limited only by your list. The paid job-data APIs are, largely, this plus a
company list and a bill.

Company identifiers come from the careers-page URL:

    boards.greenhouse.io/COMPANY          -> Greenhouse board token
    jobs.lever.co/COMPANY                 -> Lever handle
    jobs.ashbyhq.com/COMPANY              -> Ashby org name
"""

from __future__ import annotations

from typing import List

from sponsor_scout.sources.base import (
    Posting,
    SourceAdapter,
    http_json,
    infer_country,
    normalise_date,
    strip_html,
)


class GreenhouseSource(SourceAdapter):
    """One Greenhouse board. `content=true` returns the full description,
    which is what the sponsorship heuristic needs to read."""

    name = "greenhouse"
    ENDPOINT = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"

    def __init__(self, board_token: str):
        self.board_token = board_token

    def describe(self) -> str:
        return f"greenhouse:{self.board_token}"

    def fetch(self, limit: int = 100) -> List[Posting]:
        payload = http_json(self.ENDPOINT.format(token=self.board_token))
        jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
        postings: List[Posting] = []
        for job in self._iter_limited(jobs, limit):
            location = (job.get("location") or {}).get("name", "") if isinstance(job.get("location"), dict) else str(job.get("location") or "")
            text = strip_html(job.get("content"))
            postings.append(
                Posting(
                    id=f"gh-{self.board_token}-{job.get('id')}",
                    company=self.board_token.replace("-", " ").title(),
                    role_title=job.get("title") or "",
                    country=infer_country(location),
                    location=location,
                    source=self.describe(),
                    source_url=job.get("absolute_url") or "",
                    date_posted=normalise_date(job.get("updated_at") or job.get("created_at")),
                    text=text or (job.get("title") or ""),
                    raw=job,
                )
            )
        return postings


class LeverSource(SourceAdapter):
    """One Lever board. Returns a flat list rather than an envelope."""

    name = "lever"
    ENDPOINT = "https://api.lever.co/v0/postings/{handle}?mode=json"

    def __init__(self, handle: str):
        self.handle = handle

    def describe(self) -> str:
        return f"lever:{self.handle}"

    def fetch(self, limit: int = 100) -> List[Posting]:
        payload = http_json(self.ENDPOINT.format(handle=self.handle))
        jobs = payload if isinstance(payload, list) else []
        postings: List[Posting] = []
        for job in self._iter_limited(jobs, limit):
            categories = job.get("categories") or {}
            location = categories.get("location") or ""
            description = strip_html(job.get("descriptionPlain") or job.get("description"))
            # Lever splits the posting across description + lists (requirements,
            # benefits). Sponsorship language often sits in those lists, so they
            # are appended rather than dropped.
            for block in job.get("lists") or []:
                description += " " + strip_html(block.get("text")) + " " + strip_html(block.get("content"))
            postings.append(
                Posting(
                    id=f"lv-{self.handle}-{job.get('id')}",
                    company=job.get("hostedUrlCompany") or self.handle.replace("-", " ").title(),
                    role_title=job.get("text") or "",
                    country=infer_country(location),
                    location=location,
                    source=self.describe(),
                    source_url=job.get("hostedUrl") or job.get("applyUrl") or "",
                    date_posted=normalise_date(job.get("createdAt")),
                    text=description.strip() or (job.get("text") or ""),
                    raw=job,
                )
            )
        return postings


class AshbySource(SourceAdapter):
    """One Ashby board. `includeCompensation` is left off — salary isn't part
    of the sponsorship question and it slows the response."""

    name = "ashby"
    ENDPOINT = "https://api.ashbyhq.com/posting-api/job-board/{org}?includeCompensation=false"

    def __init__(self, org: str):
        self.org = org

    def describe(self) -> str:
        return f"ashby:{self.org}"

    def fetch(self, limit: int = 100) -> List[Posting]:
        payload = http_json(self.ENDPOINT.format(org=self.org))
        jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
        postings: List[Posting] = []
        for job in self._iter_limited(jobs, limit):
            location = job.get("location") or ""
            text = strip_html(job.get("descriptionHtml") or job.get("descriptionPlain"))
            postings.append(
                Posting(
                    id=f"ab-{self.org}-{job.get('id')}",
                    company=job.get("companyName") or self.org.replace("-", " ").title(),
                    role_title=job.get("title") or "",
                    country=infer_country(location),
                    location=location,
                    source=self.describe(),
                    source_url=job.get("jobUrl") or job.get("applyUrl") or "",
                    date_posted=normalise_date(job.get("publishedAt")),
                    text=text or (job.get("title") or ""),
                    raw=job,
                )
            )
        return postings


class ArbeitnowSource(SourceAdapter):
    """Arbeitnow's free, key-less board API. It aggregates the same ATSs and
    carries its own `visa_sponsorship` boolean, which is kept as
    `provider_sponsorship` — a second opinion to compare the heuristic
    against, not a replacement for reading the posting."""

    name = "arbeitnow"
    ENDPOINT = "https://www.arbeitnow.com/api/job-board-api"

    def __init__(self, visa_only: bool = False, pages: int = 1):
        self.visa_only = visa_only
        self.pages = max(1, pages)

    def describe(self) -> str:
        return f"arbeitnow{'(visa only)' if self.visa_only else ''}"

    def fetch(self, limit: int = 100) -> List[Posting]:
        postings: List[Posting] = []
        for page in range(1, self.pages + 1):
            url = self.ENDPOINT if page == 1 else f"{self.ENDPOINT}?page={page}"
            payload = http_json(url)
            jobs = payload.get("data", []) if isinstance(payload, dict) else []
            for job in jobs:
                if len(postings) >= limit:
                    return postings
                sponsors = job.get("visa_sponsorship")
                if self.visa_only and sponsors is not True:
                    continue
                location = job.get("location") or ""
                postings.append(
                    Posting(
                        id=f"an-{job.get('slug') or job.get('url')}",
                        company=job.get("company_name") or "",
                        role_title=job.get("title") or "",
                        country=infer_country(location),
                        location=location,
                        source=self.name,
                        source_url=job.get("url") or "",
                        date_posted=normalise_date(job.get("created_at")),
                        text=strip_html(job.get("description")) or (job.get("title") or ""),
                        provider_sponsorship=sponsors if isinstance(sponsors, bool) else None,
                        raw=job,
                    )
                )
        return postings
