"""Source adapters: one small class per free job-data provider.

Every adapter turns a provider's own JSON into the same `Posting` shape, so
the rest of the pipeline never learns where a posting came from. Adding a
provider means adding one file here, not touching the agent or the index.

All sources used here are FREE and need no API key:

  Greenhouse / Lever / Ashby  public job-board endpoints that any company
                              using those ATSs exposes for its own careers
                              page. You pick the companies; the data comes
                              straight from the employer.
  Arbeitnow                   an aggregator over those same ATSs, with a
                              visa-sponsorship flag of its own.

Parsing is deliberately defensive (`.get` with fallbacks, never an index
into a list that might be empty): these endpoints are public and can change
their field names without warning, and a missing field should degrade one
posting, not crash a scheduled run.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

USER_AGENT = "sponsor-scout/0.2 (personal job search; +https://github.com/jumanaMR/sponsor-scout)"
TIMEOUT_SECONDS = 25


@dataclass
class Posting:
    """The one shape the rest of the pipeline understands.

    `id` must be stable across runs for the same posting — the watcher's
    whole job is telling new from already-seen, and an id that changes each
    poll would re-alert you every time.
    """

    id: str
    company: str
    role_title: str
    country: str
    location: str
    source: str
    source_url: str
    date_posted: str
    text: str
    provider_sponsorship: Optional[bool] = None  # what the source itself claims, if anything
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not include_raw:
            data.pop("raw", None)
        return data


class SourceError(RuntimeError):
    """A provider call failed. Carries the source name so a multi-source run
    can report which one broke and carry on with the rest."""

    def __init__(self, source: str, message: str):
        super().__init__(f"[{source}] {message}")
        self.source = source


def http_json(url: str, timeout: int = TIMEOUT_SECONDS) -> Any:
    """GET a URL and parse JSON. Kept dependency-free on purpose — the whole
    ingestion layer runs on the standard library, so deploying it needs no
    extra packages beyond the pipeline's own."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        raise SourceError(url, f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SourceError(url, f"unreachable: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise SourceError(url, f"response was not JSON: {exc}") from exc


TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def strip_html(value: Any) -> str:
    """ATS descriptions come back as HTML. The sponsorship heuristic reads
    plain prose, so tags are noise — but the text between them matters, and
    a naive strip can weld two sentences together, so tags become spaces."""
    if not value:
        return ""
    text = TAG_RE.sub(" ", str(value))
    text = (
        text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
        .replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
    )
    return WS_RE.sub(" ", text).strip()


# Country inference -----------------------------------------------------------
# ATS location strings are free text ("Sydney, NSW", "Remote - EMEA"). This is
# a lookup, not a geocoder: it recognises the countries actually being targeted
# and otherwise says so, rather than guessing wrong.

COUNTRY_HINTS = {
    "Australia": ["australia", "sydney", "melbourne", "brisbane", "perth", "canberra", "adelaide", " nsw", " vic", " qld", " wa,"],
    "New Zealand": ["new zealand", "auckland", "wellington", "christchurch"],
    "Netherlands": ["netherlands", "amsterdam", "rotterdam", "utrecht", "eindhoven", "the hague"],
    "Germany": ["germany", "deutschland", "berlin", "munich", "münchen", "hamburg", "frankfurt", "cologne", "köln"],
    "United Kingdom": ["united kingdom", "england", "scotland", "wales", " uk", "uk,", "london", "manchester", "edinburgh", "bristol", "cambridge", "glasgow"],
    "Ireland": ["ireland", "dublin", "cork", "galway"],
    "Canada": ["canada", "toronto", "vancouver", "montreal", "ottawa", "calgary", "waterloo"],
    "United States": ["united states", " usa", "usa,", " us,", "new york", "san francisco", "seattle", "austin", "boston", "chicago", "denver", "atlanta", "los angeles"],
    "Singapore": ["singapore"],
    "United Arab Emirates": ["united arab emirates", "dubai", "abu dhabi"],
    "Qatar": ["qatar", "doha"],
    "India": ["india", "bangalore", "bengaluru", "hyderabad", "mumbai", "delhi", "pune", "chennai"],
    "France": ["france", "paris", "lyon", "toulouse"],
    "Spain": ["spain", "madrid", "barcelona", "valencia"],
    "Switzerland": ["switzerland", "zurich", "zürich", "geneva", "lausanne"],
    "Sweden": ["sweden", "stockholm", "gothenburg"],
    "Denmark": ["denmark", "copenhagen"],
    "Norway": ["norway", "oslo"],
    "Poland": ["poland", "warsaw", "krakow", "kraków", "wroclaw"],
    "Portugal": ["portugal", "lisbon", "porto"],
    "Belgium": ["belgium", "brussels", "antwerp", "ghent"],
    "Austria": ["austria", "vienna", "wien"],
    "Japan": ["japan", "tokyo", "osaka"],
    "Brazil": ["brazil", "são paulo", "sao paulo", "rio de janeiro"],
    "Mexico": ["mexico", "méxico", "mexico city", "guadalajara"],
}

UNKNOWN_COUNTRY = "Unknown"


def infer_country(location: str) -> str:
    """Best-effort country from a free-text location string. Returns
    'Unknown' rather than guessing — an unknown country the user can filter
    on is more useful than a confidently wrong one."""
    if not location:
        return UNKNOWN_COUNTRY
    haystack = f" {location.lower()} "
    for country, hints in COUNTRY_HINTS.items():
        if any(hint in haystack for hint in hints):
            return country
    return UNKNOWN_COUNTRY


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def normalise_date(value: Any) -> str:
    """Providers use ISO strings, epoch seconds and epoch milliseconds. All
    become YYYY-MM-DD, or today's date when unparseable."""
    if value in (None, "", 0):
        return today()
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 1e11 else value
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%d")
        except (ValueError, OSError, OverflowError):
            return today()
    text = str(value)
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[: len(fmt) + 2].rstrip("Z"), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text[:10] if len(text) >= 10 else today()


class SourceAdapter:
    """Base class. A subclass implements `fetch()` and nothing else."""

    name: str = "base"

    def fetch(self, limit: int = 100) -> List[Posting]:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> str:
        return self.name

    @staticmethod
    def _iter_limited(items: Iterable[Any], limit: int) -> Iterable[Any]:
        for index, item in enumerate(items):
            if index >= limit:
                return
            yield item
