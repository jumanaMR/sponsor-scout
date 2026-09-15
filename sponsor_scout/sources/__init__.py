"""Source adapters and the registry that builds them from config."""

from __future__ import annotations

from typing import Dict, List

from sponsor_scout.sources.ats import (
    ArbeitnowSource,
    AshbySource,
    GreenhouseSource,
    LeverSource,
)
from sponsor_scout.sources.base import (
    Posting,
    SourceAdapter,
    SourceError,
    infer_country,
    strip_html,
)

# Companies to poll, by ATS. These are starting points chosen because they
# hire engineers internationally — being on this list is NOT a claim that a
# company sponsors visas; that is what the pipeline works out from the
# postings themselves. Edit freely: the identifier is the last path segment
# of the company's careers URL.
DEFAULT_COMPANIES: Dict[str, List[str]] = {
    "greenhouse": ["canva", "atlassian", "airtable", "databricks", "elastic"],
    "lever": ["spotify", "netflix"],
    "ashby": ["ramp", "linear"],
}


def build_sources(
    greenhouse: List[str] | None = None,
    lever: List[str] | None = None,
    ashby: List[str] | None = None,
    arbeitnow: bool = True,
    arbeitnow_visa_only: bool = False,
    arbeitnow_pages: int = 1,
) -> List[SourceAdapter]:
    """Assemble the adapter list for a run. Passing nothing gives the
    defaults above plus Arbeitnow."""
    sources: List[SourceAdapter] = []
    for token in greenhouse if greenhouse is not None else DEFAULT_COMPANIES["greenhouse"]:
        sources.append(GreenhouseSource(token))
    for handle in lever if lever is not None else DEFAULT_COMPANIES["lever"]:
        sources.append(LeverSource(handle))
    for org in ashby if ashby is not None else DEFAULT_COMPANIES["ashby"]:
        sources.append(AshbySource(org))
    if arbeitnow:
        sources.append(ArbeitnowSource(visa_only=arbeitnow_visa_only, pages=arbeitnow_pages))
    return sources


__all__ = [
    "ArbeitnowSource",
    "AshbySource",
    "GreenhouseSource",
    "LeverSource",
    "Posting",
    "SourceAdapter",
    "SourceError",
    "DEFAULT_COMPANIES",
    "build_sources",
    "infer_country",
    "strip_html",
]
