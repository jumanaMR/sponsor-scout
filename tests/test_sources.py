"""Adapter tests against recorded response shapes.

These use fixtures rather than live calls on purpose: CI must not depend on
four third-party APIs being up, and a test that silently passes because a
provider returned an empty list is worse than no test. The fixtures are the
documented response shape of each ATS — `agent doctor` is what verifies the
real endpoints still match.
"""

from __future__ import annotations

import json

import pytest

from sponsor_scout.sources import ats
from sponsor_scout.sources.ats import (
    ArbeitnowSource,
    AshbySource,
    GreenhouseSource,
    LeverSource,
)
from sponsor_scout.sources.base import infer_country, normalise_date, strip_html

SPONSOR_TEXT = (
    "<p>We are an <b>approved sponsor</b> and can support a subclass 482 visa "
    "for the right candidate, including offshore applicants.</p>"
)

GREENHOUSE_PAYLOAD = {
    "jobs": [
        {
            "id": 4567,
            "title": "Senior AI Engineer",
            "updated_at": "2026-09-01T10:30:00Z",
            "location": {"name": "Sydney, Australia"},
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/4567",
            "content": SPONSOR_TEXT,
        }
    ]
}

LEVER_PAYLOAD = [
    {
        "id": "abc-123",
        "text": "Machine Learning Engineer",
        "createdAt": 1788220800000,
        "categories": {"location": "Amsterdam, Netherlands"},
        "descriptionPlain": "Join our ML team.",
        "lists": [{"text": "Requirements", "content": "<li>We offer visa sponsorship.</li>"}],
        "hostedUrl": "https://jobs.lever.co/acme/abc-123",
    }
]

ASHBY_PAYLOAD = {
    "jobs": [
        {
            "id": "xyz-9",
            "title": "AI Platform Engineer",
            "location": "Berlin, Germany",
            "publishedAt": "2026-08-20T00:00:00Z",
            "descriptionHtml": "<div>We support the EU Blue Card process.</div>",
            "jobUrl": "https://jobs.ashbyhq.com/acme/xyz-9",
            "companyName": "Acme GmbH",
        }
    ]
}

ARBEITNOW_PAYLOAD = {
    "data": [
        {
            "slug": "ml-engineer-123",
            "title": "ML Engineer",
            "company_name": "Nordwind",
            "location": "Munich",
            "created_at": 1788220800,
            "description": "<p>We sponsor work permits.</p>",
            "url": "https://www.arbeitnow.com/jobs/ml-engineer-123",
            "visa_sponsorship": True,
        },
        {
            "slug": "data-eng-456",
            "title": "Data Engineer",
            "company_name": "Ferngully",
            "location": "Munich",
            "created_at": 1788220800,
            "description": "<p>No sponsorship available.</p>",
            "url": "https://www.arbeitnow.com/jobs/data-eng-456",
            "visa_sponsorship": False,
        },
    ]
}


@pytest.fixture
def stub_http(monkeypatch):
    """Route every adapter's HTTP call to a fixture, keyed by URL fragment."""
    routes = {
        "greenhouse": GREENHOUSE_PAYLOAD,
        "lever": LEVER_PAYLOAD,
        "ashby": ASHBY_PAYLOAD,
        "arbeitnow": ARBEITNOW_PAYLOAD,
    }

    def fake(url, timeout=25):
        for key, payload in routes.items():
            if key in url:
                return json.loads(json.dumps(payload))  # fresh copy per call
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(ats, "http_json", fake)
    return routes


# --- per-adapter parsing ---------------------------------------------------


def test_greenhouse_maps_every_field(stub_http):
    posting = GreenhouseSource("acme").fetch()[0]
    assert posting.id == "gh-acme-4567"
    assert posting.role_title == "Senior AI Engineer"
    assert posting.country == "Australia"
    assert posting.date_posted == "2026-09-01"
    assert posting.source_url.endswith("/4567")
    assert "approved sponsor" in posting.text
    assert "<b>" not in posting.text


def test_lever_includes_list_blocks_in_the_text(stub_http):
    """Lever keeps requirements in a separate `lists` array — sponsorship
    language often lives there, so dropping it loses the signal entirely."""
    posting = LeverSource("acme").fetch()[0]
    assert "visa sponsorship" in posting.text.lower()
    assert posting.country == "Netherlands"
    assert posting.date_posted == "2026-09-01"  # epoch millis


def test_ashby_maps_every_field(stub_http):
    posting = AshbySource("acme").fetch()[0]
    assert posting.company == "Acme GmbH"
    assert posting.country == "Germany"
    assert "Blue Card" in posting.text


def test_arbeitnow_keeps_the_provider_flag(stub_http):
    postings = ArbeitnowSource().fetch()
    assert [p.provider_sponsorship for p in postings] == [True, False]


def test_arbeitnow_visa_only_filters(stub_http):
    postings = ArbeitnowSource(visa_only=True).fetch()
    assert len(postings) == 1
    assert postings[0].company == "Nordwind"


def test_limit_is_respected(stub_http):
    assert len(ArbeitnowSource().fetch(limit=1)) == 1


# --- shared helpers --------------------------------------------------------


def test_strip_html_does_not_weld_sentences_together():
    """A naive tag strip turns '<p>One.</p><p>Two.</p>' into 'One.Two.',
    which breaks sentence splitting and therefore chunking."""
    assert strip_html("<p>One.</p><p>Two.</p>") == "One. Two."


def test_strip_html_decodes_entities():
    assert strip_html("R&amp;D team") == "R&D team"


@pytest.mark.parametrize(
    "location,expected",
    [
        ("Sydney, NSW", "Australia"),
        ("Amsterdam", "Netherlands"),
        ("Berlin, Germany", "Germany"),
        ("London, UK", "United Kingdom"),
        ("Dublin", "Ireland"),
        ("Toronto, Canada", "Canada"),
        ("Doha", "Qatar"),
        ("", "Unknown"),
        ("Somewhere Fictional", "Unknown"),
    ],
)
def test_country_inference(location, expected):
    assert infer_country(location) == expected


def test_unknown_country_is_explicit_not_a_guess():
    """Filtering on a wrong country is worse than filtering on 'Unknown' —
    the user can see the gap and go read the posting."""
    assert infer_country("Planet Zog") == "Unknown"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2026-09-01T10:30:00Z", "2026-09-01"),
        (1788220800000, "2026-09-01"),  # epoch millis
        (1788220800, "2026-09-01"),     # epoch seconds
        ("2026-09-01", "2026-09-01"),
    ],
)
def test_date_normalisation(value, expected):
    assert normalise_date(value) == expected


def test_unparseable_date_does_not_raise():
    assert len(normalise_date("not a date at all")) == 10
