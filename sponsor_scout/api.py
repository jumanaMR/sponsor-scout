"""FastAPI service — the production face of the pipeline.

    uvicorn sponsor_scout.api:app --reload

Endpoints are deliberately thin: they read the SQLite store the agent
writes, and call the same pipeline the notebook and Streamlit app use. No
scoring logic lives here.

The agent and the API are separate processes on purpose. Ingestion is slow,
periodic and allowed to fail; serving is fast, constant and must not. A
request should never be waiting on a third-party job board.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "FastAPI is not installed. Install the API extras:\n"
        "    pip install fastapi uvicorn"
    ) from exc

from sponsor_scout.pipeline import SponsorshipRAG, sponsorship_signal
from sponsor_scout.store import PostingStore

DB_PATH = os.environ.get("SPONSOR_SCOUT_DB", "data/postings.db")

app = FastAPI(
    title="Sponsor Scout",
    version="0.2.0",
    description="Job postings scored for visa-sponsorship signal, filterable by country.",
)

# The web app is served from a different origin (a published artifact, or a
# static host), so it needs CORS. Lock the origins down before this is
# public — "*" is fine for a personal tool on localhost, not for a service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("SPONSOR_SCOUT_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SignalRequest(BaseModel):
    text: str


def get_store() -> PostingStore:
    return PostingStore(DB_PATH)


# --- read endpoints --------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, Any]:
    store = get_store()
    return {"status": "ok", "postings": len(store), "db": DB_PATH}


@app.get("/stats")
def stats() -> Dict[str, Any]:
    return get_store().stats()


@app.get("/countries")
def countries() -> List[Dict[str, Any]]:
    """Every country present, with totals — this is what populates the
    location filter in the UI, rather than a hardcoded list."""
    return get_store().countries()


@app.get("/postings")
def postings(
    country: Optional[str] = Query(None, description="Country name, or omit for all"),
    label: Optional[str] = Query(None, description="e.g. 'likely sponsoring'"),
    limit: int = Query(100, ge=1, le=1000),
) -> Dict[str, Any]:
    rows = get_store().all_postings(country=country, label=label, limit=limit)
    return {"count": len(rows), "postings": [_public(r) for r in rows]}


@app.get("/search")
def search(
    q: str = Query(..., min_length=2, description="Free-text query"),
    country: Optional[str] = None,
    top_k: int = Query(10, ge=1, le=50),
) -> Dict[str, Any]:
    """Semantic search over the stored corpus.

    The index is rebuilt per request, which is fine at a few thousand
    postings and honest about what this is: swapping in pgvector (roadmap
    stage 2) is what removes the rebuild.
    """
    rows = get_store().all_postings(country=country, limit=2000)
    if not rows:
        return {"count": 0, "results": [], "note": "No postings stored yet — run the agent first."}

    index = _build_index(rows)
    results = index.query(q, top_k=top_k)
    return {
        "count": len(results),
        "results": [
            {
                "score": float(row["score"]),
                "company": row.get("company"),
                "country": row.get("country"),
                "role_title": row.get("role_title"),
                "source_url": row.get("source_url"),
                "sponsorship_label": row.get("sponsorship_label"),
                "chunk": row["chunk_text"],
            }
            for _, row in results.iterrows()
        ],
    }


@app.get("/recommend")
def recommend(
    viewed: str = Query("", description="Comma-separated posting ids you've opened"),
    country: Optional[str] = None,
    top_k: int = Query(10, ge=1, le=50),
) -> Dict[str, Any]:
    rows = get_store().all_postings(country=country, limit=2000)
    if not rows:
        raise HTTPException(status_code=404, detail="No postings stored yet — run the agent first.")

    viewed_ids = [v for v in (viewed.split(",") if viewed else []) if v]
    if not viewed_ids:
        raise HTTPException(
            status_code=400,
            detail="Pass at least one viewed posting id, e.g. /recommend?viewed=gh-canva-123",
        )

    index = _build_index(rows)
    for doc_id in viewed_ids:
        index.record_view(doc_id)
    try:
        recs = index.recommend_next(top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"count": len(recs), "results": recs.to_dict(orient="records")}


@app.post("/signal")
def signal(request: SignalRequest) -> Dict[str, Any]:
    """Run the sponsorship heuristic on arbitrary text — paste a posting you
    found elsewhere and get the same label and evidence the pipeline uses."""
    result = sponsorship_signal(request.text)
    return {
        "label": result["label"],
        "positive_matches": result["positive_matches"],
        "negative_matches": result["negative_matches"],
    }


# --- helpers ---------------------------------------------------------------


def _build_index(rows: List[Dict[str, Any]]) -> SponsorshipRAG:
    index = SponsorshipRAG(n_components=100)
    index.add_documents(
        [
            {
                "id": r["id"],
                "company": r.get("company", ""),
                "country": r.get("country", ""),
                "role_title": r.get("role_title", ""),
                "source": r.get("source", ""),
                "source_url": r.get("source_url", ""),
                "date_posted": r.get("date_posted", ""),
                "text": r.get("text", "") or r.get("role_title", ""),
            }
            for r in rows
        ]
    )
    index.build_index()
    return index


def _public(row: Dict[str, Any]) -> Dict[str, Any]:
    """Trim the full description out of list responses — it's large, and the
    list view never shows it. /postings?limit=1000 with full text would be
    megabytes."""
    trimmed = {k: v for k, v in row.items() if k != "text"}
    trimmed["excerpt"] = (row.get("text") or "")[:280]
    return trimmed
