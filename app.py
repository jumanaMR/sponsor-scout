"""Streamlit UI for Sponsor Scout.

Run with:
    pip install -r requirements.txt
    streamlit run app.py

Everything here is a thin UI over sponsor_scout/pipeline.py — no retrieval or
scoring logic lives in this file, so the notebook, this app, and any
future FastAPI service all share one implementation.
"""

import os
import tempfile

import pandas as pd
import streamlit as st

from sponsor_scout.pipeline import (
    NewPostingWatcher,
    SponsorshipRAG,
    extract_text_from_file,
    sponsorship_signal,
)
from sponsor_scout.sample_data import SAMPLE_POSTINGS

SIGNAL_COLORS = {
    "likely sponsoring": "#2f7d4f",
    "likely NOT sponsoring": "#a6432f",
    "mixed signal - verify manually": "#b8722e",
    "no clear sponsorship language - unclear, verify manually": "#b8722e",
    "no sponsorship language found": "#6b7280",
}

st.set_page_config(page_title="Sponsor Scout", page_icon="🧭", layout="wide")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _init_state():
    if "postings" not in st.session_state:
        st.session_state.postings = list(SAMPLE_POSTINGS)
        st.session_state.using_samples = True
    if "view_history" not in st.session_state:
        st.session_state.view_history = []
    if "cv_text" not in st.session_state:
        st.session_state.cv_text = ""


def build_rag() -> SponsorshipRAG:
    """Rebuild the index from current state. Cheap at this corpus size;
    a persistent store (see the roadmap) is what removes the rebuild."""
    rag = SponsorshipRAG(n_components=50)
    rag.add_documents(st.session_state.postings, chunk_size=60, overlap=15)
    rag.build_index()
    for doc_id in st.session_state.view_history:
        rag.record_view(doc_id)
    if st.session_state.cv_text.strip():
        rag.load_cv(st.session_state.cv_text)
    return rag


def signal_badge(label: str) -> str:
    color = SIGNAL_COLORS.get(label, "#6b7280")
    short = label.split(" - ")[0]
    return (
        f"<span style='background:{color}1a;color:{color};padding:2px 8px;"
        f"border-radius:10px;font-size:12px;font-weight:600'>{short}</span>"
    )


_init_state()

# ---------------------------------------------------------------------------
# Sidebar — corpus management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Corpus")
    if st.session_state.using_samples:
        st.info(
            "Showing 12 **synthetic sample postings** — made-up companies, "
            "not real leads. Add or import real postings to replace them.",
            icon="⚠️",
        )
    st.caption(f"{len(st.session_state.postings)} postings indexed")

    st.markdown("#### Import from CSV")
    uploaded_csv = st.file_uploader(
        "Columns: company, role_title, country, source, date_posted, text",
        type=["csv"],
        key="csv_upload",
    )
    if uploaded_csv is not None and st.button("Import CSV", use_container_width=True):
        df = pd.read_csv(uploaded_csv)
        start = 0 if st.session_state.using_samples else len(st.session_state.postings)
        new_docs = [
            {
                "id": f"real-{start + i:03d}",
                "company": row.get("company", "Unknown"),
                "country": row.get("country", ""),
                "role_title": row.get("role_title", ""),
                "source": row.get("source", ""),
                "date_posted": str(row.get("date_posted", "")),
                "text": str(row.get("text", "")),
            }
            for i, row in df.iterrows()
        ]
        if st.session_state.using_samples:
            st.session_state.postings = new_docs
            st.session_state.using_samples = False
        else:
            st.session_state.postings.extend(new_docs)
        st.success(f"Imported {len(new_docs)} posting(s)")
        st.rerun()

    st.markdown("#### Add one posting")
    with st.form("add_posting", clear_on_submit=True):
        company = st.text_input("Company")
        role_title = st.text_input("Role title")
        country = st.text_input("Country")
        source = st.text_input("Source URL")
        text = st.text_area("Full posting text", height=140)
        if st.form_submit_button("Add posting", use_container_width=True):
            if company and text:
                if st.session_state.using_samples:
                    st.session_state.postings = []
                    st.session_state.using_samples = False
                st.session_state.postings.append(
                    {
                        "id": f"real-{len(st.session_state.postings):03d}",
                        "company": company,
                        "country": country,
                        "role_title": role_title,
                        "source": source,
                        "date_posted": pd.Timestamp.today().strftime("%Y-%m-%d"),
                        "text": text,
                    }
                )
                st.success(f"Added {company}")
                st.rerun()
            else:
                st.error("Company and posting text are both required.")

    if st.session_state.view_history:
        st.markdown("#### View history")
        st.caption(" → ".join(st.session_state.view_history[-6:]))
        if st.button("Clear history", use_container_width=True):
            st.session_state.view_history = []
            st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

rag = build_rag()
report = rag.company_report(st.session_state.postings)
sponsoring = int((report["sponsorship_signal"] == "likely sponsoring").sum())

st.title("Sponsor Scout")
st.caption(
    "Retrieval over job postings, scored for visa-sponsorship signal — "
    "search, recommendations, and CV matching over your own corpus."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Postings", len(st.session_state.postings))
c2.metric("Likely sponsoring", sponsoring)
c3.metric("Countries", report["country"].nunique())
c4.metric("Views recorded", len(st.session_state.view_history))

tab_browse, tab_search, tab_recs, tab_cv, tab_new = st.tabs(
    ["Browse", "Search", "Recommended", "CV match", "What's new"]
)

# ---------------------------------------------------------------------------
# Browse
# ---------------------------------------------------------------------------

with tab_browse:
    st.markdown("#### All postings")
    st.caption(
        "Opening a posting records a view, which feeds the recommender. "
        "Only an explicit click counts — nothing is recorded just by scrolling."
    )

    countries = ["All"] + sorted(report["country"].dropna().unique().tolist())
    picked = st.selectbox("Filter by country", countries)

    for doc in st.session_state.postings:
        if picked != "All" and doc.get("country") != picked:
            continue
        sig = sponsorship_signal(doc["text"])
        left, right = st.columns([5, 1])
        left.markdown(
            f"**{doc['company']}** · {doc.get('role_title', '')} · "
            f"{doc.get('country', '')} &nbsp; {signal_badge(sig['label'])}",
            unsafe_allow_html=True,
        )
        if right.button("Open", key=f"open_{doc['id']}", use_container_width=True):
            st.session_state.selected = doc["id"]
            st.session_state.view_history.append(doc["id"])
            st.rerun()

        if st.session_state.get("selected") == doc["id"]:
            st.write(doc["text"])
            evidence = sig["positive_matches"] + sig["negative_matches"]
            if evidence:
                st.caption("Matched phrases: " + ", ".join(evidence))
            if doc.get("source"):
                st.caption(f"Source: {doc['source']}")
        st.divider()

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

with tab_search:
    st.markdown("#### Semantic search")
    query = st.text_input(
        "Query",
        value="AI engineer roles with 482 visa sponsorship in Australia",
        label_visibility="collapsed",
    )
    top_k = st.slider("Results", 3, 12, 6, key="search_k")
    if query.strip():
        results = rag.query(query, top_k=top_k)
        for _, row in results.iterrows():
            sig = sponsorship_signal(row["chunk_text"])
            st.markdown(
                f"**{row.get('company', '?')}** · {row.get('country', '?')} · "
                f"`{row['score']:.2f}` &nbsp; {signal_badge(sig['label'])}",
                unsafe_allow_html=True,
            )
            st.caption(row["chunk_text"])
            st.divider()
        st.info(
            "The score ranks by similarity to your wording — it doesn't know "
            "whether a match is a yes or a no on sponsorship. Read the badge, "
            "not just the ranking.",
            icon="ℹ️",
        )

# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

with tab_recs:
    st.markdown("#### Recommended next")
    has_cv = bool(st.session_state.cv_text.strip())
    cv_weight = st.slider(
        "How much your CV counts vs. what you've viewed",
        0.0, 1.0, 0.4 if has_cv else 0.0, 0.1,
        disabled=not has_cv,
        help="0 = purely view history, 1 = purely CV. Load a CV to enable.",
    )
    # Gate on the index's own history, not session_state — that's what
    # recommend_next() actually reads.
    if not rag.view_history and not has_cv:
        st.warning(
            "No signal yet — open a posting on the Browse tab, or load your "
            "CV on the CV match tab.",
            icon="👀",
        )
    else:
        recs = rag.recommend_next(
            top_k=8,
            cv_weight=cv_weight if has_cv else 0.0,
        )
        for _, row in recs.iterrows():
            st.markdown(
                f"**{row['company']}** · {row['country']} · {row['role_title']} "
                f"&nbsp;·&nbsp; `{row['score']:.2f}` &nbsp; "
                f"{signal_badge(row['sponsorship_signal'])}",
                unsafe_allow_html=True,
            )
            st.caption(f"Because you viewed: {row['because_you_viewed']}")
            st.divider()

# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------

with tab_cv:
    st.markdown("#### Your CV")
    cv_file = st.file_uploader("Upload .pdf, .docx or .txt", type=["pdf", "docx", "txt"])
    if cv_file is not None:
        suffix = os.path.splitext(cv_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(cv_file.getbuffer())
            tmp_path = tmp.name
        try:
            st.session_state.cv_text = extract_text_from_file(tmp_path)
            st.success(f"Loaded {cv_file.name} ({len(st.session_state.cv_text)} chars)")
        except Exception as exc:  # noqa: BLE001 - surfaced to the user directly
            st.error(f"Couldn't read that file: {exc}")
        finally:
            os.unlink(tmp_path)

    st.session_state.cv_text = st.text_area(
        "…or paste your CV text",
        value=st.session_state.cv_text,
        height=180,
    )

    if st.session_state.cv_text.strip():
        st.markdown("#### Postings ranked against your CV")
        matches = rag.cv_match_report(top_k=10)
        for _, row in matches.iterrows():
            st.markdown(
                f"**{row['company']}** · {row['country']} · {row['role_title']} "
                f"&nbsp;·&nbsp; `{row['score']:.2f}` &nbsp; "
                f"{signal_badge(row['sponsorship_signal'])}",
                unsafe_allow_html=True,
            )
            st.divider()

# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------

with tab_new:
    st.markdown("#### Postings new since your last check")
    st.caption(
        "Backed by NewPostingWatcher, which persists seen ids to "
        "`seen_postings.json` — the same function a scheduled job would call "
        "before emailing you a digest."
    )
    watcher = NewPostingWatcher(state_path="seen_postings.json")
    preview = watcher.check(
        rag,
        st.session_state.postings,
        mark_seen=False,
        cv_weight=0.5,
    )
    if preview.empty:
        st.success("Nothing new since your last check.", icon="✅")
    else:
        st.dataframe(preview, use_container_width=True, hide_index=True)
        if st.button(f"Mark all {len(preview)} as seen"):
            watcher.check(rag, st.session_state.postings, mark_seen=True, cv_weight=0.5)
            st.rerun()
