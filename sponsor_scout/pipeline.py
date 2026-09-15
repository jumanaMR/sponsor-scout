# rag_pipeline.py
#
# A small RAG (retrieval-augmented generation) pipeline for finding
# visa-sponsorship signal in job postings: chunking -> embedding ->
# vector index -> similarity search -> sponsorship heuristic -> answer.
#
# EMBEDDING BACKEND
# -----------------
# Embedding is delegated to sponsor_scout.embeddings, which implements two
# backends behind one small interface (EmbeddingBackend): TfidfSvdBackend
# (classic LSA — no extra dependency, no model download, works fully
# offline — this project's original backend, built in an environment with
# no outbound PyPI access) and SentenceTransformerBackend (real neural
# embeddings via the optional `sentence-transformers` package). `embedding_
# backend="auto"` (the default) prefers the neural backend and falls back
# to TF-IDF automatically when sentence-transformers isn't installed or its
# model can't be fetched. See sponsor_scout/embeddings.py for the details.
#
# To also upgrade the vector index itself (currently plain numpy cosine
# similarity, rebuilt per request — fine at a few thousand postings), swap
# it for a FAISS or pgvector index:
#
#     import faiss
#     index = faiss.IndexFlatIP(vectors.shape[1])
#     index.add(vectors)
#     scores, ids = index.search(query_vector, top_k)
#
# Nothing else in this file needs to change — add_documents/query/answer
# keep the same interface either way.

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from sponsor_scout.embeddings import EmbeddingBackend, resolve_backend

# ---------------------------------------------------------------------------
# File loading (CV upload)
# ---------------------------------------------------------------------------


def extract_text_from_file(path: str) -> str:
    """Extract plain text from a .txt/.md, .pdf, or .docx file — used to
    load a CV from whatever format it's actually saved in."""
    ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    if ext in ("txt", "md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == "pdf":
        import pypdf

        reader = pypdf.PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == "docx":
        import docx

        document = docx.Document(path)
        return "\n".join(p.text for p in document.paragraphs)
    raise ValueError(f"Unsupported CV file type: '.{ext}' - use .txt, .pdf, or .docx")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, max_words: int = 60, overlap_words: int = 15) -> List[str]:
    """Pack sentences into ~max_words chunks, each overlapping the previous
    chunk's tail by overlap_words so retrieval doesn't lose context at a
    chunk boundary."""
    sentences = split_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    for sent in sentences:
        sent_words = sent.split()
        if current and len(current) + len(sent_words) > max_words:
            chunks.append(" ".join(current))
            overlap = current[-overlap_words:] if overlap_words else []
            current = overlap + sent_words
        else:
            current.extend(sent_words)
    if current:
        chunks.append(" ".join(current))
    return chunks or ([text.strip()] if text.strip() else [])


# ---------------------------------------------------------------------------
# Sponsorship heuristic
# ---------------------------------------------------------------------------

POSITIVE_PHRASES = [
    "will sponsor", "can sponsor", "able to sponsor", "visa sponsorship",
    "sponsor a visa", "sponsor a work permit", "approved sponsor",
    "licensed sponsor", "registered sponsor", "blue card", "482",
    "skilled worker visa", "global talent stream", "lmia", "relocation",
    "work permit sponsorship", "sponsor a skilled visa", "sponsor relocation",
]

NEGATIVE_PHRASES = [
    "unable to sponsor", "not able to offer visa", "do not sponsor",
    "no sponsorship", "not currently registered as a sponsor",
    "cannot sponsor", "must already have the right to work",
    "existing work rights required", "not able to provide visa sponsorship",
    "not able to sponsor",
]

# Marks a posting as genuinely unclear (rather than silently letting a
# negated positive phrase get counted as a real positive signal).
AMBIGUOUS_MARKERS = [
    "unclear", "not specified", "tbd", "to be confirmed", "unsure",
    "contact recruiter", "recommend contacting",
]

# Words that, found shortly before a positive phrase, negate it — e.g. "no
# mention of visa sponsorship" should not count as a positive match on
# "visa sponsorship".
NEGATION_WORDS = {"no", "not", "n't", "without", "never", "unclear", "unsure"}


def _phrase_present(lower_text: str, phrase: str) -> bool:
    """Substring match that respects the START of a word, so the positive
    phrase 'able to sponsor' does not fire inside 'unable to sponsor'.
    The end is deliberately left open so 'approved sponsor' still matches
    'approved sponsors'."""
    for match in re.finditer(re.escape(phrase), lower_text):
        before = lower_text[match.start() - 1] if match.start() else " "
        if not before.isalpha():
            return True
    return False


def _is_negated(lower_text: str, phrase: str, window: int = 4) -> bool:
    idx = lower_text.find(phrase)
    if idx == -1:
        return False
    preceding_words = re.findall(r"[a-z']+", lower_text[:idx])[-window:]
    return any(w in NEGATION_WORDS or w.endswith("n't") for w in preceding_words)


def sponsorship_signal(text: str) -> Dict:
    """Cheap keyword heuristic — NOT a substitute for reading the posting
    yourself. Flags language likely to indicate sponsorship is or isn't
    offered, so you know which postings are worth a closer look. Accounts
    for simple negation (e.g. "no mention of visa sponsorship" doesn't
    count as a positive hit) but is still just pattern matching — always
    verify manually before relying on it."""
    lower = text.lower()
    pos_hits = [
        p for p in POSITIVE_PHRASES
        if _phrase_present(lower, p) and not _is_negated(lower, p)
    ]
    neg_hits = [p for p in NEGATIVE_PHRASES if _phrase_present(lower, p)]
    ambiguous_hits = [p for p in AMBIGUOUS_MARKERS if _phrase_present(lower, p)]

    if pos_hits and not neg_hits:
        label = "likely sponsoring"
    elif neg_hits and not pos_hits:
        label = "likely NOT sponsoring"
    elif pos_hits and neg_hits:
        label = "mixed signal - verify manually"
    elif ambiguous_hits:
        label = "no clear sponsorship language - unclear, verify manually"
    else:
        label = "no sponsorship language found"
    return {"label": label, "positive_matches": pos_hits, "negative_matches": neg_hits}


# How much a recommendation's score is nudged by its sponsorship_signal
# label, as a multiplier on `sponsorship_boost_weight` in recommend_next().
SPONSORSHIP_BOOST = {
    "likely sponsoring": 1.0,
    "mixed signal - verify manually": 0.5,
    "no clear sponsorship language - unclear, verify manually": 0.0,
    "no sponsorship language found": 0.0,
    "likely NOT sponsoring": -1.0,
}


# ---------------------------------------------------------------------------
# Vector store / RAG pipeline
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict


class SponsorshipRAG:
    def __init__(self, n_components: int = 100, embedding_backend: str = "auto"):
        """embedding_backend: 'auto' (default, prefers neural embeddings via
        sentence-transformers and falls back to TF-IDF+SVD when that's not
        installed or its model can't be fetched), 'tfidf', or
        'sentence-transformers' to force one. n_components only affects the
        tfidf backend (SVD dimensionality) — sentence-transformers uses its
        model's native embedding size."""
        self._requested_components = n_components
        self._embedding_backend_name = embedding_backend
        self.backend: Optional[EmbeddingBackend] = None
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None
        self._fitted = False
        self.view_history: List[str] = []  # doc_ids, oldest first
        self.cv_vector: Optional[np.ndarray] = None
        self.cv_text: Optional[str] = None

    def add_documents(self, docs: List[Dict], chunk_size: int = 60, overlap: int = 15):
        for doc in docs:
            for i, ctext in enumerate(chunk_text(doc["text"], max_words=chunk_size, overlap_words=overlap)):
                self.chunks.append(
                    Chunk(
                        chunk_id=f"{doc['id']}::chunk{i}",
                        doc_id=doc["id"],
                        text=ctext,
                        metadata={k: v for k, v in doc.items() if k != "text"},
                    )
                )

    def build_index(self):
        if not self.chunks:
            raise ValueError("No chunks to index - call add_documents first.")
        texts = [c.text for c in self.chunks]

        self.backend = resolve_backend(self._embedding_backend_name, n_components=self._requested_components)
        dense = self.backend.fit_transform(texts)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self.vectors = dense / norms
        self._fitted = True
        return self

    def _embed_query(self, query: str) -> np.ndarray:
        if not self._fitted or self.backend is None:
            raise ValueError("Index not built - call build_index() first.")
        dense = self.backend.transform([query])
        norm = np.linalg.norm(dense, axis=1, keepdims=True)
        norm[norm == 0] = 1e-9
        return dense / norm

    def query(self, query_text: str, top_k: int = 5) -> pd.DataFrame:
        qvec = self._embed_query(query_text)
        sims = cosine_similarity(qvec, self.vectors)[0]
        order = np.argsort(-sims)[:top_k]
        rows = []
        for idx in order:
            c = self.chunks[idx]
            row = {"score": round(float(sims[idx]), 4), "chunk_text": c.text}
            row.update(c.metadata)
            rows.append(row)
        return pd.DataFrame(rows)

    def company_report(self, docs: List[Dict]) -> pd.DataFrame:
        """Full-document (not chunk-level) sponsorship signal per posting."""
        rows = []
        for doc in docs:
            sig = sponsorship_signal(doc["text"])
            rows.append(
                {
                    "company": doc.get("company"),
                    "country": doc.get("country"),
                    "role_title": doc.get("role_title"),
                    "sponsorship_signal": sig["label"],
                    "evidence_phrases": ", ".join(sig["positive_matches"] + sig["negative_matches"]) or "-",
                    "source": doc.get("source"),
                    "date_posted": doc.get("date_posted"),
                }
            )
        return pd.DataFrame(rows).sort_values("sponsorship_signal")

    def answer(self, query_text: str, top_k: int = 5) -> str:
        """Retrieval + extractive synthesis (no LLM call — pure template).
        Swap this for a real LLM call (Claude/OpenAI) over the retrieved
        chunks once you have API access, for a more natural-language answer."""
        results = self.query(query_text, top_k=top_k)
        if results.empty:
            return "No matching postings found."
        lines = [f'Query: "{query_text}"', ""]
        seen = set()
        for _, row in results.iterrows():
            company = row.get("company", "Unknown")
            if company in seen:
                continue
            seen.add(company)
            sig = sponsorship_signal(row["chunk_text"])
            lines.append(
                f"- {company} ({row.get('country', '?')}, {row.get('role_title', '?')}) "
                f"- match score {row['score']:.2f} - {sig['label']}"
            )
            snippet = row["chunk_text"]
            if len(snippet) > 220:
                snippet = snippet[:220].rsplit(" ", 1)[0] + "..."
            lines.append(f'  "{snippet}"')
        return "\n".join(lines)

    # -- Recommendations ("because you looked at...") -----------------------

    def _doc_level_vectors(self):
        """Mean-pool this doc's chunk vectors into one vector per doc_id, so
        recommendations compare whole postings rather than individual
        chunks. Returns (vectors, metadata, full_text), all keyed by doc_id,
        in first-seen order."""
        if not self._fitted:
            raise ValueError("Index not built - call build_index() first.")
        doc_ids: List[str] = []
        seen = set()
        for c in self.chunks:
            if c.doc_id not in seen:
                seen.add(c.doc_id)
                doc_ids.append(c.doc_id)

        vectors, metadata, texts = {}, {}, {}
        for doc_id in doc_ids:
            idxs = [i for i, c in enumerate(self.chunks) if c.doc_id == doc_id]
            mean_vec = self.vectors[idxs].mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            vectors[doc_id] = mean_vec / norm if norm > 0 else mean_vec
            metadata[doc_id] = self.chunks[idxs[0]].metadata
            texts[doc_id] = " ".join(self.chunks[i].text for i in idxs)
        return vectors, metadata, texts

    def record_view(self, doc_id: str):
        """Log that you looked at this posting - like a watch-history entry.
        Purely implicit: no rating or like needed, matching how YouTube's
        'because you watched' rail works off view history alone."""
        self.view_history.append(doc_id)

    # -- CV upload / matching -------------------------------------------

    def load_cv(self, cv_text: str) -> np.ndarray:
        """Embed your CV/resume text into the same vector space as the
        postings, so it can anchor recommendations - either on its own via
        cv_match_report() (useful before you've viewed anything: a cold
        start that doesn't depend on view history), or blended into
        recommend_next() via its cv_weight argument."""
        if not self._fitted:
            raise ValueError("Index not built - call build_index() first.")
        dense = self.backend.transform([cv_text])
        norm = np.linalg.norm(dense, axis=1, keepdims=True)
        norm[norm == 0] = 1e-9
        self.cv_vector = (dense / norm)[0]
        self.cv_text = cv_text
        return self.cv_vector

    def load_cv_from_file(self, path: str) -> np.ndarray:
        """Convenience wrapper: extract_text_from_file() + load_cv() in one
        call. Accepts .txt, .pdf, or .docx."""
        return self.load_cv(extract_text_from_file(path))

    def cv_match_report(
        self, top_k: Optional[int] = None, sponsorship_boost_weight: float = 0.15
    ) -> pd.DataFrame:
        """Rank every indexed posting by similarity to your loaded CV,
        nudged by sponsorship signal - your cold-start view before you've
        looked at (and therefore recommend_next()-able from) anything."""
        if self.cv_vector is None:
            raise ValueError("No CV loaded - call load_cv() or load_cv_from_file() first.")
        doc_vectors, doc_meta, doc_texts = self._doc_level_vectors()
        rows = []
        for doc_id, vec in doc_vectors.items():
            similarity = float(np.dot(self.cv_vector, vec))
            sig = sponsorship_signal(doc_texts[doc_id])
            boost = SPONSORSHIP_BOOST.get(sig["label"], 0.0) * sponsorship_boost_weight
            meta = doc_meta[doc_id]
            rows.append(
                {
                    "doc_id": doc_id,
                    "company": meta.get("company"),
                    "country": meta.get("country"),
                    "role_title": meta.get("role_title"),
                    "cv_similarity": round(similarity, 4),
                    "sponsorship_signal": sig["label"],
                    "score": round(similarity + boost, 4),
                }
            )
        df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        return df.head(top_k) if top_k else df

    def recommend_next(
        self,
        viewed_doc_ids: Optional[List[str]] = None,
        top_k: int = 5,
        recency_half_life: float = 3.0,
        sponsorship_boost_weight: float = 0.15,
        cv_weight: float = 0.0,
    ) -> pd.DataFrame:
        """Content-based 'you looked at X, here's what's next' recommender.

        Builds a single taste vector from the postings you've viewed
        (recency-weighted - a view three positions back counts half as much
        as your most recent one, same shape as a watch-history decay), then
        ranks every un-viewed posting by cosine similarity to that vector.
        Similarity is nudged by SPONSORSHIP_BOOST so two similarly-relevant
        postings don't rank equally if one clearly sponsors and one clearly
        doesn't. Each row also names the viewed posting it most resembles,
        for a "because you looked at ..." explanation.

        With a single user and a small corpus this is necessarily
        content-based (similarity to what you've looked at), not
        collaborative filtering (similarity to what people *like you*
        looked at) - Netflix/YouTube blend both because they have millions
        of users; you'd only get collaborative signal here by pooling
        multiple people's view histories.

        cv_weight (0-1) blends your loaded CV (see load_cv/load_cv_from_file)
        into the taste vector - e.g. cv_weight=0.3 means the taste vector is
        70% "what you've viewed" and 30% "what your CV says". With no view
        history at all, set cv_weight=1.0 to recommend from the CV alone
        (equivalent to cv_match_report(), included here so a single method
        works whether or not you have view history yet).
        """
        viewed = list(viewed_doc_ids if viewed_doc_ids is not None else self.view_history)
        doc_vectors, doc_meta, doc_texts = self._doc_level_vectors()
        viewed = [d for d in viewed if d in doc_vectors]
        if not viewed and cv_weight < 1.0:
            raise ValueError(
                "No usable view history - call record_view(doc_id) first, "
                "pass viewed_doc_ids explicitly, or set cv_weight=1.0 to "
                "recommend from your loaded CV alone."
            )
        if cv_weight > 0 and self.cv_vector is None:
            raise ValueError("cv_weight > 0 but no CV loaded - call load_cv() or load_cv_from_file() first.")

        if viewed:
            n = len(viewed)
            weights = np.array([0.5 ** ((n - 1 - i) / recency_half_life) for i in range(n)])
            weights = weights / weights.sum()
            view_vector = np.sum([w * doc_vectors[d] for w, d in zip(weights, viewed)], axis=0)
            view_norm = np.linalg.norm(view_vector)
            if view_norm > 0:
                view_vector = view_vector / view_norm
            if cv_weight > 0 and self.cv_vector is not None:
                taste_vector = (1 - cv_weight) * view_vector + cv_weight * self.cv_vector
            else:
                taste_vector = view_vector
        else:
            taste_vector = self.cv_vector

        norm = np.linalg.norm(taste_vector)
        if norm > 0:
            taste_vector = taste_vector / norm

        viewed_set = set(viewed)
        rows = []
        for doc_id, vec in doc_vectors.items():
            if doc_id in viewed_set:
                continue
            similarity = float(np.dot(taste_vector, vec))
            sig = sponsorship_signal(doc_texts[doc_id])
            boost = SPONSORSHIP_BOOST.get(sig["label"], 0.0) * sponsorship_boost_weight
            if viewed:
                nearest_viewed = max(viewed, key=lambda v: float(np.dot(doc_vectors[v], vec)))
                because = doc_meta[nearest_viewed].get("company")
            else:
                because = "your CV"
            meta = doc_meta[doc_id]
            rows.append(
                {
                    "doc_id": doc_id,
                    "company": meta.get("company"),
                    "country": meta.get("country"),
                    "role_title": meta.get("role_title"),
                    "similarity": round(similarity, 4),
                    "sponsorship_signal": sig["label"],
                    "score": round(similarity + boost, 4),
                    "because_you_viewed": because,
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=["doc_id", "company", "country", "role_title", "similarity",
                         "sponsorship_signal", "score", "because_you_viewed"]
            )
        return (
            pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# New-posting watcher ("notifications")
# ---------------------------------------------------------------------------
#
# This environment can't send a real push/email notification - no internet
# access to a mail API, and nothing is scraping job boards for you yet. What
# it CAN do, and what's actually implemented below, is the part that matters
# once you do wire up a real channel: persist which postings you've already
# seen, diff each new batch against that, and score+format only what's new
# into an alert-ready digest. Rerun this after adding rows to your CSV (or
# after any other ingestion step you build) and it tells you what's new.
#
# To get an actual notification from this: put NewPostingWatcher.check()
# behind a scheduled job (cron, GitHub Actions on a schedule, or an AWS
# Lambda on an EventBridge timer - see the project roadmap) that (1) fetches
# fresh postings from a real source, (2) calls check(), and (3) if the
# result isn't empty, sends notify_digest()'s output through an email API
# (e.g. AWS SES) or a Slack/Discord webhook.


class NewPostingWatcher:
    """Tracks which postings (by doc_id) you've already seen across runs of
    this notebook/script, and reports only what's new - the diff step behind
    a job-alert notification."""

    def __init__(self, state_path: str = "seen_postings.json"):
        self.state_path = state_path
        self.seen_ids = self._load_state()

    def _load_state(self) -> set:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return set(json.load(f).get("seen_ids", []))
        return set()

    def _save_state(self):
        with open(self.state_path, "w") as f:
            json.dump({"seen_ids": sorted(self.seen_ids)}, f, indent=2)

    def _reference_vector(self, rag: "SponsorshipRAG", doc_vectors: Dict, cv_weight: float):
        """Same blending rule as SponsorshipRAG.recommend_next(), reused so
        'new posting' scores are directly comparable to recommend_next()'s
        scores. Returns None if there's no CV and no view history yet
        (nothing to personalize against)."""
        has_view = bool(rag.view_history)
        has_cv = rag.cv_vector is not None
        if not has_view and not has_cv:
            return None

        view_vec = None
        if has_view:
            viewed = [d for d in rag.view_history if d in doc_vectors]
            if viewed:
                n = len(viewed)
                weights = np.array([0.5 ** ((n - 1 - i) / 3.0) for i in range(n)])
                weights = weights / weights.sum()
                view_vec = np.sum([w * doc_vectors[d] for w, d in zip(weights, viewed)], axis=0)
                vn = np.linalg.norm(view_vec)
                view_vec = view_vec / vn if vn > 0 else view_vec

        if has_cv and view_vec is not None:
            ref_vec = (1 - cv_weight) * view_vec + cv_weight * rag.cv_vector
        elif has_cv:
            ref_vec = rag.cv_vector
        else:
            ref_vec = view_vec

        if ref_vec is None:
            return None
        rn = np.linalg.norm(ref_vec)
        return ref_vec / rn if rn > 0 else ref_vec

    def check(
        self,
        rag: "SponsorshipRAG",
        docs: List[Dict],
        mark_seen: bool = True,
        cv_weight: float = 0.5,
    ) -> pd.DataFrame:
        """docs: the full current corpus, as passed to add_documents(). Only
        postings whose id wasn't seen on a previous check() are scored and
        returned (best match first, if a CV or view history is available to
        score against - otherwise just listed with sponsorship signal)."""
        new_docs = [d for d in docs if d["id"] not in self.seen_ids]
        columns = ["doc_id", "company", "country", "role_title", "match_score", "sponsorship_signal"]
        if not new_docs:
            return pd.DataFrame(columns=columns)

        doc_vectors, _doc_meta, doc_texts = rag._doc_level_vectors()
        ref_vec = self._reference_vector(rag, doc_vectors, cv_weight)

        rows = []
        for d in new_docs:
            doc_id = d["id"]
            if doc_id not in doc_texts:
                continue  # not indexed yet - re-run build_index() to include it
            sig = sponsorship_signal(doc_texts[doc_id])
            match_score = float(np.dot(ref_vec, doc_vectors[doc_id])) if ref_vec is not None else None
            rows.append(
                {
                    "doc_id": doc_id,
                    "company": d.get("company"),
                    "country": d.get("country"),
                    "role_title": d.get("role_title"),
                    "match_score": round(match_score, 4) if match_score is not None else None,
                    "sponsorship_signal": sig["label"],
                }
            )

        df = pd.DataFrame(rows, columns=columns)
        if not df.empty and df["match_score"].notna().all():
            df = df.sort_values("match_score", ascending=False)
        df = df.reset_index(drop=True)

        if mark_seen:
            self.seen_ids.update(d["id"] for d in new_docs)
            self._save_state()
        return df

    def notify_digest(self, rag: "SponsorshipRAG", docs: List[Dict], **kwargs) -> str:
        """Same result as check(), formatted as a human-readable alert - the
        string you'd hand to an email/Slack sender once one is wired up."""
        df = self.check(rag, docs, **kwargs)
        if df.empty:
            return "No new postings since your last check."
        lines = [f"{len(df)} new posting(s) since your last check:", ""]
        for _, row in df.iterrows():
            score_str = f"match {row['match_score']:.2f} - " if pd.notna(row["match_score"]) else ""
            lines.append(
                f"- {row['company']} ({row['country']}, {row['role_title']}) - "
                f"{score_str}{row['sponsorship_signal']}"
            )
        return "\n".join(lines)
