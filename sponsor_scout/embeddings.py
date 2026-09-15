"""Embedding backends for SponsorshipRAG.

Two implementations of the same small interface:

    TfidfSvdBackend            TF-IDF + TruncatedSVD (classic LSA). Needs
                                nothing beyond scikit-learn, which is
                                already a core dependency, and no model
                                download — it is the backend that is
                                guaranteed to work with zero network access.

    SentenceTransformerBackend Real neural sentence embeddings via the
                                `sentence-transformers` package. Not a core
                                dependency (`pip install sponsor-scout[embeddings]`)
                                and its model weights are fetched from the
                                Hugging Face Hub on first use, then cached.

`resolve_backend()` is what `SponsorshipRAG` calls: pass "auto" (the
default) to prefer neural embeddings and fall back to TF-IDF automatically
when sentence-transformers isn't installed or its model can't be fetched,
or force one explicitly with "tfidf" / "sentence-transformers".

Both backends return plain (not unit-normalized) dense vectors —
SponsorshipRAG normalizes uniformly after embedding, so cosine similarity
behaves the same regardless of which backend produced the vectors, and nothing
in pipeline.py needs to know which one is active.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

DEFAULT_SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"


class EmbeddingBackend:
    """Interface: fit_transform() embeds the corpus once, in build_index();
    transform() embeds new text (a query or a CV) into that same space
    afterward."""

    name = "base"

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def transform(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfSvdBackend(EmbeddingBackend):
    """TF-IDF + TruncatedSVD. See the module docstring above — this is the
    always-available fallback."""

    name = "tfidf"

    def __init__(self, n_components: int = 100):
        self._requested_components = n_components
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        # max_df=0.95 drops terms appearing in more than 95% of chunks. On a
        # corpus of one or two documents that ratio rounds below min_df=1
        # and sklearn raises rather than returning an empty vocabulary. A
        # tiny corpus is a real state (a freshly-seeded store, a
        # single-company filter), so relax the ratio there instead of
        # letting the caller hit a 500.
        max_df = 1.0 if len(texts) < 5 else 0.95
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=max_df, min_df=1)
        tfidf = self.vectorizer.fit_transform(texts)
        max_components = max(1, min(tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        n_components = max(1, min(self._requested_components, max_components))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        return self.svd.fit_transform(tfidf)

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.vectorizer is None or self.svd is None:
            raise ValueError("fit_transform() must be called before transform().")
        return self.svd.transform(self.vectorizer.transform(texts))


class SentenceTransformerBackend(EmbeddingBackend):
    """Real neural sentence embeddings. Requires the `embeddings` extra:

        pip install sponsor-scout[embeddings]

    The import and model load happen in __init__ (not at module import
    time) so that importing sponsor_scout.embeddings never requires the
    package to be installed — only actually constructing this class does.
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Install it with:\n"
                "    pip install sponsor-scout[embeddings]\n"
                "or use embedding_backend='tfidf' to run without it."
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        # A pretrained sentence encoder has nothing to fit per-corpus —
        # embedding is the same operation whether it's the first call
        # (the corpus) or a later one (a query/CV).
        return self.transform(texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), show_progress_bar=False))


def resolve_backend(name: str, n_components: int = 100) -> EmbeddingBackend:
    """name: 'auto' (default) | 'tfidf' | 'sentence-transformers'.

    'auto' prefers real neural embeddings and falls back to TF-IDF+SVD when
    sentence-transformers isn't installed, or when constructing it fails for
    any other reason (e.g. no network to fetch the model on first use,
    since that's the one failure mode a purely offline environment — the
    condition this project originally shipped under — will always hit).
    """
    if name == "tfidf":
        return TfidfSvdBackend(n_components=n_components)
    if name == "sentence-transformers":
        return SentenceTransformerBackend()
    if name != "auto":
        raise ValueError(
            f"Unknown embedding_backend: {name!r} (use 'auto', 'tfidf', or 'sentence-transformers')"
        )

    try:
        return SentenceTransformerBackend()
    except ImportError:
        logger.info(
            "sentence-transformers not installed - using TF-IDF+SVD. "
            "Install the 'embeddings' extra for neural embeddings."
        )
        return TfidfSvdBackend(n_components=n_components)
    except Exception as exc:  # noqa: BLE001 - model load can fail many ways (no network, disk, corrupt cache)
        logger.warning("sentence-transformers failed to load (%s) - falling back to TF-IDF+SVD.", exc)
        return TfidfSvdBackend(n_components=n_components)
