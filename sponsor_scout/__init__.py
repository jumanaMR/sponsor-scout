"""Sponsor Scout — retrieval over job postings, scored for visa-sponsorship signal."""

from sponsor_scout.pipeline import (
    NewPostingWatcher,
    SponsorshipRAG,
    chunk_text,
    extract_text_from_file,
    sponsorship_signal,
)

__all__ = [
    "NewPostingWatcher",
    "SponsorshipRAG",
    "chunk_text",
    "extract_text_from_file",
    "sponsorship_signal",
]

__version__ = "0.1.0"
