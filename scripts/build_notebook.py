"""Hand-builds sponsorship_rag.ipynb (no nbformat/jupyter available in this
environment — pypi.org is blocked by network policy here — so this writes
the notebook JSON directly and executes each code cell for real to capture
genuine outputs, exactly like re-running the notebook would produce)."""

import contextlib
import io
import json

CELLS = []
NAMESPACE = {}


def md(text: str):
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def code(source: str):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(source, "<cell>", "exec"), NAMESPACE)
    output_text = buf.getvalue()
    outputs = []
    if output_text:
        outputs.append({
            "output_type": "stream",
            "name": "stdout",
            "text": output_text.splitlines(keepends=True),
        })
    CELLS.append({
        "cell_type": "code",
        "execution_count": len([c for c in CELLS if c["cell_type"] == "code"]) + 1,
        "metadata": {},
        "outputs": outputs,
        "source": source.splitlines(keepends=True),
    })


# ---------------------------------------------------------------------------

md("""# Visa-Sponsorship RAG Finder — prototype

A small retrieval-augmented pipeline for job hunting: paste in job postings,
it chunks + embeds them into a vector index, and lets you query for
visa-sponsorship signal across companies and countries — built for
targeting internationally-sponsored roles (482 / Skilled Worker / Blue Card /
LMIA / Global Talent Stream, etc.) as an offshore applicant.

**Two honesty notes before you use this:**

1. The example postings in this notebook (`sample_data.py`) are **synthetic
   — made-up companies, written only to exercise the pipeline.** They are
   not real leads. Replace them with real postings you collect yourself
   (see Section 6) before trusting any output.
2. This environment has no internet access to install packages, so the
   "vector" embeddings here are TF-IDF + SVD (scikit-learn, already
   installed) rather than a neural embedding model. It's a legitimate
   dense-vector technique (this is classic LSA) and the architecture —
   chunk → embed → index → cosine-similarity retrieve → synthesize — is the
   same shape you'd use with `sentence-transformers` + `FAISS`. Section 8
   shows exactly what to swap once you have internet access (e.g. running
   this on your own laptop) to upgrade to neural embeddings.
""")

md("## 1. Setup")

code("""\
import sys
sys.path.insert(0, "..")   # so the notebook works without installing the package

from sponsor_scout.sample_data import SAMPLE_POSTINGS
from sponsor_scout.pipeline import SponsorshipRAG, chunk_text, sponsorship_signal
import pandas as pd

pd.set_option("display.max_colwidth", 60)
print(f"Loaded {len(SAMPLE_POSTINGS)} synthetic sample postings.")
""")

md("""## 2. Chunking

Each posting gets split into overlapping word-chunks before embedding, so a
long posting doesn't get squashed into one vector and retrieval can surface
just the relevant paragraph (e.g. the sponsorship sentence) rather than the
whole document.""")

code("""\
example = SAMPLE_POSTINGS[0]
chunks = chunk_text(example["text"], max_words=25, overlap_words=8)
print(f"'{example['company']}' posting -> {len(chunks)} chunks (max_words=25 for this demo):\\n")
for i, c in enumerate(chunks):
    print(f"[{i}] {c}\\n")
""")

md("## 3. Build the vector index\n\nDefault chunk size (60 words, 15 overlap) — smaller than the demo above so each chunk still reads naturally.")

code("""\
rag = SponsorshipRAG(n_components=50)
rag.add_documents(SAMPLE_POSTINGS, chunk_size=60, overlap=15)
rag.build_index()
print(f"Indexed {len(rag.chunks)} chunks from {len(SAMPLE_POSTINGS)} postings.")
print(f"Vector matrix shape: {rag.vectors.shape}")
""")

md("""## 4. Query the index

`rag.answer()` retrieves the top-k most similar chunks, de-duplicates by
company, and attaches the sponsorship-signal heuristic to each — this is the
"RAG" part: retrieval feeding a synthesized (here, template-based, not
LLM-generated) answer.""")

code("""\
print(rag.answer("AI engineer roles with 482 visa sponsorship in Australia", top_k=6))
""")

code("""\
print(rag.answer("machine learning roles in the Netherlands or Germany with relocation support", top_k=6))
""")

md("""**Reading the results:** the match *score* ranks by semantic similarity
to your query wording — it does **not** know whether that match is a "yes"
or "no" on sponsorship (notice the second query above surfaces a `likely
sponsoring` Delta Cloud Systems posting near the top just because it's
about relocation in the Netherlands). The `sponsorship_signal` label next to
each result is the part that actually tells you yes/no/unclear — always
read that, not just the ranking.""")

md("## 5. Company-level sponsorship report\n\nOne row per posting, independent of any query — good for a quick scan or exporting to CSV.")

code("""\
report = rag.company_report(SAMPLE_POSTINGS)
print(report.to_string(index=False))
""")

code("""\
report.to_csv("sponsorship_report.csv", index=False)
print("Saved sponsorship_report.csv")
""")

md("""## 6. Add your own real postings

`data/sponsor_leads_template.csv` has the columns this pipeline
expects: `company, role_title, country, source, date_posted, text`. Paste
the full text of real postings you find (LinkedIn, SEEK, Indeed, a
company's own careers page, etc.) into the `text` column — the more of the
original wording you keep, the better both retrieval and the
sponsorship-signal heuristic work.

This cell loads that CSV and merges it with the synthetic sample data so
you can see the combined pipeline run end-to-end. Once you've replaced the
template with real rows, drop the `SAMPLE_POSTINGS` half of the merge (or
just point `add_documents` at `real_docs` alone).""")

code("""\
real_df = pd.read_csv("../data/sponsor_leads_template.csv")
real_docs = [
    {
        "id": f"real-{i:03d}",
        "company": row["company"],
        "country": row["country"],
        "role_title": row["role_title"],
        "source": row["source"],
        "date_posted": row["date_posted"],
        "text": row["text"],
    }
    for i, row in real_df.iterrows()
]
print(f"Loaded {len(real_docs)} row(s) from sponsor_leads_template.csv")

combined_rag = SponsorshipRAG(n_components=50)
combined_rag.add_documents(SAMPLE_POSTINGS + real_docs, chunk_size=60, overlap=15)
combined_rag.build_index()
print(f"Combined index: {len(combined_rag.chunks)} chunks from {len(SAMPLE_POSTINGS) + len(real_docs)} postings")
print()
print(combined_rag.company_report(SAMPLE_POSTINGS + real_docs).tail(len(real_docs) + 1).to_string(index=False))
""")

md("""## 7. Sponsorship-signal heuristic on its own

You can also run the heuristic directly on any pasted paragraph, without
going through the vector index at all — useful for a quick check while
you're reading a posting.""")

code("""\
sample_paragraph = (
    "We are an approved sponsor and can support a Temporary Skill Shortage "
    "(subclass 482) visa for offshore candidates relocating to Sydney."
)
result = sponsorship_signal(sample_paragraph)
print(f"Label: {result['label']}")
print(f"Positive matches: {result['positive_matches']}")
print(f"Negative matches: {result['negative_matches']}")
""")

md("""## 8. Recommendations — "because you looked at ..." (Netflix/YouTube-style)

The idea: every time you open/read a posting, log it with `record_view()`.
The recommender builds a single **taste vector** from the postings you've
viewed — weighted so a recent view counts more than one from a while back,
the same shape as a watch-history decay — then ranks every posting you
*haven't* viewed by similarity to that taste vector. Two similarly-relevant
postings don't rank equally, either: `sponsorship_signal` nudges the score
up for postings that look like they sponsor and down for ones that clearly
don't, since a purely "similar content" match isn't actually useful here if
the company won't sponsor.

One honest caveat: with one user and a dozen postings, this is
**content-based** filtering (similar to what *you* looked at) — real
Netflix/YouTube recommenders also lean on **collaborative** filtering
(similar to what people *like you* looked at), which needs many users'
histories to work. There's no collaborative signal to draw on with a single
person's job search, so this notebook only implements the content-based
half — still the core mechanic behind "because you watched X" rails.""")

code("""\
# Simulate a session: you open two Australia-based AI/automation postings.
rag.record_view("syn-001")  # Nimbus Data Labs
rag.record_view("syn-003")  # Harbor Robotics
print(f"View history: {rag.view_history}")
""")

code("""\
recommendations = rag.recommend_next(top_k=6)
print(recommendations.to_string(index=False))
""")

md("""Silverline Cognition comes out on top — another Australia-based AI/
automation role with a strong `likely sponsoring` signal, most similar to
the Nimbus Data Labs posting you viewed. Notice postings with a `likely NOT
sponsoring` signal (Aurora Analytics, Cliffside Softworks, Ferngully Data
Co) drop toward the bottom even when their content is topically similar —
that's the sponsorship boost/penalty at work, not just raw similarity.

Call `record_view()` again each time you open a new posting and re-run
`recommend_next()` — the taste vector updates automatically since it always
rebuilds from `rag.view_history` (or pass your own list of doc_ids via
`viewed_doc_ids=` if you're tracking views somewhere else, like a
spreadsheet).""")

md("""## 9. CV upload & matching

Load your CV/resume so postings can be ranked by similarity to it directly
— useful as a cold start before you've viewed anything (`cv_match_report()`),
or blended into the view-history recommender (`recommend_next(cv_weight=...)`).

`load_cv_from_file(path)` accepts `.txt`, `.pdf`, or `.docx` (this
environment happens to already have `pypdf` and `python-docx` installed —
if you're running this elsewhere and it's missing either, `pip install
pypdf python-docx`). The cell below uses a short placeholder CV instead of
a real upload, since no CV file was provided to this session — replace
`CV_TEXT` with your actual CV text, or point `load_cv_from_file()` at your
real file, e.g.:

```python
rag.load_cv_from_file("my_cv.pdf")   # or "my_cv.docx" / "my_cv.txt"
```""")

code("""\
# PLACEHOLDER — replace with your real CV text (or use load_cv_from_file()
# on an actual .txt/.pdf/.docx once you have one attached to this session).
CV_TEXT = '''
AI and Automation Engineer with experience building RAG pipelines, vector
search, chunking strategies, and LLM evaluation in Python. Background in
machine learning, computer vision, and workflow automation (n8n). Seeking
an AI/automation engineering role with visa sponsorship for an offshore
applicant, targeting Australia, the Netherlands, Germany, the UK, Ireland,
or Canada.
'''

rag.load_cv(CV_TEXT)
print("CV loaded and embedded into the same vector space as the postings.")
""")

code("""\
print(rag.cv_match_report(top_k=6).to_string(index=False))
""")

md("""And blended with view history — `cv_weight=0.4` means the ranking is
60% "what you've viewed" (from Section 8: Nimbus Data Labs, Harbor Robotics)
and 40% "what your CV says":""")

code("""\
print(rag.recommend_next(cv_weight=0.4, top_k=6).to_string(index=False))
""")

md("""## 10. New-posting notifications

`NewPostingWatcher` persists which postings you've already seen (to a small
JSON file) and, each time you re-run it against your current corpus, tells
you only what's new — scored against your CV/view taste and sponsorship
signal, same as the recommender. Run it again after adding rows to
`data/sponsor_leads_template.csv`.

**Honest limitation:** this session has no internet access to an email or
push API, and nothing here is scraping job boards for you automatically —
so this *is* the notification logic, not yet a notification you'd receive
on your phone. The project roadmap covers wiring `notify_digest()`'s output
into a real channel (AWS SES email, a Slack webhook) behind a scheduled
job, plus where the "new postings" would actually come from.""")

code("""\
from sponsor_scout.pipeline import NewPostingWatcher

watcher = NewPostingWatcher(state_path="seen_postings.json")
print(watcher.notify_digest(rag, SAMPLE_POSTINGS, cv_weight=0.4))
""")

code("""\
# Run again with no new data — everything's already marked seen.
print(watcher.notify_digest(rag, SAMPLE_POSTINGS, cv_weight=0.4))
""")

md("""Now simulate a new posting showing up (e.g. you added a row to the
CSV). Rebuilding the index refits the vector space on the larger corpus —
which means the CV and view-history vectors need to be re-embedded in that
new space too (`load_cv()` again, `record_view()` replayed); this is a real
limitation of a sparse embedding that gets refit on each rebuild, and part
of why Section 11 below suggests a fixed neural embedding model once you're
past the prototype stage — a fixed model doesn't need re-embedding old
state just because the corpus grew.""")

code("""\
new_posting = {
    "id": "syn-999",
    "company": "Pinehollow AI",
    "country": "Australia",
    "role_title": "AI Engineer",
    "source": "synthetic-example",
    "date_posted": "2026-09-13",
    "text": (
        "Pinehollow AI is an approved 482 sponsor hiring an offshore AI "
        "engineer for our Perth office, with relocation support and "
        "RAG / vector search experience preferred."
    ),
}
updated_corpus = SAMPLE_POSTINGS + [new_posting]

rag_updated = SponsorshipRAG(n_components=50)
rag_updated.add_documents(updated_corpus, chunk_size=60, overlap=15)
rag_updated.build_index()
for doc_id in rag.view_history:
    rag_updated.record_view(doc_id)
rag_updated.load_cv(CV_TEXT)

# Watcher state (seen_postings.json) persists from the runs above, so only
# the brand-new posting shows up here.
print(watcher.notify_digest(rag_updated, updated_corpus, cv_weight=0.4))
""")

md("""## 11. Upgrading to neural embeddings later

This notebook's environment has no internet access, so it uses TF-IDF+SVD.
On your own machine (`pip install sentence-transformers faiss-cpu`), swap
the embedding step in `sponsor_scout/pipeline.py` for:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts, normalize_embeddings=True)
```

and the vector index for a FAISS `IndexFlatIP`. The rest of the pipeline
(`add_documents` / `query` / `answer` / `sponsorship_signal`) keeps the same
interface either way — that's the part worth having in a portfolio: a
clean separation between chunking, embedding, indexing, and retrieval, so
the embedding backend is a swappable implementation detail rather than
baked into the rest of the code.

**Next steps:**
- Swap in real postings (Section 6) as you collect them from job boards.
- Optionally replace the template-based `answer()` synthesis with a real
  LLM call (Claude/OpenAI) over the retrieved chunks, once you have API
  access, for a more natural-language summary.
- If this becomes a recurring tool rather than a one-off notebook, it's a
  reasonable candidate for the "depth project" slot in your portfolio
  structure (a small FastAPI wrapper + a persisted vector store would take
  it from prototype to shippable).""")

# ---------------------------------------------------------------------------

notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("sponsorship_rag.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Wrote sponsorship_rag.ipynb")
