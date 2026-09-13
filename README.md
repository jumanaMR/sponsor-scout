# Sponsor Scout

Retrieval over job postings, scored for visa-sponsorship signal — search, recommendations, and CV matching over a corpus you build yourself.

Built to solve a specific problem: as an offshore applicant, most of the job postings you can find are ones you aren't eligible for, and the sentence that tells you which is which is buried somewhere in the middle of the description. This chunks postings, embeds them into a vector index, and surfaces both *relevance* and *whether the company appears to sponsor* — separately, because those are different questions.

```
┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────────┐
│ postings │──▶│ chunking │──▶│  embeddings  │──▶│  vector   │──▶│  retrieval   │
│ (CSV/UI) │   │ +overlap │   │ TF-IDF + SVD │   │   index   │   │  + reranking │
└──────────┘   └──────────┘   └──────────────┘   └───────────┘   └──────┬───────┘
                                                                        │
        ┌───────────────────────────┬───────────────────────────┬───────┴────────┐
        ▼                           ▼                           ▼                ▼
  semantic search           recommendations              CV matching      new-posting
  "482 sponsorship            "because you              cold-start          watcher
   in Australia"             looked at X"               ranking          (alert diff)
                                    │                        │                │
                                    └────────────┬───────────┘                │
                                                 ▼                            │
                                    sponsorship signal (heuristic) ───────────┘
                                    boosts / penalises every ranking
```

## What it does

**Semantic search** retrieves the closest *chunk* of any posting rather than whole documents, so a long posting surfaces on the paragraph that actually matches your query.

**Sponsorship signal** labels each posting `likely sponsoring` / `likely NOT sponsoring` / `unclear` / `no sponsorship language`, with negation handling and the matched phrases shown as evidence. It is a filter for what to read closely, not a classifier you should trust blindly.

**Recommendations** build a recency-weighted taste vector from the postings you've opened — a recent view counts more than an older one — then rank everything you haven't seen, nudged by sponsorship signal so a clear "no" can't outrank a clear "yes" on content similarity alone. Each recommendation names the posting it came from ("because you looked at …").

**CV matching** embeds your CV into the same vector space, so it can rank postings directly. That's the cold start: it works before you've opened anything, and blends with view history as you do (`cv_weight`).

**New-posting watcher** persists which postings you've already seen and reports only what's new, scored the same way. This is the diff step behind a job alert — the piece a scheduled job would call before emailing you a digest.

## Quickstart

```bash
git clone https://github.com/jumanaMR/sponsor-scout.git
cd sponsor-scout
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Or work through the notebook, which runs the whole pipeline end to end with commentary:

```bash
jupyter notebook notebooks/sponsorship_rag.ipynb
```

Or use it as a library:

```python
from sponsor_scout import SponsorshipRAG
from sponsor_scout.sample_data import SAMPLE_POSTINGS

rag = SponsorshipRAG(n_components=50)
rag.add_documents(SAMPLE_POSTINGS)
rag.build_index()

rag.load_cv_from_file("my_cv.pdf")          # .pdf / .docx / .txt
print(rag.cv_match_report(top_k=10))

rag.record_view("syn-001")                   # you opened this posting
print(rag.recommend_next(cv_weight=0.4))     # 60% view history, 40% CV
```

### Adding real postings

`data/sponsor_leads_template.csv` has the columns the pipeline expects — `company, role_title, country, source, date_posted, text`. Paste in the full text of real postings; the more of the original wording you keep, the better both retrieval and the sponsorship heuristic work.

Your filled-in CSVs and your CV are gitignored. Don't commit them — they're personal data, and a public repo is a strange place for your job search to live.

## Repo layout

```
sponsor_scout/       the pipeline — chunking, embedding, index, heuristic, recommender
app.py               Streamlit UI over the pipeline
webapp/index.html    standalone browser version (JS port, no Python needed)
notebooks/           the pipeline end to end, with commentary
tests/               pytest suite, including two regression tests (see below)
data/                CSV template for your own postings
scripts/             notebook build + a Streamlit smoke test that needs no Streamlit
```

## Design decisions

**TF-IDF + SVD, not a neural embedding model.** The first version was built in an environment with no outbound network access, so `sentence-transformers` wasn't installable. TF-IDF + `TruncatedSVD` is a real dense-vector technique (this is classic LSA) and kept the *architecture* honest — chunking, an index, cosine retrieval, reranking — with a documented three-line upgrade path. It's a constraint handled openly rather than a shortcut presented as a choice. The upgrade is in the roadmap below.

**Content-based, not collaborative filtering.** The recommender ranks by similarity to what *you* looked at. Netflix and YouTube blend that with collaborative filtering — similarity to what people *like you* looked at — which needs many users' histories. With a single user there's no collaborative signal to draw on, so this implements the content-based half and says so, rather than claiming to be something it isn't.

**The heuristic is a filter, not a classifier.** Keyword matching with negation handling gets you a useful triage signal cheaply. It does not get you a reliable yes/no, and the UI never presents it as one — every label ships with the phrases that produced it so you can disagree with it at a glance.

**One pipeline, three front ends.** The notebook, the Streamlit app, and a future API all call the same `sponsor_scout.pipeline` — no scoring logic is duplicated in a UI layer. (The browser version under `webapp/` is the deliberate exception: it's a JS port so the page can run with no backend, and its output is checked against the Python.)

## Two bugs worth reading the tests for

Both were found by reading output, not by the code looking wrong, and both are the same failure mode — naive substring matching against a phrase list. They're pinned as named regression tests in `tests/test_sponsorship_signal.py`:

1. **Negation.** `"no mention of visa sponsorship"` contains the positive phrase `"visa sponsorship"`, so a posting that explicitly says nothing about sponsorship was labelled `likely sponsoring`. Fixed with a negation-window check on the words preceding a match.

2. **Word boundaries.** `"unable to sponsor"` contains the positive phrase `"able to sponsor"`, so a company explicitly refusing to sponsor came back as `mixed signal`. Fixed by requiring a match to begin at a word start — while deliberately leaving the *end* open, so `"approved sponsors"` still matches `"approved sponsor"`.

The second one is why the first fix wasn't enough on its own: the same class of bug had two different surfaces, and only output review caught either.

## Limitations

- **The sample postings are synthetic.** The twelve entries in `sponsor_scout/sample_data.py` are invented companies written to exercise the pipeline. They are not real leads and must not be read as claims about any real company's sponsorship policy.
- **The heuristic will be wrong sometimes.** It reads language, not policy. A company can advertise sponsorship and decline you; a posting can omit it and the company sponsor happily.
- **Nothing ingests postings automatically yet.** You add them by hand or by CSV. See the roadmap.
- **Retrieval scores don't understand your query's polarity.** Searching "companies that won't sponsor" returns postings *about* sponsorship, ranked by wording similarity. The label column is what answers that question.

## Roadmap

| Stage | Status | |
|---|---|---|
| 0 | done | Prototype: chunking, retrieval, recommender, CV matching, watcher |
| 1 | done | Package layout, pytest suite, CI |
| 2 | next | pgvector on Postgres + a FastAPI layer over the existing methods |
| 3 | | Real ingestion from a compliant source (Adzuna API, or parsing your own job-alert emails) |
| 4 | | Deploy: Terraform for RDS / Lambda / API Gateway / EventBridge / SES, so the watcher emails a digest on a schedule |
| 5 | | Swap TF-IDF+SVD for a fixed neural embedding model |

On ingestion: scraping LinkedIn, SEEK or Indeed directly is against their terms of service, and getting an account flagged while you're actively applying through it is a bad trade. Official APIs and your own inbox are the routes worth building on.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

CI runs the suite on Python 3.11 and 3.12 on every push and pull request.

`scripts/smoke_test_app.py` executes `app.py` against a stub Streamlit module — it catches mismatches between the UI and the pipeline API without needing Streamlit installed. It's how the "collapsed expanders silently record a view for every posting" bug got caught before it shipped.

## License

MIT — see [LICENSE](LICENSE).
