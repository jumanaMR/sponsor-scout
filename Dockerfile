# Sponsor Scout — one image, three entry points (API, Streamlit app, agent).
#
# Built from the declared extras in pyproject.toml rather than a separate
# requirements file, so the container and a local `pip install .[app,api]`
# can't drift apart.
#
# Note on reproducibility: an image is only as reproducible as what it
# installs. pyproject pins lower bounds (>=), so rebuilding this six months
# from now can resolve to different versions of pandas and scikit-learn.
# If you need byte-identical environments across machines, generate a lock
# on a machine where everything works —
#
#     pip freeze > requirements.lock.txt
#
# — and swap the install line below for `pip install -r requirements.lock.txt`.
# That, not the container, is what actually makes it repeatable.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SPONSOR_SCOUT_DB=/app/data/postings.db

WORKDIR /app

# Dependency layer first: these change far less often than application code,
# so a code edit doesn't trigger a full reinstall of scikit-learn.
COPY pyproject.toml README.md LICENSE ./
COPY sponsor_scout ./sponsor_scout
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[app,api]"

COPY app.py ./
COPY data/sponsor_leads_template.csv ./data/

# Don't run as root. The store lives in /app/data, which is a mounted volume
# in compose, so it has to be writable by this user.
RUN useradd --create-home --shell /bin/bash scout \
    && mkdir -p /app/data \
    && chown -R scout:scout /app
USER scout

EXPOSE 8000 8501

# python rather than curl: the slim image has no curl, and adding one just
# for a healthcheck is 10MB for nothing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "sponsor_scout.api:app", "--host", "0.0.0.0", "--port", "8000"]
