"""Smoke-test app.py without Streamlit installed.

Streamlit can't be pip-installed in this sandbox (no PyPI access), so this
injects a stub `streamlit` module that records calls and returns plausible
widget values, then executes app.py top to bottom. It won't catch layout
problems, but it does catch the thing that actually breaks a Streamlit app
on first run: a mismatch between what app.py calls and what rag_pipeline
actually exposes.
"""

import sys
import types
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CALLS = []


class SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc

    def __setattr__(self, k, v):
        self[k] = v


class Widget:
    """Stands in for st, st.sidebar, a column, a tab — all of which expose
    the same widget API in Streamlit."""

    def __init__(self, name="st"):
        self._name = name

    # -- containers ------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [Widget(f"{self._name}.col{i}") for i in range(n)]

    def tabs(self, labels):
        return [Widget(f"{self._name}.tab[{label}]") for label in labels]

    @contextmanager
    def expander(self, label, **kw):
        CALLS.append(("expander", label))
        yield Widget("expander")

    @contextmanager
    def form(self, key, **kw):
        CALLS.append(("form", key))
        yield Widget("form")

    def container(self, **kw):
        return Widget("container")

    # -- output ----------------------------------------------------------
    def _noop(self, name):
        def fn(*a, **kw):
            CALLS.append((name, a[:1]))
        return fn

    def __getattr__(self, name):
        return self._noop(name)

    # -- widgets with return values --------------------------------------
    def button(self, *a, **kw):
        return False

    def form_submit_button(self, *a, **kw):
        return False

    def file_uploader(self, *a, **kw):
        return None

    def text_input(self, label, value="", **kw):
        return value

    def text_area(self, label, value="", **kw):
        return value

    def selectbox(self, label, options, **kw):
        return list(options)[0]

    def slider(self, label, min_value=None, max_value=None, value=None, *a, **kw):
        return value if value is not None else min_value

    def checkbox(self, *a, **kw):
        return False

    def metric(self, label, value, **kw):
        CALLS.append(("metric", (label, value)))

    def dataframe(self, df, **kw):
        CALLS.append(("dataframe", (len(df),)))

    def rerun(self):
        CALLS.append(("rerun", ()))

    def set_page_config(self, **kw):
        CALLS.append(("set_page_config", ()))

    def markdown(self, body, **kw):
        CALLS.append(("markdown", (str(body)[:60],)))

    def get(self, *a, **kw):  # st.session_state.get proxied in app code
        raise AttributeError


stub = Widget("st")
stub.session_state = SessionState()
stub.sidebar = Widget("st.sidebar")
sys.modules["streamlit"] = stub

# ---------------------------------------------------------------------------

print("Executing app.py against stub Streamlit...\n")
source = open(REPO_ROOT / "app.py").read()
exec(compile(source, "app.py", "exec"), {"__name__": "__main__"})

print("app.py executed with no exceptions.")
print(f"Recorded {len(CALLS)} Streamlit calls.")

metrics = [c for c in CALLS if c[0] == "metric"]
print("\nHeader metrics rendered:")
for _, (label, value) in metrics:
    print(f"  {label}: {value}")

kinds = {}
for name, _ in CALLS:
    kinds[name] = kinds.get(name, 0) + 1
print("\nCall mix:", dict(sorted(kinds.items(), key=lambda kv: -kv[1])))
