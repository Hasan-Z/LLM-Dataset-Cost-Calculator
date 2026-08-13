"""
Shared fixtures for LLM Cost Calculator tests.
Run with:  pytest tests/ -v
"""
import io
import json
import os
import sys
import tempfile
import shutil
import pytest
import pandas as pd

# ── Make sure main.py is importable ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Minimal pricing data used by tests ──────────────────────────────────────
MOCK_PRICING = {
    "gpt-4o": {
        "input":         5.0,
        "output":        15.0,
        "cached_input":  2.5,
        "pinned":        False,
        "litellm_tracked": True,
        "mode":          "chat",
    },
    "gpt-4o-mini": {
        "input":         0.15,
        "output":        0.6,
        "cached_input":  0.075,
        "pinned":        False,
        "litellm_tracked": True,
        "mode":          "chat",
    },
    "gpt-4": {
        "input":         30.0,
        "output":        60.0,
        "cached_input":  None,
        "pinned":        False,
        "litellm_tracked": True,
        "mode":          "chat",
    },
}

MOCK_ANTHROPIC_PRICING = {
    "claude-3-5-sonnet-20241022": {
        "input":         3.0,
        "output":        15.0,
        "cached_input":  0.3,
        "pinned":        False,
        "litellm_tracked": False,
        "mode":          "chat",
    },
}

MOCK_GEMINI_PRICING = {
    "gemini-1.5-pro": {
        "input":         1.25,
        "output":        5.0,
        "cached_input":  0.3125,
        "pinned":        False,
        "litellm_tracked": False,
        "mode":          "chat",
    },
}


def make_excel_bytes(data: dict) -> bytes:
    """Create an in-memory Excel file from a dict of {col: [values]}."""
    df  = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf.read()


@pytest.fixture(scope="session")
def tmp_project(tmp_path_factory):
    """
    Create a temporary project directory with:
      - main.py (symlinked / copied)
      - templates/ (symlinked / copied)
      - mock JSON pricing files
    Returns the path to the temp dir.
    """
    src = os.path.dirname(os.path.dirname(__file__))
    tmp = tmp_path_factory.mktemp("project")

    # Copy / link main.py and templates
    shutil.copy(os.path.join(src, "main.py"), tmp / "main.py")
    templates_src = os.path.join(src, "templates")
    if os.path.isdir(templates_src):
        shutil.copytree(templates_src, tmp / "templates")
    else:
        (tmp / "templates").mkdir()

    # Write mock pricing JSON files
    full_openai = {**MOCK_PRICING,
                   "last_updated": "2025-01-01 00:00 UTC",
                   "last_updated_source": "manual"}
    full_anthropic = {**MOCK_ANTHROPIC_PRICING,
                      "last_updated": "2025-01-01 00:00 UTC",
                      "last_updated_source": "manual"}
    full_gemini = {**MOCK_GEMINI_PRICING,
                   "last_updated": "2025-01-01 00:00 UTC",
                   "last_updated_source": "manual"}

    (tmp / "openai_ai_models_pricing.json").write_text(json.dumps(full_openai, indent=2))
    (tmp / "anthropic_ai_models_pricing.json").write_text(json.dumps(full_anthropic, indent=2))
    (tmp / "google_ai_models_pricing.json").write_text(json.dumps(full_gemini, indent=2))

    return tmp


@pytest.fixture(scope="session")
def client(tmp_project):
    """FastAPI TestClient pointed at a clean temp project directory."""
    orig_dir = os.getcwd()
    os.chdir(tmp_project)

    # Re-import app fresh from the temp dir
    import importlib
    if "main" in sys.modules:
        del sys.modules["main"]
    import main as app_module
    importlib.reload(app_module)

    from fastapi.testclient import TestClient
    tc = TestClient(app_module.app)

    yield tc

    os.chdir(orig_dir)


@pytest.fixture
def sample_excel() -> bytes:
    """A simple 3-row Excel with columns: prompt, response, context."""
    return make_excel_bytes({
        "prompt":   ["Hello world", "Tell me a joke", "What is AI?"],
        "response": ["Hi there",    "Why did the chicken...", "AI stands for..."],
        "context":  ["general",     "humor",                  "technology"],
    })


@pytest.fixture
def pricing_override() -> str:
    """JSON string of mock pricing suitable for pricing_override form field."""
    return json.dumps(MOCK_PRICING)
