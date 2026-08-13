# LLM Cost Calculator — Test Suite

## Running tests

```bat
# Windows (from project root)
runme_tests.bat
```

```bash
# Mac / Linux (from project root)
python -m pytest tests/ -v
```

Run a single file:
```bash
python -m pytest tests/test_html_integrity.py -v
python -m pytest tests/test_logic.py -v
python -m pytest tests/test_api.py -v
```

Run a single class or test:
```bash
python -m pytest tests/test_api.py::TestCalculate -v
python -m pytest tests/test_api.py::TestCalculate::test_budget_exceeded_shows_warning -v
```

---

## Test files

### `test_logic.py` — Pure Python unit tests
No HTTP, no server. Tests run in milliseconds.

| Class | What it tests |
|-------|--------------|
| `TestCountTokens` | `count_tokens()` — empty, None, unknown model fallback, non-string coercion |
| `TestSafePrice` | `safe_price()` — None, per-token → $/1M conversion, invalid strings |
| `TestInferCapabilities` | GPT-4o, o1, TTS, Claude Sonnet/Haiku, Gemini Flash capability inference |
| `TestParseLitellmPricing` | Prefix filtering, missing costs, per-token conversion, capabilities |
| `TestProviderHelpers` | `provider_file()`, `provider_prefixes()` for all three providers |
| `TestCostMath` | Input/output/total cost formula, column breakdown percentage |

### `test_api.py` — HTTP endpoint tests
Uses FastAPI's `TestClient` — no real network, no browser.

| Class | Endpoint(s) tested |
|-------|-------------------|
| `TestHomePage` | `GET /` |
| `TestUpload` | `POST /upload` |
| `TestUploadText` | `POST /upload-text` |
| `TestCalculate` | `POST /calculate` |
| `TestCalculateText` | `POST /calculate-text` |
| `TestProviderData` | `GET /provider-data` — including stale-data and tab-switch order tests |
| `TestSavePricing` | `POST /save-pricing` |
| `TestTogglePin` | `POST /toggle-pin` — including pin order and dropdown star tests |
| `TestFetchPricing` | `GET /fetch-pricing` — network failure mock, untracked list |
| `TestExport` | `POST /export` — all sheets, budget note, untracked badge, content type |

### `test_html_integrity.py` — HTML/JS structural tests
Catches the class of bugs most common in this project: truncated files, missing JS functions, syntax errors, stale-data patterns.

| Class | What it checks |
|-------|---------------|
| `TestIndexHTML` | File completeness, JS syntax, 19 critical functions, 16 DOM IDs, provider tabs, theme, no template-builder leftovers, ⭐ update logic, fresh-fetch pattern |
| `TestSummaryHTML` | Completeness, JS syntax, all 14 Jinja template variables, all result sections, export button/function |
| `TestPricingJSON` | All 3 files exist, valid JSON, `gpt-4o` present, all models have `pinned` field, no NaN/Infinity |

### `conftest.py` — Shared fixtures
- `MOCK_PRICING` — 3 OpenAI models with known prices for deterministic tests
- `make_excel_bytes()` — creates in-memory Excel files for upload tests
- `tmp_project` fixture — isolated temp directory with copied `main.py`, `templates/`, and mock JSON files
- `client` fixture — `TestClient` pointed at the temp project
- `sample_excel` fixture — 3-row Excel with `prompt`, `response`, `context` columns
- `pricing_override` fixture — JSON string of mock pricing for `pricing_override` form field

---

## What the tests catch

| Bug class | Caught by |
|-----------|-----------|
| Truncated `index.html` / `summary.html` | `test_file_not_empty`, `test_has_closing_body` |
| JS syntax error killing all interactivity | `test_js_syntax_valid` |
| Missing function (e.g. `handleFile` deleted) | `test_function_defined[*]` |
| Missing DOM element | `test_dom_id_present[*]` |
| Cloudflare email obfuscation left in HTML | `test_no_cf_email_obfuscation` |
| NaN/Infinity in pricing JSON (breaks `JSON.parse`) | `test_no_nan_in_json` |
| Missing `pinned` field (breaks pin feature) | `test_all_models_have_pinned_field` |
| Wrong cost calculation | `TestCostMath` |
| Budget alert not showing | `test_budget_exceeded_shows_warning` |
| Column breakdown missing | `test_col_breakdown_present_for_multi_col` |
| Export producing invalid Excel or missing sheets | `TestExport` |
| Stale `ALL_PROVIDERS` cache losing pins after tab switch | `test_tab_switch_sees_fresh_data`, `test_pinned_models_at_top_after_tab_switch` |
| Pinned models not staying at top after tab switch | `test_pinned_models_at_top_after_tab_switch`, `test_rebuild_pricing_table_uses_ordered_models` |
| `⭐` not appearing in dropdown after pin | `test_reorder_model_selector_updates_star` |
| `switchProvider` using stale cache instead of fetching | `test_switch_provider_fetches_fresh_data` |
| `rebuildPricingTable` ignoring pinned order | `test_rebuild_pricing_table_uses_ordered_models` |

---

## Adding tests for a new feature

### 1. New endpoint
Add a class to `tests/test_api.py`:

```python
class TestMyNewEndpoint:
    def test_returns_200(self, client):
        r = client.get("/my-endpoint")
        assert r.status_code == 200

    def test_returns_expected_data(self, client):
        r = client.get("/my-endpoint")
        assert "key" in r.json()
```

### 2. New JS function or DOM element
Add to the parametrize lists in `tests/test_html_integrity.py`:

```python
# In test_function_defined parametrize:
"myNewFunction",

# In test_dom_id_present parametrize:
"myNewElementId",
```

### 3. New Jinja template variable
Add to `test_template_variable_present` in `TestSummaryHTML`:

```python
"my_new_variable",
```

### 4. New pricing logic
Add to `tests/test_logic.py`:

```python
class TestMyNewLogic:
    def test_something(self):
        result = my_function(input)
        assert result == expected
```

---

## Dependencies

```
pytest
fastapi[all]
httpx
pandas
openpyxl
tiktoken
node  # for JS syntax checks (optional — tests skip gracefully if absent)
```
