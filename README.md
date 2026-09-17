# ⬡ LLM Dataset Cost Calculator

A self-hosted web tool for estimating the **token cost of processing Excel datasets** through LLM APIs. Upload a spreadsheet, configure which columns are input/output/cached, pick a model, and get a full cost breakdown — with multi-provider comparison, projection, and Excel export.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **Multi-provider pricing** — OpenAI, Anthropic, Gemini side-by-side
- **Excel / Text / Word upload** — process `.xlsx`, `.txt`, or `.docx` files
- **Column-level configuration** — mark each column as input / output / cached input with a multiplier
- **Cost projection** — scale costs to 1K / 10K / 100K / 1M rows + custom
- **Model comparison** — all models ranked cheapest → most expensive for your dataset
- **Token breakdown by column** — see which column drives your token usage
- **Budget alerts** — set a max budget; get warned if exceeded with cheapest alternative
- **Model pinning ⭐** — pin favourites to the top of the list and dropdown
- **Export to Excel** — full report with Summary, Model Comparison, Row Breakdown, Column Breakdown sheets
- **Live pricing sync** — fetch latest rates from [LiteLLM](https://github.com/BerriAI/litellm) with change highlighting
- **Untracked model badges** — `⚠ untracked by LiteLLM` shown for manually-maintained rates
- **Light / Dark mode**
- **Test suite** — 190+ pytest tests covering all endpoints, JS integrity, pricing math

---

## Screenshots

> Upload → Configure → Calculate → Export

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/Hasan-Z/LLM-Dataset-Cost-Calculator.git
cd LLM-Dataset-Cost-Calculator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
# Windows
app/runme.bat

# Mac / Linux
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://127.0.0.1:8000** in your browser.

---

## Project Structure

```
app/
├── main.py                          # FastAPI application
├── runme.bat                        # Windows launcher
├── runme_tests.bat                  # Windows test runner
├── README.md                        # This file
├── openai_ai_models_pricing.json    # OpenAI model prices (editable)
├── anthropic_ai_models_pricing.json # Anthropic model prices (editable)
├── google_ai_models_pricing.json    # Gemini model prices (editable)
├── templates/
│   ├── index.html                   # Main calculator UI
│   └── summary.html                 # Results page
└── tests/
    ├── conftest.py                  # Shared fixtures and mock data
    ├── test_api.py                  # HTTP endpoint tests
    ├── test_logic.py                # Unit tests (pricing math, token counting)
    ├── test_html_integrity.py       # HTML/JS integrity tests
    └── README.md                    # How to write and run tests
```

---

## How It Works

### 1. Upload your dataset
Supports `.xlsx` / `.xls`, `.txt`, and `.docx` files.

### 2. Configure columns *(Excel only)*
For each column, choose:
- **Input** — tokens billed at the input rate
- **Output** — tokens billed at the output rate
- **Cached Input** — tokens billed at the cached input rate (prompt caching)
- **Skip** — ignore this column
- **Multiplier** — scale token count (e.g. `2` = count twice)

### 3. Select a model and calculate
The calculator:
- Counts tokens per row per column using [tiktoken](https://github.com/openai/tiktoken)
- Applies the model's $/1M token rates
- Shows a full row-by-row breakdown and model comparison table

### 4. Export
Download a `.xlsx` report with:
- **Summary** — metadata, rates, totals, cost projection
- **Model Comparison** — all models ranked by cost for your dataset
- **Column Breakdown** — token/cost contribution per column
- **Row Breakdown** — per-row token and cost detail

---

## Pricing Management

Prices are stored in three JSON files (one per provider). You can:

- **Edit manually** — directly edit the values in the pricing table on the UI
- **Fetch from LiteLLM** — click **⟳ Fetch Latest Rates** to pull live prices; changed rows highlight in amber
- **Save** — click **💾 Save Changes** to persist to disk
- Models not tracked by LiteLLM are shown with `⚠ untracked by LiteLLM` badge

### Pricing JSON format

```json
{
  "gpt-4o": {
    "input": 5.0,
    "output": 15.0,
    "cached_input": 2.5,
    "pinned": false,
    "litellm_tracked": true
  }
}
```

All prices are in **$/1M tokens**.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Home page |
| `POST` | `/upload` | Detect columns from Excel file |
| `POST` | `/upload-text` | Analyse Text/Word file |
| `POST` | `/calculate` | Calculate costs for Excel dataset |
| `POST` | `/calculate-text` | Calculate costs for Text/Word file |
| `GET` | `/provider-data` | Fresh pricing data for a provider (tab switching) |
| `POST` | `/save-pricing` | Persist edited prices to JSON |
| `POST` | `/toggle-pin` | Pin/unpin a model |
| `GET` | `/fetch-pricing` | Fetch latest rates from LiteLLM |
| `POST` | `/export` | Generate Excel report |

---

## Running Tests

```bash
# Windows
runme_tests.bat

# Mac / Linux
python -m pytest tests/ -v
```

**190+ tests** covering:
- All HTTP endpoints
- Token counting and pricing math
- `infer_capabilities` for OpenAI / Anthropic / Gemini
- HTML completeness and JS syntax validity
- All 18 critical JS functions present
- Pricing JSON integrity (no NaN, all models have `pinned` field)
- Stale-data bugs (pin state and saved prices survive tab switches)

See [`tests/README.md`](tests/README.md) for details on adding tests for new features.

---

## Adding New Models

Edit the relevant JSON file directly, or use the UI:

1. Open the pricing table
2. Make your edits
3. Click **💾 Save Changes**

Or run **⟳ Fetch Latest Rates** to pull all current models from LiteLLM automatically.

---

## Requirements

- Python 3.10+
- Node.js (for JS syntax checks in tests — optional)

```
fastapi
uvicorn
jinja2
pandas
openpyxl
tiktoken
httpx
python-multipart
python-docx
```

---

## License

MIT — free to use, modify, and distribute.
