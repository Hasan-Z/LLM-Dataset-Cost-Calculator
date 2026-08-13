import json
import io
import hashlib
import httpx
import openpyxl
from datetime import datetime, timezone
import pandas as pd
import tiktoken
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

PROVIDERS = {
    "openai":    {"file": "openai_ai_models_pricing.json",    "prefixes": ("gpt-", "o1", "o3", "o4", "chatgpt")},
    "anthropic": {"file": "anthropic_ai_models_pricing.json", "prefixes": ("claude-",)},
    "gemini":    {"file": "google_ai_models_pricing.json",    "prefixes": ("gemini-",)},
}
LITELLM_PRICING_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

# Legacy alias for backward compat
PRICING_FILE    = PROVIDERS["openai"]["file"]
OPENAI_PREFIXES = PROVIDERS["openai"]["prefixes"]


def provider_file(provider: str) -> str:
    return PROVIDERS.get(provider, PROVIDERS["openai"])["file"]


def provider_prefixes(provider: str) -> tuple:
    return PROVIDERS.get(provider, PROVIDERS["openai"])["prefixes"]


def load_pricing(provider: str = "openai") -> dict:
    fname = provider_file(provider)
    try:
        with open(fname, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_pricing(data: dict, source: str = "manual", provider: str = "openai"):
    data["last_updated"]        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data["last_updated_source"] = source
    # Ensure every model has pinned field
    for key, val in data.items():
        if key in ("last_updated", "last_updated_source"):
            continue
        if isinstance(val, dict) and "pinned" not in val:
            val["pinned"] = False
    with open(provider_file(provider), "w") as f:
        json.dump(data, f, indent=2)


def count_tokens(text, model_name):
    if text is None:
        return 0
    text = str(text)
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def safe_price(val):
    if val is None:
        return None
    try:
        return round(float(val) * 1_000_000, 6)
    except (TypeError, ValueError):
        return None


def infer_capabilities(model_name: str, mode: str, info: dict) -> dict:
    n             = model_name.lower()
    is_tts        = "tts" in n or mode == "audio_speech"
    is_transcribe = "transcribe" in n or "whisper" in n or mode == "audio_transcription"
    is_realtime   = "realtime" in n
    is_audio      = "audio" in n and not is_tts and not is_transcribe
    is_search     = "search" in n and "realtime" not in n
    is_embedding  = mode == "embedding"
    is_image      = mode == "image_generation"
    is_moderation = mode == "moderation"
    is_o_series   = n.startswith("o1") or n.startswith("o3") or n.startswith("o4")
    is_gpt5       = n.startswith("gpt-5")
    is_gpt41      = n.startswith("gpt-4.1")
    is_gpt4o      = n.startswith("gpt-4o") and not is_tts and not is_transcribe
    is_chat       = mode == "chat" or mode is None

    # ── Anthropic Claude ──
    is_claude        = n.startswith("claude-")
    is_claude_opus   = is_claude and "opus" in n
    is_claude_sonnet = is_claude and "sonnet" in n
    is_claude_haiku  = is_claude and "haiku" in n
    is_claude3plus   = is_claude and any(x in n for x in ["claude-3", "claude-4"])

    # ── Google Gemini ──
    is_gemini        = n.startswith("gemini-")
    is_gemini_pro    = is_gemini and "pro" in n
    is_gemini_flash  = is_gemini and "flash" in n

    caps = {k: False for k in [
        "supports_vision", "supports_function_calling", "supports_reasoning",
        "supports_web_search", "supports_audio_input", "supports_audio_output",
        "supports_prompt_caching", "supports_response_schema",
    ]}

    # ── OpenAI rules ──
    if is_tts:
        caps["supports_audio_output"] = True
    elif is_transcribe:
        caps["supports_audio_input"] = True
    elif is_realtime or (is_audio and is_chat):
        caps.update({"supports_vision": True, "supports_function_calling": True,
                     "supports_audio_input": True, "supports_audio_output": True,
                     "supports_response_schema": True})
    elif is_search:
        caps.update({"supports_function_calling": True, "supports_web_search": True,
                     "supports_vision": True})
    elif is_o_series:
        caps.update({"supports_vision": True, "supports_reasoning": True,
                     "supports_function_calling": True, "supports_prompt_caching": True,
                     "supports_response_schema": True})
    elif is_gpt5:
        caps.update({"supports_vision": True, "supports_reasoning": True,
                     "supports_function_calling": True, "supports_web_search": True,
                     "supports_prompt_caching": True, "supports_response_schema": True})
    elif is_gpt41 or is_gpt4o:
        caps.update({"supports_vision": True, "supports_function_calling": True,
                     "supports_prompt_caching": True, "supports_response_schema": True})

    # ── Anthropic Claude rules ──
    elif is_claude_opus or is_claude_sonnet:
        caps.update({"supports_vision": True, "supports_function_calling": True,
                     "supports_prompt_caching": True, "supports_response_schema": True,
                     "supports_reasoning": is_claude_opus or "3-7" in n or "4" in n})
    elif is_claude_haiku:
        caps.update({"supports_vision": True, "supports_function_calling": True,
                     "supports_prompt_caching": True, "supports_response_schema": True})
    elif is_claude:
        caps.update({"supports_vision": True, "supports_function_calling": True,
                     "supports_prompt_caching": True, "supports_response_schema": True})

    # ── Google Gemini rules ──
    elif is_gemini_pro:
        caps.update({"supports_vision": True, "supports_function_calling": True,
                     "supports_prompt_caching": True, "supports_response_schema": True,
                     "supports_reasoning": "2.5" in n})
    elif is_gemini_flash:
        caps.update({"supports_vision": True, "supports_function_calling": True,
                     "supports_prompt_caching": True, "supports_response_schema": True,
                     "supports_reasoning": "2.5" in n})
    elif is_gemini:
        caps.update({"supports_vision": True, "supports_function_calling": True,
                     "supports_response_schema": True})

    elif not (is_embedding or is_image or is_moderation) and is_chat:
        caps.update({"supports_function_calling": True, "supports_response_schema": True})

    # LiteLLM explicit flags always override inference
    for key in list(caps.keys()):
        if key in info:
            caps[key] = bool(info[key])

    return caps


def parse_litellm_pricing(raw: dict, prefixes: tuple = None) -> dict:
    result = {}
    active_prefixes = prefixes if prefixes else OPENAI_PREFIXES
    for model_name, info in raw.items():
        if not isinstance(info, dict):
            continue
        if not any(model_name.startswith(p) for p in active_prefixes):
            continue
        input_cost     = safe_price(info.get("input_cost_per_token"))
        output_cost    = safe_price(info.get("output_cost_per_token"))
        cached_cost    = safe_price(info.get("cache_read_input_token_cost"))
        reasoning_cost = safe_price(info.get("output_cost_per_reasoning_token"))
        if input_cost is None or output_cost is None:
            continue
        mode = info.get("mode")
        caps = infer_capabilities(model_name, mode or "", info)
        result[model_name] = {
            "input": input_cost, "cached_input": cached_cost,
            "output": output_cost, "reasoning_output": reasoning_cost,
            "max_input_tokens": info.get("max_input_tokens"),
            "max_output_tokens": info.get("max_output_tokens"),
            "mode": mode, "deprecation_date": info.get("deprecation_date"),
            **caps,
        }
    return result


def safe_json_float(v):
    """Convert float to JSON-safe value — replace NaN/Inf with None."""
    if v is None:
        return None
    try:
        f = float(v)
        if f != f or f == float('inf') or f == float('-inf'):  # NaN or Inf check
            return None
        return f
    except (TypeError, ValueError):
        return v


def process_provider_pricing(provider: str):
    """Load, split, and sort pricing for a provider. Returns (prices_only, metadata, sorted_models, last_updated, last_updated_source)."""
    pricing             = load_pricing(provider)
    last_updated        = pricing.pop("last_updated", None)
    last_updated_source = pricing.pop("last_updated_source", "manual")

    PRICE_FIELDS = {"input", "cached_input", "output", "reasoning_output"}
    CAP_KEYS     = {"supports_vision", "supports_function_calling", "supports_reasoning",
                    "supports_web_search", "supports_audio_input", "supports_audio_output",
                    "supports_prompt_caching", "supports_response_schema"}
    prices_only = {}
    metadata    = {}
    for model, data in pricing.items():
        if not isinstance(data, dict):
            continue
        prices_only[model] = {k: safe_json_float(v) for k, v in data.items() if k in PRICE_FIELDS}
        raw_meta           = {k: v for k, v in data.items() if k not in PRICE_FIELDS}
        has_explicit_caps  = any(k in raw_meta for k in CAP_KEYS)
        caps               = infer_capabilities(model, raw_meta.get("mode", "") or "", raw_meta)
        caps["_capabilities_inferred"] = not has_explicit_caps
        metadata[model]    = {**raw_meta, **caps}

    def sort_key(m):
        return (0 if metadata.get(m, {}).get("pinned") else 1, m.lower())

    return prices_only, metadata, sorted(pricing.keys(), key=sort_key), last_updated, last_updated_source


@app.get("/provider-data")
async def provider_data(provider: str = "openai"):
    """Return fresh pricing + metadata for a provider (called on tab switch)."""
    prices, meta, models, lu, lus = process_provider_pricing(provider)
    return JSONResponse({
        "prices":              prices,
        "metadata":            meta,
        "models":              models,
        "last_updated":        lu,
        "last_updated_source": lus,
    })


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Load all 3 providers
    all_providers = {}
    for p in ["openai", "anthropic", "gemini"]:
        prices, meta, models, lu, lus = process_provider_pricing(p)
        all_providers[p] = {
            "prices":              prices,
            "metadata":            meta,
            "models":              models,
            "last_updated":        lu,
            "last_updated_source": lus,
        }

    # Default to openai for initial render
    oa = all_providers["openai"]
    return templates.TemplateResponse("index.html", {
        "request":             request,
        "models":              oa["models"],
        "pricing":             oa["prices"],
        "metadata":            oa["metadata"],
        "last_updated":        oa["last_updated"],
        "last_updated_source": oa["last_updated_source"],
        "all_providers":       all_providers,
    })


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Read sheet names and column headers only — never parses data rows.
    Uses openpyxl read_only mode which is ~450× faster than pd.parse for large files.
    Returns all sheets' columns so the browser can switch tabs without re-uploading.
    """
    content = await file.read()
    wb      = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    sheets  = wb.sheetnames
    all_columns: dict = {}
    for sheet_name in sheets:
        ws     = wb[sheet_name]
        header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
        all_columns[sheet_name] = [str(c) if c is not None else "" for c in header]
    wb.close()
    first = sheets[0] if sheets else ""
    return {
        "sheets":      sheets,
        "all_columns": all_columns,          # all sheets → columns map (for tab switching)
        "columns":     all_columns.get(first, []),  # first sheet columns (backward compat)
        "active_sheet": first,
    }


@app.post("/upload-text")
async def upload_text(file: UploadFile = File(...), file_type: str = Form(...)):
    try:
        raw = await file.read()
        if file_type == "text":
            text = raw.decode("utf-8", errors="replace")
        elif file_type == "word":
            try:
                import docx as docx_lib
                doc  = docx_lib.Document(io.BytesIO(raw))
                text = "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                return JSONResponse({"error": "python-docx not installed. Run: pip install python-docx"})
        else:
            return JSONResponse({"error": f"Unknown file type: {file_type}"})
        chars      = len(text)
        words      = len(text.split())
        lines      = len(text.splitlines())
        try:
            est_tokens = len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:
            est_tokens = words * 4 // 3
        return JSONResponse({"chars": chars, "words": words, "lines": lines, "est_tokens": est_tokens})
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.post("/save-pricing")
async def save_pricing_endpoint(request: Request):
    body     = await request.json()
    provider = body.pop("__provider__", "openai")
    source   = body.pop("__source__", "manual")
    try:
        existing = load_pricing(provider)
        existing.pop("last_updated", None)
        existing.pop("last_updated_source", None)
        PRICE_FIELDS = {"input", "cached_input", "output", "reasoning_output"}
        for model, prices in body.items():
            if model not in existing:
                existing[model] = {}
            for field in PRICE_FIELDS:
                if field in prices:
                    existing[model][field] = prices[field]
        save_pricing(existing, source, provider)
        pricing             = load_pricing(provider)
        last_updated        = pricing.pop("last_updated", None)
        last_updated_source = pricing.pop("last_updated_source", "manual")
        return JSONResponse({"success": True, "last_updated": last_updated, "source": last_updated_source})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/toggle-pin")
async def toggle_pin(request: Request):
    body     = await request.json()
    model    = body.get("model")
    provider = body.get("provider", "openai")
    if not model:
        return JSONResponse({"success": False, "error": "No model specified"}, status_code=400)
    try:
        existing = load_pricing(provider)
        if model not in existing or not isinstance(existing[model], dict):
            return JSONResponse({"success": False, "error": "Model not found"}, status_code=404)
        new_pinned = not existing[model].get("pinned", False)
        existing[model]["pinned"] = new_pinned
        with open(provider_file(provider), "w") as f:
            json.dump(existing, f, indent=2)
        return JSONResponse({"success": True, "pinned": new_pinned})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/fetch-pricing")
async def fetch_pricing(provider: str = "openai"):
    current = load_pricing(provider)
    current.pop("last_updated", None)
    current.pop("last_updated_source", None)
    prefixes = provider_prefixes(provider)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(LITELLM_PRICING_URL)
            resp.raise_for_status()
            raw  = resp.json()
        fetched = parse_litellm_pricing(raw, prefixes)
        if not fetched:
            return JSONResponse({"success": False, "error": f"No matching models found for provider '{provider}'.", "current": current})

        # Update litellm_tracked flag on every model in the JSON
        full_pricing = load_pricing(provider)
        tracking_changed = False
        fetched_names = set(fetched.keys())
        for model, data in full_pricing.items():
            if model in ("last_updated", "last_updated_source"):
                continue
            if not isinstance(data, dict):
                continue
            was_tracked = data.get("litellm_tracked", None)
            is_tracked  = model in fetched_names
            if was_tracked != is_tracked:
                full_pricing[model]["litellm_tracked"] = is_tracked
                tracking_changed = True
        # Also mark new fetched models not yet in JSON
        for model in fetched_names:
            if model not in full_pricing:
                full_pricing[model] = {**fetched[model], "litellm_tracked": True, "pinned": False}
                tracking_changed = True
        if tracking_changed:
            with open(provider_file(provider), "w") as f:
                json.dump(full_pricing, f, indent=2)

        # Build changes vs current UI
        changes = {}
        for model, prices in fetched.items():
            if model in current:
                model_changes = {}
                for field, new_val in prices.items():
                    old_val = current[model].get(field) if isinstance(current[model], dict) else None
                    try:
                        nv = round(float(new_val), 6) if new_val is not None else None
                    except (TypeError, ValueError):
                        nv = new_val
                    try:
                        ov = round(float(old_val), 6) if old_val is not None else None
                    except (TypeError, ValueError):
                        ov = old_val
                    if nv != ov:
                        model_changes[field] = {"old": old_val, "new": new_val}
                if model_changes:
                    changes[model] = model_changes
            else:
                changes[model] = {f: {"old": None, "new": v} for f, v in prices.items()}

        # Include tracked status in response so JS can update badges immediately
        untracked = [m for m in current if m not in fetched_names]
        return JSONResponse({
            "success":   True,
            "fetched":   fetched,
            "changes":   changes,
            "current":   current,
            "untracked": untracked,
        })
    except httpx.HTTPError as e:
        return JSONResponse({"success": False, "error": f"Network error: {str(e)}", "current": current})
    except Exception as e:
        return JSONResponse({"success": False, "error": f"Unexpected error: {str(e)}", "current": current})




@app.post("/export")
async def export_excel(request: Request):
    """Generate and return an Excel file with 3 sheets: Summary, Model Comparison, Row Breakdown."""
    from fastapi.responses import StreamingResponse
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    import io as _io

    data            = await request.json()
    selected_model  = data.get("selected_model", "")
    has_cached_cols = data.get("has_cached_cols", False)
    caching_supported = data.get("caching_supported", True)
    file_name       = data.get("file_name", "")
    file_md5        = data.get("file_md5", "")
    model_rates     = data.get("model_rates", {})
    budget          = data.get("budget", 0) or 0
    totals          = data.get("totals", {})
    rows            = data.get("rows", [])
    model_comp      = data.get("model_comparison", [])
    col_breakdown   = data.get("col_breakdown", [])

    wb = openpyxl.Workbook()

    # ── Styles ──
    def hdr_font():  return Font(bold=True, color="FFFFFF", name="Calibri", size=10)
    def hdr_fill():  return PatternFill("solid", fgColor="1F4E79")
    def sub_fill():  return PatternFill("solid", fgColor="2E75B6")
    def warn_fill(): return PatternFill("solid", fgColor="C55A11")
    def grn_font():  return Font(bold=True, color="375623", name="Calibri", size=10)
    def center():    return Alignment(horizontal="center", vertical="center")
    def right():     return Alignment(horizontal="right")
    thin = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
    )

    def style_header_row(ws, row_num, fill):
        for cell in ws[row_num]:
            cell.font      = hdr_font()
            cell.fill      = fill
            cell.alignment = center()
            cell.border    = thin

    def auto_width(ws):
        for col in ws.columns:
            max_len = max((len(str(c.value or "")) for c in col), default=10)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 40)

    def fmt_cost(v): return round(float(v), 6) if v is not None else 0

    # ── Sheet 1: Summary ──
    ws1 = wb.active
    ws1.title = "Summary"
    ws1.append(["LLM Cost Calculator — Summary"])
    ws1["A1"].font = Font(bold=True, size=14, name="Calibri", color="1F4E79")
    ws1.append(["Selected Model", selected_model])
    # Note if selected model is untracked by LiteLLM
    sel_model_data = next((m for m in model_comp if m.get("model") == selected_model), None)
    if sel_model_data and sel_model_data.get("litellm_tracked") is False:
        ws1.append(["", "⚠ This model is not tracked by LiteLLM — rates are manually maintained"])
        r = ws1.max_row
        ws1.cell(r, 2).font = Font(italic=True, color="9298A8", name="Calibri", size=9)
    ws1.append(["File Name",      file_name])
    ws1.append(["File MD5",       file_md5])
    ws1.append(["Export Date",    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")])
    if budget and budget > 0:
        ws1.append(["Budget ($)", round(float(budget), 4)])

    # Caching notice if applicable
    if has_cached_cols and not caching_supported:
        ws1.append([])
        ws1.append([f"⚠ NOTE: {selected_model} does not support prompt caching — cached input columns were billed at the standard input rate."])
        r = ws1.max_row
        ws1.cell(r, 1).font      = Font(bold=True, color="C55A11", name="Calibri", size=10)
        ws1.cell(r, 1).fill      = PatternFill("solid", fgColor="FFF2CC")
        ws1.merge_cells(f"A{r}:B{r}")

    # Budget exceeded notice
    total_cost = totals.get("total_cost", 0) or 0
    if budget and budget > 0 and total_cost > budget:
        ws1.append([])
        overage = total_cost - budget
        ws1.append([f"💸 BUDGET EXCEEDED: ${total_cost:.4f} vs ${budget:.4f} budget (${overage:.4f} over)"])
        r = ws1.max_row
        ws1.cell(r, 1).font      = Font(bold=True, color="C55A11", name="Calibri", size=10)
        ws1.cell(r, 1).fill      = PatternFill("solid", fgColor="FFF2CC")
        ws1.merge_cells(f"A{r}:B{r}")
        # Find cheapest model within budget
        cheaper = [m for m in model_comp if m.get("total_cost", 0) <= budget]
        if cheaper:
            cheaper_sorted = sorted(cheaper, key=lambda x: x["total_cost"])
            ws1.append([f"✓ Cheapest model within budget: {cheaper_sorted[0]['model']} — ${cheaper_sorted[0]['total_cost']:.4f}"])
            r = ws1.max_row
            ws1.cell(r, 1).font = Font(bold=True, color="375623", name="Calibri", size=10)
            ws1.cell(r, 1).fill = PatternFill("solid", fgColor="E2EFDA")
            ws1.merge_cells(f"A{r}:B{r}")

    ws1.append([])

    # ── Calculation Rates ──
    ws1.append(["Calculation Rates"])
    ws1.cell(ws1.max_row, 1).font = Font(bold=True, size=11, name="Calibri", color="1F4E79")

    rate_headers = ["Token Type", "Rate ($/1M tokens)"]
    ws1.append(rate_headers)
    style_header_row(ws1, ws1.max_row, hdr_fill())

    rate_rows = [("Input",  model_rates.get("input",  0))]
    if has_cached_cols:
        cached_rate_label = "Cached Input" if caching_supported else "Cached Input (billed as Input)"
        rate_rows.append((cached_rate_label, model_rates.get("cached_input", 0)))
    rate_rows.append(("Output", model_rates.get("output", 0)))

    for label, rate in rate_rows:
        ws1.append([label, round(float(rate), 6) if rate else 0])
        r = ws1.max_row
        ws1.cell(r, 1).font      = Font(bold=True, name="Calibri", size=10)
        ws1.cell(r, 2).alignment = right()
        for col in [1, 2]:
            ws1.cell(r, col).border = thin

    ws1.append([])

    summary_headers = ["Metric", "Value"]
    ws1.append(summary_headers)
    style_header_row(ws1, ws1.max_row, hdr_fill())

    summary_rows = [
        ("Total Rows",       totals.get("rows", 0)),
        ("Input Tokens",     totals.get("input_tokens", 0)),
        ("Output Tokens",    totals.get("output_tokens", 0)),
        ("Total Tokens",     totals.get("total_tokens", 0)),
        ("Input Cost ($)",   fmt_cost(totals.get("input_cost", 0))),
        ("Output Cost ($)",  fmt_cost(totals.get("output_cost", 0))),
        ("Total Cost ($)",   fmt_cost(totals.get("total_cost", 0))),
    ]
    if has_cached_cols:
        summary_rows.insert(3, ("Cached Tokens", totals.get("cached_tokens", 0)))
        summary_rows.insert(7, ("Cached Cost ($)", fmt_cost(totals.get("cached_cost", 0))))

    for label, value in summary_rows:
        ws1.append([label, value])
        row = ws1.max_row
        ws1.cell(row, 1).font      = Font(bold=True, name="Calibri", size=10)
        ws1.cell(row, 2).alignment = right()
        for col in [1, 2]:
            ws1.cell(row, col).border = thin

    auto_width(ws1)

    # ── Row Breakdown (single row only — merged into Summary before Projection) ──
    row_headers = ["Row", "Input Tokens", "Output Tokens", "Total Tokens",
                   "Input Cost ($)", "Output Cost ($)", "Total Cost ($)"]
    if has_cached_cols:
        row_headers.insert(2, "Cached Tokens")
        row_headers.insert(6, "Cached Cost ($)")

    if len(rows) == 1:
        ws1.append([])
        ws1.append(["Row Breakdown"])
        ws1.cell(ws1.max_row, 1).font = Font(bold=True, size=11, name="Calibri", color="1F4E79")
        ws1.append(row_headers)
        style_header_row(ws1, ws1.max_row, hdr_fill())
        row = rows[0]
        row_data = [
            row["row"],
            row.get("input_tokens", 0),
            row.get("output_tokens", 0),
            row.get("total_tokens", 0),
            fmt_cost(row.get("input_cost", 0)),
            fmt_cost(row.get("output_cost", 0)),
            fmt_cost(row.get("total_cost", 0)),
        ]
        if has_cached_cols:
            row_data.insert(2, row.get("cached_tokens", 0))
            row_data.insert(6, fmt_cost(row.get("cached_cost", 0)))
        ws1.append(row_data)
        r = ws1.max_row
        for col in range(1, len(row_headers) + 1):
            ws1.cell(r, col).border    = thin
            ws1.cell(r, col).alignment = Alignment(horizontal="left") if col == 1 else right()
        ws1.cell(r, len(row_headers)).font = Font(bold=True, color="375623", name="Calibri", size=10)
        auto_width(ws1)

    # ── Sheet 1b: Cost Projection (appended to Summary sheet) ──
    num_rows   = totals.get("rows", 1) or 1
    per_input  = totals.get("input_tokens",  0) / num_rows
    per_output = totals.get("output_tokens", 0) / num_rows
    per_cached = totals.get("cached_tokens", 0) / num_rows
    per_total  = totals.get("total_tokens",  0) / num_rows
    per_cost   = totals.get("total_cost",    0) / num_rows

    ws1.append([])
    ws1.append(["Cost Projection"])
    ws1.cell(ws1.max_row, 1).font = Font(bold=True, size=12, name="Calibri", color="1F4E79")

    proj_headers = ["Scale", "Rows", "Input Tokens", "Output Tokens", "Total Tokens", "Total Cost ($)"]
    if has_cached_cols:
        proj_headers.insert(3, "Cached Tokens")
    ws1.append(proj_headers)
    style_header_row(ws1, ws1.max_row, hdr_fill())  # same blue as other headers

    # Read custom scale from request if sent
    custom_rows = data.get("custom_projection_rows", 0)

    proj_scales = [
        ("Current (" + str(num_rows) + ")", num_rows, False),
        ("1K",    1_000,       False),
        ("10K",   10_000,      False),
        ("100K",  100_000,     False),
        ("1M",    1_000_000,   False),
    ]
    if custom_rows and custom_rows > 0:
        proj_scales.append(("Custom (" + str(int(custom_rows)) + ")", int(custom_rows), True))

    custom_fill = PatternFill("solid", fgColor="DBEAFE")  # light blue for custom row

    for label, scale_rows, is_custom in proj_scales:
        row_data = [
            label,
            scale_rows,
            round(per_input  * scale_rows),
            round(per_output * scale_rows),
            round(per_total  * scale_rows),
            round(per_cost   * scale_rows, 6),
        ]
        if has_cached_cols:
            row_data.insert(3, round(per_cached * scale_rows))
        ws1.append(row_data)
        r = ws1.max_row
        for col in range(1, len(proj_headers) + 1):
            ws1.cell(r, col).border    = thin
            ws1.cell(r, col).alignment = right() if col > 1 else Alignment(horizontal="left")
            if is_custom:
                ws1.cell(r, col).fill = custom_fill
                ws1.cell(r, col).font = Font(bold=True, color="1D4ED8", name="Calibri", size=10)
        # Green for total cost column
        cost_col = len(proj_headers)
        if not is_custom:
            ws1.cell(r, cost_col).font = Font(bold=True, color="375623", name="Calibri", size=10)

    auto_width(ws1)
    ws2 = wb.create_sheet("Model Comparison")
    comp_headers = ["#", "Model", "Input Cost ($)", "Output Cost ($)", "Total Cost ($)"]
    if has_cached_cols:
        comp_headers.insert(3, "Cached Cost ($)")

    ws2.append(comp_headers)
    style_header_row(ws2, 1, hdr_fill())

    for i, m in enumerate(model_comp, 1):
        row_data = [
            i,
            ("⭐ " if m.get("pinned") else "") + m["model"],
            fmt_cost(m.get("input_cost", 0)),
            fmt_cost(m.get("output_cost", 0)),
            fmt_cost(m.get("total_cost", 0)),
        ]
        if has_cached_cols:
            cached_label = fmt_cost(m.get("cached_cost", 0))
            if not m.get("caching_supported"):
                cached_label = str(cached_label) + " (billed as input)"
            row_data.insert(3, cached_label)

        # Append untracked note to model name if applicable
        if m.get("litellm_tracked") is False:
            row_data[1] = row_data[1] + " ⚠ untracked by LiteLLM"

        ws2.append(row_data)
        r = ws2.max_row
        # Highlight selected model
        if m["model"] == selected_model:
            for col in range(1, len(comp_headers) + 1):
                ws2.cell(r, col).fill = PatternFill("solid", fgColor="E2EFDA")
        # Highlight cheapest
        if i == 1:
            ws2.cell(r, len(comp_headers)).font = grn_font()
        for col in range(1, len(comp_headers) + 1):
            ws2.cell(r, col).border = thin
            if col > 2:
                ws2.cell(r, col).alignment = right()

    auto_width(ws2)

    # ── Sheet 2b: Column Breakdown (if available) ──
    if col_breakdown:
        ws_cb = wb.create_sheet("Column Breakdown")
        cb_headers = ["Column", "Type", "Tokens", "% of Total", "Cost ($)"]
        ws_cb.append(cb_headers)
        style_header_row(ws_cb, 1, hdr_fill())

        type_fills = {
            "input":        PatternFill("solid", fgColor="DBEAFE"),  # blue
            "cached_input": PatternFill("solid", fgColor="FEF3C7"),  # amber
            "output":       PatternFill("solid", fgColor="DCFCE7"),  # green
        }
        for cb in col_breakdown:
            cb_type = cb.get("type", "")
            ws_cb.append([
                cb.get("column", ""),
                cb_type,
                cb.get("tokens", 0),
                str(cb.get("pct", 0)) + "%",
                fmt_cost(cb.get("cost", 0)),
            ])
            r = ws_cb.max_row
            for col in range(1, 6):
                ws_cb.cell(r, col).border    = thin
                ws_cb.cell(r, col).alignment = right() if col >= 3 else Alignment(horizontal="left")
            # Colour the type cell
            fill = type_fills.get(cb_type)
            if fill:
                ws_cb.cell(r, 2).fill = fill
            # Green cost
            ws_cb.cell(r, 5).font = Font(bold=True, color="375623", name="Calibri", size=10)
        auto_width(ws_cb)

    # ── Sheet 3: Row Breakdown (multi-row — separate sheet) ──
    if len(rows) > 1:
        ws3 = wb.create_sheet("Row Breakdown")
        ws3.append(row_headers)
        style_header_row(ws3, 1, hdr_fill())

        for row in rows:
            row_data = [
                row["row"],
                row.get("input_tokens", 0),
                row.get("output_tokens", 0),
                row.get("total_tokens", 0),
                fmt_cost(row.get("input_cost", 0)),
                fmt_cost(row.get("output_cost", 0)),
                fmt_cost(row.get("total_cost", 0)),
            ]
            if has_cached_cols:
                row_data.insert(2, row.get("cached_tokens", 0))
                row_data.insert(6, fmt_cost(row.get("cached_cost", 0)))

            ws3.append(row_data)
            r = ws3.max_row
            for col in range(1, len(row_headers) + 1):
                ws3.cell(r, col).border    = thin
                ws3.cell(r, col).alignment = Alignment(horizontal="left") if col == 1 else right()
            if r % 2 == 0:
                for col in range(1, len(row_headers) + 1):
                    ws3.cell(r, col).fill = PatternFill("solid", fgColor="F2F2F2")
            ws3.cell(r, len(row_headers)).font = Font(bold=True, color="375623", name="Calibri", size=10)

        auto_width(ws3)

    # Save to buffer and stream
    buf = _io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = f"llm_cost_{selected_model}_{datetime.now().strftime('%Y%m%d')}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )



@app.post("/calculate-text", response_class=HTMLResponse)
async def calculate_text(
    request: Request,
    file: UploadFile     = File(...),
    file_type: str       = Form(...),
    model_name: str      = Form(...),
    pricing_override: str = Form(default=None),
    budget: float        = Form(default=0)
):
    try:
        raw = await file.read()
        if file_type == "text":
            text = raw.decode("utf-8", errors="replace")
        elif file_type == "word":
            try:
                import docx as docx_lib
                doc  = docx_lib.Document(io.BytesIO(raw))
                text = "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                return HTMLResponse("<h2>Error: python-docx not installed. Run: pip install python-docx</h2>", status_code=400)
        else:
            return HTMLResponse(f"<h2>Unknown file type: {file_type}</h2>", status_code=400)
    except Exception as e:
        return HTMLResponse(f"<h2>Error reading file: {e}</h2>", status_code=400)

    pricing = json.loads(pricing_override) if pricing_override else load_pricing()
    pricing.pop("last_updated", None)
    pricing.pop("last_updated_source", None)

    if model_name not in pricing:
        return HTMLResponse(f"<h2>Model not found.</h2>", status_code=400)

    input_price  = pricing[model_name].get("input") or 0
    input_tokens = count_tokens(text, model_name)
    input_cost   = (input_tokens / 1_000_000) * input_price

    rows = [{"row": 1, "input_tokens": input_tokens, "output_tokens": 0, "cached_tokens": 0,
              "total_tokens": input_tokens, "input_cost": input_cost,
              "output_cost": 0.0, "cached_cost": 0.0, "total_cost": input_cost}]

    mc = []
    for model, price in pricing.items():
        if not isinstance(price, dict):
            continue
        m_cost = (input_tokens / 1_000_000) * (price.get("input") or 0)
        mc.append({"model": model, "input_cost": m_cost, "output_cost": 0.0, "cached_cost": 0.0,
                   "total_cost": m_cost, "caching_supported": price.get("cached_input") is not None,
                   "pinned": bool(price.get("pinned", False)),
                   "litellm_tracked": price.get("litellm_tracked", None)})
    mc.sort(key=lambda x: (0 if x.get("pinned") else 1, x["total_cost"]))

    return templates.TemplateResponse("summary.html", {
        "request": request, "rows": rows, "selected_model": model_name,
        "total_input_tokens": input_tokens, "total_output_tokens": 0, "total_cached_tokens": 0,
        "dataset_total_tokens": input_tokens, "total_input_cost": input_cost,
        "total_output_cost": 0.0, "total_cached_cost": 0.0, "dataset_total_cost": input_cost,
        "model_comparison": mc, "has_cached_cols": False, "caching_supported": True,
        "file_name": file.filename or "unknown", "file_md5": hashlib.md5(raw).hexdigest(),
        "budget": budget, "col_breakdown": [],
    })


def _process_sheet(df, cols, model_name, input_price, output_price, cached_price_eff):
    """Process one sheet's DataFrame. Returns (rows, col_totals, ti, to, tc, tic, toc, tcc)."""
    col_totals = {
        col: {"type": cfg.get("type", "skip"), "tokens": 0, "cost": 0.0}
        for col, cfg in cols.items()
        if cfg.get("type", "skip") != "skip"
    }
    rows = []
    ti = to = tc = tic = toc = tcc = 0
    for i, row in df.iterrows():
        ri = ro = rc = 0
        for col, cfg in cols.items():
            t = cfg.get("type", "skip")
            if t == "skip":
                continue
            m = cfg.get("multiplier", 1)
            if not isinstance(m, (int, float)) or m <= 0:
                continue
            n = int(count_tokens(row.get(col, ""), model_name) * m)
            if t == "input":
                ri += n
                col_totals[col]["tokens"] += n
                col_totals[col]["cost"]   += (n / 1_000_000) * input_price
            elif t == "output":
                ro += n
                col_totals[col]["tokens"] += n
                col_totals[col]["cost"]   += (n / 1_000_000) * output_price
            elif t == "cached_input":
                rc += n
                col_totals[col]["tokens"] += n
                col_totals[col]["cost"]   += (n / 1_000_000) * cached_price_eff
        ic = (ri / 1_000_000) * input_price
        oc = (ro / 1_000_000) * output_price
        cc = (rc / 1_000_000) * cached_price_eff
        ti += ri; to += ro; tc += rc
        tic += ic; toc += oc; tcc += cc
        rows.append({"row": i + 1,
                     "input_tokens": ri,  "output_tokens": ro,  "cached_tokens": rc,
                     "total_tokens": ri + ro + rc,
                     "input_cost":   ic,  "output_cost":   oc,  "cached_cost":   cc,
                     "total_cost":   ic + oc + cc})
    return rows, col_totals, ti, to, tc, tic, toc, tcc


def _build_col_breakdown(col_totals, total_all_tokens):
    breakdown = []
    for col, info in col_totals.items():
        breakdown.append({
            "column": col,
            "type":   info["type"],
            "tokens": info["tokens"],
            "cost":   round(info["cost"], 6),
            "pct":    round(info["tokens"] / total_all_tokens * 100, 1) if total_all_tokens else 0,
        })
    breakdown.sort(key=lambda x: x["tokens"], reverse=True)
    return breakdown


@app.post("/calculate", response_class=HTMLResponse)
async def calculate(
    request: Request,
    file: UploadFile               = File(...),
    model_name: str                = Form(...),
    all_columns_config: str        = Form(default=None),   # {sheet: {col: {type,mult}}} multi-sheet
    columns_config: str            = Form(default=None),   # legacy single-sheet fallback
    pricing_override: str          = Form(default=None),
    provider: str                  = Form(default="openai"),
    budget: float                  = Form(default=0),
    sheet_name: str                = Form(default=None),   # legacy single-sheet fallback
):
    raw       = await file.read()
    file_md5  = hashlib.md5(raw).hexdigest()
    file_name = file.filename or "unknown"
    xl        = pd.ExcelFile(io.BytesIO(raw))

    pricing = json.loads(pricing_override) if pricing_override else load_pricing(provider)
    pricing.pop("last_updated", None)
    pricing.pop("last_updated_source", None)
    if model_name not in pricing:
        return HTMLResponse("<h2>Model not found.</h2>", status_code=400)

    mp               = pricing[model_name]
    input_price      = mp.get("input") or 0
    output_price     = mp.get("output") or 0
    cached_price     = mp.get("cached_input")
    caching_supported = cached_price is not None
    cached_price_eff = cached_price if cached_price is not None else input_price

    # ── Determine per-sheet column config ───────────────────────────────────
    if all_columns_config:
        # New multi-sheet mode: {sheetName: {col: {type, multiplier}}}
        sheet_cols_map = json.loads(all_columns_config)
        # Only process sheets that have at least one non-skip column
        active_sheets = {
            sname: cols for sname, cols in sheet_cols_map.items()
            if sname in xl.sheet_names
            and any(c.get("type", "skip") != "skip" for c in cols.values())
        }
    else:
        # Legacy single-sheet mode
        cols = json.loads(columns_config) if columns_config else {}
        target = sheet_name if sheet_name and sheet_name in xl.sheet_names else xl.sheet_names[0]
        active_sheets = {target: cols}

    if not active_sheets:
        return HTMLResponse("<h2>No columns selected. Please mark at least one column as input or output.</h2>", status_code=400)

    multi_sheet = len(active_sheets) > 1

    # ── Process each sheet ──────────────────────────────────────────────────
    grand_ti = grand_to = grand_tc = 0
    grand_tic = grand_toc = grand_tcc = 0
    all_rows            = []   # flat list (legacy compat for single-sheet)
    sheet_results       = []   # per-sheet breakdown for results page
    grand_col_totals    = {}   # merged across all sheets

    has_cached_cols = False

    for sname, cols in active_sheets.items():
        if any(c.get("type") == "cached_input" for c in cols.values()):
            has_cached_cols = True

        df = xl.parse(sname)
        rows, col_totals, ti, to, tc, tic, toc, tcc = _process_sheet(
            df, cols, model_name, input_price, output_price, cached_price_eff
        )

        total_sheet_tokens = ti + to + tc
        col_breakdown = _build_col_breakdown(col_totals, total_sheet_tokens)

        sheet_results.append({
            "sheet":              sname,
            "rows":               rows,
            "row_count":          len(rows),
            "input_tokens":       ti,
            "output_tokens":      to,
            "cached_tokens":      tc,
            "total_tokens":       total_sheet_tokens,
            "input_cost":         tic,
            "output_cost":        toc,
            "cached_cost":        tcc,
            "total_cost":         tic + toc + tcc,
            "col_breakdown":      col_breakdown,
        })

        all_rows.extend(rows)
        grand_ti  += ti;  grand_to  += to;  grand_tc  += tc
        grand_tic += tic; grand_toc += toc; grand_tcc += tcc

        # Merge col_totals into grand totals
        for col, info in col_totals.items():
            key = f"{sname}::{col}" if multi_sheet else col
            if key not in grand_col_totals:
                grand_col_totals[key] = {"type": info["type"], "tokens": 0, "cost": 0.0}
            grand_col_totals[key]["tokens"] += info["tokens"]
            grand_col_totals[key]["cost"]   += info["cost"]

    # ── Grand totals ────────────────────────────────────────────────────────
    grand_total_tokens = grand_ti + grand_to + grand_tc
    grand_col_breakdown = _build_col_breakdown(grand_col_totals, grand_total_tokens)

    # ── Model comparison (on grand totals) ──────────────────────────────────
    mc = []
    for model, price in pricing.items():
        if not isinstance(price, dict):
            continue
        mi  = price.get("input") or 0
        mo  = price.get("output") or 0
        mcp = price.get("cached_input")
        mce = mcp if mcp is not None else mi
        mc.append({
            "model":            model,
            "input_cost":       (grand_ti / 1_000_000) * mi,
            "output_cost":      (grand_to / 1_000_000) * mo,
            "cached_cost":      (grand_tc / 1_000_000) * mce,
            "total_cost":       (grand_ti / 1_000_000) * mi + (grand_to / 1_000_000) * mo + (grand_tc / 1_000_000) * mce,
            "caching_supported": mcp is not None,
            "pinned":           bool(price.get("pinned", False)),
            "litellm_tracked":  price.get("litellm_tracked", None),
        })
    mc.sort(key=lambda x: (0 if x.get("pinned") else 1, x["total_cost"]))

    return templates.TemplateResponse("summary.html", {
        "request":             request,
        "selected_model":      model_name,
        "rows":                all_rows,
        "total_input_tokens":  grand_ti,
        "total_output_tokens": grand_to,
        "total_cached_tokens": grand_tc,
        "dataset_total_tokens": grand_total_tokens,
        "total_input_cost":    grand_tic,
        "total_output_cost":   grand_toc,
        "total_cached_cost":   grand_tcc,
        "dataset_total_cost":  grand_tic + grand_toc + grand_tcc,
        "model_comparison":    mc,
        "has_cached_cols":     has_cached_cols,
        "caching_supported":   caching_supported,
        "file_name":           file_name,
        "file_md5":            file_md5,
        "budget":              budget,
        "col_breakdown":       grand_col_breakdown,
        "sheet_results":       sheet_results,   # NEW — per-sheet breakdown
        "multi_sheet":         multi_sheet,     # NEW — True when >1 sheet selected
    })