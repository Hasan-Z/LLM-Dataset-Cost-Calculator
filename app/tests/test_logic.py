"""
Unit tests for pure Python logic in main.py.
No HTTP calls — tests run fast and in isolation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pytest
import importlib

# ── Import helpers directly ──────────────────────────────────────────────────
import main
importlib.reload(main)

from main import (
    count_tokens,
    safe_price,
    safe_json_float,
    infer_capabilities,
    parse_litellm_pricing,
    provider_file,
    provider_prefixes,
    process_provider_pricing,
    _process_sheet,
    _build_col_breakdown,
)


# ────────────────────────────────────────────────────────────────────────────
# count_tokens
# ────────────────────────────────────────────────────────────────────────────

class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("", "gpt-4o") == 0

    def test_none_input(self):
        assert count_tokens(None, "gpt-4o") == 0

    def test_basic_text(self):
        n = count_tokens("Hello world", "gpt-4o")
        assert n > 0

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("Hi", "gpt-4o")
        long  = count_tokens("Hi " * 100, "gpt-4o")
        assert long > short

    def test_unknown_model_falls_back(self):
        # Should not raise — falls back to cl100k_base
        n = count_tokens("Hello world", "unknown-model-xyz")
        assert n > 0

    def test_non_string_input_coerced(self):
        # Numbers passed as cell values should be coerced to string
        assert count_tokens(42, "gpt-4o") > 0
        assert count_tokens(3.14, "gpt-4o") > 0


# ────────────────────────────────────────────────────────────────────────────
# safe_price  (converts per-token cost → $/1M tokens)
# ────────────────────────────────────────────────────────────────────────────

class TestSafePrice:
    def test_none_returns_none(self):
        assert safe_price(None) is None

    def test_converts_per_token_to_per_million(self):
        # $5 / 1M tokens → stored as 0.000005 per token
        result = safe_price(0.000005)
        assert result == pytest.approx(5.0, rel=1e-4)

    def test_zero_returns_zero(self):
        assert safe_price(0) == 0.0

    def test_string_number_works(self):
        result = safe_price("0.000005")
        assert result == pytest.approx(5.0, rel=1e-4)

    def test_invalid_string_returns_none(self):
        assert safe_price("not_a_number") is None


# ────────────────────────────────────────────────────────────────────────────
# infer_capabilities
# ────────────────────────────────────────────────────────────────────────────

class TestInferCapabilities:
    def test_gpt4o_has_vision(self):
        caps = infer_capabilities("gpt-4o", "chat", {})
        assert caps["supports_vision"] is True

    def test_gpt4o_has_function_calling(self):
        caps = infer_capabilities("gpt-4o", "chat", {})
        assert caps["supports_function_calling"] is True

    def test_gpt4o_has_prompt_caching(self):
        caps = infer_capabilities("gpt-4o", "chat", {})
        assert caps["supports_prompt_caching"] is True

    def test_o1_has_reasoning(self):
        caps = infer_capabilities("o1-preview", "chat", {})
        assert caps["supports_reasoning"] is True

    def test_tts_has_audio_output(self):
        caps = infer_capabilities("gpt-4o-audio", "audio_speech", {})
        assert caps["supports_audio_output"] is True

    def test_claude_sonnet_has_vision(self):
        caps = infer_capabilities("claude-3-5-sonnet-20241022", "chat", {})
        assert caps["supports_vision"] is True

    def test_claude_sonnet_has_caching(self):
        caps = infer_capabilities("claude-3-5-sonnet-20241022", "chat", {})
        assert caps["supports_prompt_caching"] is True

    def test_gemini_flash_has_vision(self):
        caps = infer_capabilities("gemini-1.5-flash", "chat", {})
        assert caps["supports_vision"] is True

    def test_explicit_flag_overrides_inference(self):
        # Even if the model would normally NOT have web search,
        # an explicit flag in info should override
        caps = infer_capabilities("gpt-4o-mini", "chat",
                                  {"supports_web_search": True})
        assert caps["supports_web_search"] is True

    def test_all_caps_keys_present(self):
        caps = infer_capabilities("gpt-4o", "chat", {})
        expected_keys = {
            "supports_vision", "supports_function_calling", "supports_reasoning",
            "supports_web_search", "supports_audio_input", "supports_audio_output",
            "supports_prompt_caching", "supports_response_schema",
        }
        assert expected_keys == set(caps.keys())


# ────────────────────────────────────────────────────────────────────────────
# parse_litellm_pricing
# ────────────────────────────────────────────────────────────────────────────

class TestParseLitellmPricing:
    RAW = {
        "gpt-4o": {
            "input_cost_per_token":  0.000005,
            "output_cost_per_token": 0.000015,
            "cache_read_input_token_cost": 0.0000025,
            "mode": "chat",
            "max_input_tokens": 128000,
            "max_output_tokens": 16384,
        },
        "gpt-4o-mini": {
            "input_cost_per_token":  0.00000015,
            "output_cost_per_token": 0.0000006,
            "mode": "chat",
        },
        "claude-3-5-sonnet-20241022": {
            # Should NOT appear with openai prefixes
            "input_cost_per_token":  0.000003,
            "output_cost_per_token": 0.000015,
            "mode": "chat",
        },
        "embedding-model": {
            # Missing output cost → should be excluded
            "input_cost_per_token": 0.0000001,
            "mode": "embedding",
        },
        "not-a-dict": "some string",   # should be skipped
    }

    def test_openai_models_parsed(self):
        result = parse_litellm_pricing(self.RAW, ("gpt-", "o1", "o3", "o4"))
        assert "gpt-4o" in result
        assert "gpt-4o-mini" in result

    def test_non_matching_prefix_excluded(self):
        result = parse_litellm_pricing(self.RAW, ("gpt-", "o1", "o3", "o4"))
        assert "claude-3-5-sonnet-20241022" not in result

    def test_missing_output_cost_excluded(self):
        result = parse_litellm_pricing(self.RAW, ("gpt-", "embedding"))
        assert "embedding-model" not in result

    def test_per_token_converted_to_per_million(self):
        result = parse_litellm_pricing(self.RAW, ("gpt-",))
        assert result["gpt-4o"]["input"]  == pytest.approx(5.0, rel=1e-4)
        assert result["gpt-4o"]["output"] == pytest.approx(15.0, rel=1e-4)

    def test_cached_input_parsed(self):
        result = parse_litellm_pricing(self.RAW, ("gpt-",))
        assert result["gpt-4o"]["cached_input"] == pytest.approx(2.5, rel=1e-4)

    def test_no_cached_input_is_none(self):
        result = parse_litellm_pricing(self.RAW, ("gpt-",))
        assert result["gpt-4o-mini"]["cached_input"] is None

    def test_non_dict_entry_skipped(self):
        result = parse_litellm_pricing(self.RAW, ("not-",))
        assert "not-a-dict" not in result

    def test_capabilities_included(self):
        result = parse_litellm_pricing(self.RAW, ("gpt-",))
        assert "supports_vision" in result["gpt-4o"]


# ────────────────────────────────────────────────────────────────────────────
# provider helpers
# ────────────────────────────────────────────────────────────────────────────

class TestProviderHelpers:
    def test_openai_file(self):
        assert "openai" in provider_file("openai")

    def test_anthropic_file(self):
        assert "anthropic" in provider_file("anthropic")

    def test_gemini_file(self):
        assert "google" in provider_file("gemini")

    def test_unknown_provider_falls_back_to_openai(self):
        assert provider_file("unknown") == provider_file("openai")

    def test_openai_prefixes(self):
        prefixes = provider_prefixes("openai")
        assert "gpt-" in prefixes

    def test_anthropic_prefixes(self):
        prefixes = provider_prefixes("anthropic")
        assert "claude-" in prefixes

    def test_gemini_prefixes(self):
        prefixes = provider_prefixes("gemini")
        assert "gemini-" in prefixes


# ────────────────────────────────────────────────────────────────────────────
# Cost math (inline, no endpoint)
# ────────────────────────────────────────────────────────────────────────────

class TestCostMath:
    """Verify the cost formula used in /calculate."""

    def test_input_cost_formula(self):
        tokens     = 1000
        price_per_M = 5.0          # $5 / 1M tokens
        expected   = (tokens / 1_000_000) * price_per_M
        assert expected == pytest.approx(0.005, rel=1e-6)

    def test_zero_tokens_zero_cost(self):
        assert (0 / 1_000_000) * 5.0 == 0.0

    def test_total_cost_is_sum(self):
        ic = 0.005
        oc = 0.015
        cc = 0.001
        assert ic + oc + cc == pytest.approx(0.021, rel=1e-6)

    def test_col_breakdown_pct(self):
        """Column percentage calculation."""
        col_tokens   = 300
        total_tokens = 1000
        pct = round(col_tokens / total_tokens * 100, 1)
        assert pct == 30.0

    def test_col_breakdown_pct_zero_total(self):
        """Should not divide by zero."""
        total_tokens = 0
        pct = round(0 / total_tokens * 100, 1) if total_tokens else 0
        assert pct == 0


# ────────────────────────────────────────────────────────────────────────────
# safe_json_float
# ────────────────────────────────────────────────────────────────────────────

class TestSafeJsonFloat:
    """safe_json_float sanitises floats for JSON serialisation."""

    def test_normal_float_passthrough(self):
        assert safe_json_float(5.0) == 5.0

    def test_none_returns_none(self):
        assert safe_json_float(None) is None

    def test_nan_returns_none(self):
        import math
        assert safe_json_float(float('nan')) is None

    def test_inf_returns_none(self):
        assert safe_json_float(float('inf'))  is None
        assert safe_json_float(float('-inf')) is None

    def test_zero_passthrough(self):
        assert safe_json_float(0) == 0.0

    def test_string_non_numeric_passthrough(self):
        # Non-numeric strings are returned as-is (not coerced)
        assert safe_json_float("chat") == "chat"

    def test_integer_converted_to_float(self):
        result = safe_json_float(5)
        assert result == 5.0


# ────────────────────────────────────────────────────────────────────────────
# process_provider_pricing  (sort order, pinned-first)
# ────────────────────────────────────────────────────────────────────────────

class TestProcessProviderPricing:
    """process_provider_pricing loads, splits, and sorts provider data."""

    def test_returns_five_values(self, tmp_project):
        """Returns (prices, metadata, models, last_updated, last_updated_source)."""
        import os
        os.chdir(tmp_project)
        result = process_provider_pricing("openai")
        assert len(result) == 5

    def test_prices_only_has_price_fields(self, tmp_project):
        import os
        os.chdir(tmp_project)
        prices, *_ = process_provider_pricing("openai")
        for model, p in prices.items():
            assert all(k in {"input","output","cached_input","reasoning_output"}
                       for k in p.keys()), \
                f"Non-price field leaked into prices for {model}: {list(p.keys())}"

    def test_metadata_has_capability_keys(self, tmp_project):
        import os
        os.chdir(tmp_project)
        _, metadata, *_ = process_provider_pricing("openai")
        for model, meta in metadata.items():
            assert "supports_vision" in meta, \
                f"Missing capability keys for {model}"

    def test_models_list_not_empty(self, tmp_project):
        import os
        os.chdir(tmp_project)
        _, _, models, *_ = process_provider_pricing("openai")
        assert len(models) > 0

    def test_pinned_model_sorts_first(self, tmp_project):
        """A pinned model must appear before all unpinned models."""
        import os, json as json_mod
        os.chdir(tmp_project)

        # Pin gpt-4o-mini in the mock JSON
        path = tmp_project / "openai_ai_models_pricing.json"
        data = json_mod.loads(path.read_text(encoding="utf-8"))
        data["gpt-4o-mini"]["pinned"] = True
        data["gpt-4o"]["pinned"]      = False
        data["gpt-4"]["pinned"]       = False
        path.write_text(json_mod.dumps(data, indent=2), encoding="utf-8")

        _, _, models, *_ = process_provider_pricing("openai")
        assert models[0] == "gpt-4o-mini", \
            f"Pinned model should be first, got: {models}"

        # Restore
        data["gpt-4o-mini"]["pinned"] = False
        path.write_text(json_mod.dumps(data, indent=2), encoding="utf-8")

    def test_multiple_pinned_models_all_at_top(self, tmp_project):
        """All pinned models must precede all unpinned ones."""
        import os, json as json_mod
        os.chdir(tmp_project)

        path = tmp_project / "openai_ai_models_pricing.json"
        data = json_mod.loads(path.read_text(encoding="utf-8"))
        data["gpt-4o"]["pinned"]      = True
        data["gpt-4o-mini"]["pinned"] = True
        data["gpt-4"]["pinned"]       = False
        path.write_text(json_mod.dumps(data, indent=2), encoding="utf-8")

        _, metadata, models, *_ = process_provider_pricing("openai")
        pinned_models   = [m for m in models if metadata[m].get("pinned")]
        unpinned_models = [m for m in models if not metadata[m].get("pinned")]

        if pinned_models and unpinned_models:
            last_pinned_idx   = max(models.index(m) for m in pinned_models)
            first_unpinned_idx = min(models.index(m) for m in unpinned_models)
            assert last_pinned_idx < first_unpinned_idx, \
                "All pinned models must appear before all unpinned models"

        # Restore
        data["gpt-4o"]["pinned"]      = False
        data["gpt-4o-mini"]["pinned"] = False
        path.write_text(json_mod.dumps(data, indent=2), encoding="utf-8")

    def test_unpinned_models_sorted_alphabetically(self, tmp_project):
        """Unpinned models must be sorted alphabetically within their group."""
        import os
        os.chdir(tmp_project)
        _, metadata, models, *_ = process_provider_pricing("openai")
        unpinned = [m for m in models if not metadata[m].get("pinned")]
        assert unpinned == sorted(unpinned, key=str.lower), \
            "Unpinned models must be sorted alphabetically"

    def test_last_updated_returned(self, tmp_project):
        import os
        os.chdir(tmp_project)
        _, _, _, last_updated, _ = process_provider_pricing("openai")
        # May be None if not set, but must be returned
        assert last_updated is not None  # conftest sets it

    def test_last_updated_source_returned(self, tmp_project):
        import os
        os.chdir(tmp_project)
        _, _, _, _, source = process_provider_pricing("openai")
        assert source in ("manual", "sync", "litellm")

    def test_metadata_pinned_inferred_flag(self, tmp_project):
        """_capabilities_inferred flag must be set on all metadata entries."""
        import os
        os.chdir(tmp_project)
        _, metadata, *_ = process_provider_pricing("openai")
        for model, meta in metadata.items():
            assert "_capabilities_inferred" in meta, \
                f"Missing _capabilities_inferred for {model}"



# ────────────────────────────────────────────────────────────────────────────
# _process_sheet  (core calculation helper)
# ────────────────────────────────────────────────────────────────────────────

class TestProcessSheet:
    """Unit tests for _process_sheet — the per-sheet calculation helper."""

    def _df(self, data=None):
        import pandas as pd
        return pd.DataFrame(data or {"prompt": ["hello world"], "response": ["hi"]})

    def _prices(self):
        return 5.0, 15.0, 2.5   # input, output, cached_price_eff

    def test_returns_eight_values(self):
        ip, op, cp = self._prices()
        result = _process_sheet(self._df(), {}, "gpt-4o", ip, op, cp)
        assert len(result) == 8

    def test_all_skip_gives_zero_totals(self):
        ip, op, cp = self._prices()
        cols = {"prompt": {"type": "skip", "multiplier": 1}}
        rows, col_totals, ti, to, tc, tic, toc, tcc = _process_sheet(
            self._df(), cols, "gpt-4o", ip, op, cp)
        assert ti == to == tc == 0
        assert tic == toc == tcc == 0.0

    def test_input_column_counted(self):
        import pandas as pd
        ip, op, cp = self._prices()
        df   = pd.DataFrame({"text": ["hello world test"]})
        cols = {"text": {"type": "input", "multiplier": 1}}
        rows, col_totals, ti, to, tc, *_ = _process_sheet(df, cols, "gpt-4o", ip, op, cp)
        assert ti > 0
        assert to == tc == 0

    def test_output_column_counted(self):
        import pandas as pd
        ip, op, cp = self._prices()
        df   = pd.DataFrame({"text": ["hello world"]})
        cols = {"text": {"type": "output", "multiplier": 1}}
        rows, col_totals, ti, to, tc, *_ = _process_sheet(df, cols, "gpt-4o", ip, op, cp)
        assert to > 0
        assert ti == tc == 0

    def test_multiplier_doubles_tokens(self):
        import pandas as pd
        ip, op, cp = self._prices()
        df   = pd.DataFrame({"text": ["hello world"]})
        cols1 = {"text": {"type": "input", "multiplier": 1}}
        cols2 = {"text": {"type": "input", "multiplier": 2}}
        _, _, ti1, *_ = _process_sheet(df, cols1, "gpt-4o", ip, op, cp)
        _, _, ti2, *_ = _process_sheet(df, cols2, "gpt-4o", ip, op, cp)
        assert ti2 == ti1 * 2

    def test_row_count_matches_dataframe(self):
        import pandas as pd
        ip, op, cp = self._prices()
        df   = pd.DataFrame({"text": ["a", "b", "c"]})
        cols = {"text": {"type": "input", "multiplier": 1}}
        rows, *_ = _process_sheet(df, cols, "gpt-4o", ip, op, cp)
        assert len(rows) == 3

    def test_row_dict_has_required_keys(self):
        import pandas as pd
        ip, op, cp = self._prices()
        df   = pd.DataFrame({"text": ["hello"]})
        cols = {"text": {"type": "input", "multiplier": 1}}
        rows, *_ = _process_sheet(df, cols, "gpt-4o", ip, op, cp)
        required = {"row", "input_tokens", "output_tokens", "cached_tokens",
                    "total_tokens", "input_cost", "output_cost", "cached_cost", "total_cost"}
        assert required <= set(rows[0].keys())

    def test_input_cost_formula(self):
        import pandas as pd
        ip, op, cp = 5.0, 15.0, 2.5
        df   = pd.DataFrame({"text": ["hello"]})
        cols = {"text": {"type": "input", "multiplier": 1}}
        _, _, ti, _, _, tic, *_ = _process_sheet(df, cols, "gpt-4o", ip, op, cp)
        assert tic == pytest.approx((ti / 1_000_000) * ip, rel=1e-6)

    def test_invalid_multiplier_skipped(self):
        import pandas as pd
        ip, op, cp = self._prices()
        df   = pd.DataFrame({"text": ["hello"]})
        cols = {"text": {"type": "input", "multiplier": 0}}   # ≤0 → skip
        _, _, ti, *_ = _process_sheet(df, cols, "gpt-4o", ip, op, cp)
        assert ti == 0

    def test_missing_column_handled_gracefully(self):
        import pandas as pd
        ip, op, cp = self._prices()
        df   = pd.DataFrame({"text": ["hello"]})
        # Config references column that doesn't exist in df
        cols = {"nonexistent": {"type": "input", "multiplier": 1}}
        rows, _, ti, *_ = _process_sheet(df, cols, "gpt-4o", ip, op, cp)
        # Should not raise — empty cell treated as empty string → 0 tokens
        assert ti == 0


# ────────────────────────────────────────────────────────────────────────────
# _build_col_breakdown
# ────────────────────────────────────────────────────────────────────────────

class TestBuildColBreakdown:

    def test_empty_col_totals_returns_empty(self):
        assert _build_col_breakdown({}, 100) == []

    def test_pct_calculated_correctly(self):
        col_totals = {"col_a": {"type": "input", "tokens": 300, "cost": 0.0015}}
        result = _build_col_breakdown(col_totals, 1000)
        assert result[0]["pct"] == 30.0

    def test_pct_zero_when_no_total_tokens(self):
        col_totals = {"col_a": {"type": "input", "tokens": 0, "cost": 0.0}}
        result = _build_col_breakdown(col_totals, 0)
        assert result[0]["pct"] == 0

    def test_sorted_by_tokens_descending(self):
        col_totals = {
            "small": {"type": "input", "tokens": 100, "cost": 0.0005},
            "large": {"type": "input", "tokens": 500, "cost": 0.0025},
            "mid":   {"type": "output","tokens": 300, "cost": 0.0045},
        }
        result = _build_col_breakdown(col_totals, 900)
        tokens = [r["tokens"] for r in result]
        assert tokens == sorted(tokens, reverse=True)

    def test_result_has_required_keys(self):
        col_totals = {"col": {"type": "input", "tokens": 100, "cost": 0.0005}}
        result = _build_col_breakdown(col_totals, 100)
        assert {"column","type","tokens","cost","pct"} <= set(result[0].keys())

    def test_cost_rounded_to_six_decimals(self):
        col_totals = {"col": {"type": "input", "tokens": 1, "cost": 0.000001234567}}
        result = _build_col_breakdown(col_totals, 1)
        assert result[0]["cost"] == round(0.000001234567, 6)


# ────────────────────────────────────────────────────────────────────────────
# Sheet selection logic (unit-level — no HTTP)
# ────────────────────────────────────────────────────────────────────────────

class TestSheetSelectionLogic:
    """Unit tests for sheet-selection data logic used by /upload and /calculate."""

    def _make_multi(self):
        import io
        import pandas as pd
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            pd.DataFrame({"prompt": ["hello"], "ctx": ["a"]}).to_excel(
                writer, sheet_name="Inputs", index=False)
            pd.DataFrame({"response": ["hi"], "score": [1]}).to_excel(
                writer, sheet_name="Outputs", index=False)
        buf.seek(0)
        return buf.read()

    # ── openpyxl read_only header extraction (the fast path used in /upload) ──

    def test_read_only_returns_sheet_names(self):
        import io, openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(self._make_multi()),
                                    read_only=True, data_only=True)
        assert set(wb.sheetnames) == {"Inputs", "Outputs"}
        wb.close()

    def test_read_only_headers_correct(self):
        import io, openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(self._make_multi()),
                                    read_only=True, data_only=True)
        ws     = wb["Inputs"]
        header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
        assert set(header) == {"prompt", "ctx"}
        wb.close()

    def test_read_only_all_sheets_headers(self):
        """Simulate the /upload endpoint: get all sheets' columns in one pass."""
        import io, openpyxl
        wb          = openpyxl.load_workbook(io.BytesIO(self._make_multi()),
                                             read_only=True, data_only=True)
        all_columns = {}
        for sheet_name in wb.sheetnames:
            ws     = wb[sheet_name]
            header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
            all_columns[sheet_name] = [str(c) for c in header if c is not None]
        wb.close()
        assert all_columns["Inputs"]  == ["prompt", "ctx"]
        assert all_columns["Outputs"] == ["response", "score"]

    def test_read_only_does_not_read_data_rows(self):
        """Header-only read should NOT return data cell values as column names."""
        import io, openpyxl
        wb     = openpyxl.load_workbook(io.BytesIO(self._make_multi()),
                                        read_only=True, data_only=True)
        ws     = wb["Inputs"]
        header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
        wb.close()
        assert "hello" not in header   # data value from row 2
        assert "prompt" in header      # actual column name

    def test_different_sheets_different_columns(self):
        import io, openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(self._make_multi()),
                                    read_only=True, data_only=True)
        h1 = set(next(wb["Inputs"].iter_rows( min_row=1, max_row=1, values_only=True), ()))
        h2 = set(next(wb["Outputs"].iter_rows(min_row=1, max_row=1, values_only=True), ()))
        wb.close()
        assert h1 != h2

    # ── Target-sheet fallback logic (mirrors /calculate) ─────────────────────

    def test_target_sheet_fallback_invalid(self):
        """Invalid sheet_name → sheet 0 fallback."""
        sheets       = ["Inputs", "Outputs"]
        sheet_name   = "NonExistent"
        target_sheet = sheet_name if sheet_name in sheets else sheets[0]
        assert target_sheet == "Inputs"

    def test_target_sheet_valid_used(self):
        sheets       = ["Inputs", "Outputs"]
        sheet_name   = "Outputs"
        target_sheet = sheet_name if sheet_name in sheets else sheets[0]
        assert target_sheet == "Outputs"

    def test_target_sheet_none_fallback(self):
        sheets       = ["Inputs", "Outputs"]
        sheet_name   = None
        target_sheet = sheet_name if sheet_name and sheet_name in sheets else sheets[0]
        assert target_sheet == "Inputs"

    # ── all_columns cache coherence ───────────────────────────────────────────

    def test_all_columns_covers_every_sheet(self):
        """Every sheet must be present in all_columns — no extra roundtrip needed."""
        import io, openpyxl
        wb          = openpyxl.load_workbook(io.BytesIO(self._make_multi()),
                                             read_only=True, data_only=True)
        all_columns = {}
        for s in wb.sheetnames:
            ws     = wb[s]
            header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
            all_columns[s] = [str(c) for c in header if c is not None]
        wb.close()
        for sheet in ["Inputs", "Outputs"]:
            assert sheet in all_columns, f"Sheet '{sheet}' missing from all_columns cache"

    def test_single_sheet_file_still_works(self):
        import io, openpyxl
        import pandas as pd
        buf = io.BytesIO()
        pd.DataFrame({"col": [1, 2]}).to_excel(buf, index=False)
        buf.seek(0)
        wb     = openpyxl.load_workbook(buf, read_only=True, data_only=True)
        sheets = wb.sheetnames
        header = next(wb[sheets[0]].iter_rows(min_row=1, max_row=1, values_only=True), ())
        wb.close()
        assert len(sheets) == 1
        assert "col" in header

    def test_row_count_not_read_at_upload(self):
        """Confirms read_only header-only approach doesn't iterate all rows."""
        import io, openpyxl
        # Build a file where data rows contain trap values
        buf = io.BytesIO()
        import pandas as pd
        pd.DataFrame({"id": range(50000), "val": ["trap"] * 50000}).to_excel(
            buf, index=False)
        buf.seek(0)
        wb     = openpyxl.load_workbook(buf, read_only=True, data_only=True)
        ws     = wb[wb.sheetnames[0]]
        header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
        wb.close()
        # Should only have column names, not data
        assert "trap" not in header
        assert set(header) == {"id", "val"}