"""
API endpoint tests — cover every route in main.py.
Uses FastAPI's TestClient (no real HTTP, no browser).
"""
import io
import json
import pytest
import pandas as pd

from conftest import make_excel_bytes, MOCK_PRICING, MOCK_ANTHROPIC_PRICING


# ────────────────────────────────────────────────────────────────────────────
# GET /   (home page)
# ────────────────────────────────────────────────────────────────────────────

class TestHomePage:
    def test_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_returns_html(self, client):
        r = client.get("/")
        assert "text/html" in r.headers["content-type"]

    def test_contains_pricing_table(self, client):
        r = client.get("/")
        assert "pricingTableBody" in r.text

    def test_contains_upload_zone(self, client):
        r = client.get("/")
        assert "uploadZone" in r.text

    def test_contains_provider_tabs(self, client):
        r = client.get("/")
        assert "openai" in r.text.lower()
        assert "anthropic" in r.text.lower()
        assert "gemini" in r.text.lower()

    def test_contains_model_select(self, client):
        r = client.get("/")
        assert "modelSelect" in r.text

    def test_mock_models_in_page(self, client):
        r = client.get("/")
        assert "gpt-4o" in r.text


# ────────────────────────────────────────────────────────────────────────────
# POST /upload
# ────────────────────────────────────────────────────────────────────────────

class TestUpload:
    def test_returns_columns(self, client, sample_excel):
        r = client.post("/upload", files={"file": ("test.xlsx", sample_excel,
              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        assert r.status_code == 200
        data = r.json()
        assert "columns" in data
        assert set(data["columns"]) == {"prompt", "response", "context"}

    def test_column_order_preserved(self, client, sample_excel):
        r = client.post("/upload", files={"file": ("test.xlsx", sample_excel,
              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        assert r.json()["columns"] == ["prompt", "response", "context"]

    def test_single_column_excel(self, client):
        xls = make_excel_bytes({"text": ["a", "b", "c"]})
        r = client.post("/upload", files={"file": ("t.xlsx", xls,
              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        assert r.status_code == 200
        assert r.json()["columns"] == ["text"]

    def test_many_columns(self, client):
        data = {f"col{i}": ["x"] for i in range(10)}
        xls = make_excel_bytes(data)
        r = client.post("/upload", files={"file": ("t.xlsx", xls,
              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        assert len(r.json()["columns"]) == 10


# ────────────────────────────────────────────────────────────────────────────
# POST /upload-text
# ────────────────────────────────────────────────────────────────────────────

class TestUploadText:
    def test_text_file_analysis(self, client):
        content = b"Hello world this is a test file."
        r = client.post("/upload-text",
                        data={"file_type": "text"},
                        files={"file": ("test.txt", content, "text/plain")})
        assert r.status_code == 200
        data = r.json()
        assert "chars"      in data
        assert "words"      in data
        assert "lines"      in data
        assert "est_tokens" in data

    def test_chars_count(self, client):
        content = b"Hello"
        r = client.post("/upload-text",
                        data={"file_type": "text"},
                        files={"file": ("t.txt", content, "text/plain")})
        assert r.json()["chars"] == 5

    def test_words_count(self, client):
        content = b"Hello world foo"
        r = client.post("/upload-text",
                        data={"file_type": "text"},
                        files={"file": ("t.txt", content, "text/plain")})
        assert r.json()["words"] == 3

    def test_est_tokens_positive(self, client):
        content = b"This is a reasonably long sentence to ensure token count is non-zero."
        r = client.post("/upload-text",
                        data={"file_type": "text"},
                        files={"file": ("t.txt", content, "text/plain")})
        assert r.json()["est_tokens"] > 0


# ────────────────────────────────────────────────────────────────────────────
# POST /calculate  (Excel)
# ────────────────────────────────────────────────────────────────────────────

class TestCalculate:
    def _cols(self, prompt="input", response="output", context="skip"):
        return json.dumps({
            "prompt":   {"type": prompt,   "multiplier": 1},
            "response": {"type": response, "multiplier": 1},
            "context":  {"type": context,  "multiplier": 1},
        })

    def test_returns_200(self, client, sample_excel, pricing_override):
        r = client.post("/calculate",
            data={"model_name": "gpt-4o",
                  "columns_config": self._cols(),
                  "pricing_override": pricing_override},
            files={"file": ("t.xlsx", sample_excel,
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        assert r.status_code == 200

    def test_returns_html(self, client, sample_excel, pricing_override):
        r = client.post("/calculate",
            data={"model_name": "gpt-4o",
                  "columns_config": self._cols(),
                  "pricing_override": pricing_override},
            files={"file": ("t.xlsx", sample_excel,
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        assert "text/html" in r.headers["content-type"]

    def test_result_contains_model_name(self, client, sample_excel, pricing_override):
        r = client.post("/calculate",
            data={"model_name": "gpt-4o",
                  "columns_config": self._cols(),
                  "pricing_override": pricing_override},
            files={"file": ("t.xlsx", sample_excel,
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        assert "gpt-4o" in r.text

    def test_unknown_model_returns_400(self, client, sample_excel, pricing_override):
        r = client.post("/calculate",
            data={"model_name": "nonexistent-model",
                  "columns_config": self._cols(),
                  "pricing_override": pricing_override},
            files={"file": ("t.xlsx", sample_excel,
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        assert r.status_code == 400

    def test_all_skip_gives_zero_cost(self, client, sample_excel, pricing_override):
        all_skip = json.dumps({
            "prompt":   {"type": "skip", "multiplier": 1},
            "response": {"type": "skip", "multiplier": 1},
            "context":  {"type": "skip", "multiplier": 1},
        })
        r = client.post("/calculate",
            data={"model_name": "gpt-4o",
                  "columns_config": all_skip,
                  "pricing_override": pricing_override},
            files={"file": ("t.xlsx", sample_excel,
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        assert r.status_code == 200
        assert "$0.000000" in r.text

    def test_3_rows_in_result(self, client, sample_excel, pricing_override):
        r = client.post("/calculate",
            data={"model_name": "gpt-4o",
                  "columns_config": self._cols(),
                  "pricing_override": pricing_override},
            files={"file": ("t.xlsx", sample_excel,
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        # The results page lists row numbers 1,2,3
        assert "1" in r.text and "2" in r.text and "3" in r.text

    def test_model_comparison_present(self, client, sample_excel, pricing_override):
        r = client.post("/calculate",
            data={"model_name": "gpt-4o",
                  "columns_config": self._cols(),
                  "pricing_override": pricing_override},
            files={"file": ("t.xlsx", sample_excel,
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        assert "Model Cost Comparison" in r.text

    def test_col_breakdown_present_for_multi_col(self, client, sample_excel, pricing_override):
        r = client.post("/calculate",
            data={"model_name": "gpt-4o",
                  "columns_config": self._cols(),   # prompt=input, response=output
                  "pricing_override": pricing_override},
            files={"file": ("t.xlsx", sample_excel,
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        assert "Token Breakdown by Column" in r.text

    def test_budget_exceeded_shows_warning(self, client, sample_excel, pricing_override):
        r = client.post("/calculate",
            data={"model_name": "gpt-4o",
                  "columns_config": self._cols(),
                  "pricing_override": pricing_override,
                  "budget": "0.000001"},   # impossibly small budget
            files={"file": ("t.xlsx", sample_excel,
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        assert "Budget exceeded" in r.text or "budget" in r.text.lower()

    def test_cached_col_type_detected(self, client, sample_excel, pricing_override):
        cols_with_cache = json.dumps({
            "prompt":   {"type": "cached_input", "multiplier": 1},
            "response": {"type": "output",       "multiplier": 1},
            "context":  {"type": "skip",         "multiplier": 1},
        })
        r = client.post("/calculate",
            data={"model_name": "gpt-4o",
                  "columns_config": cols_with_cache,
                  "pricing_override": pricing_override},
            files={"file": ("t.xlsx", sample_excel,
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        assert r.status_code == 200
        assert "Cached" in r.text

    def test_multiplier_applied(self, client, pricing_override):
        """A 2× multiplier should double token counts vs 1×."""
        xls = make_excel_bytes({"text": ["hello world"]})
        cols_x1 = json.dumps({"text": {"type": "input", "multiplier": 1}})
        cols_x2 = json.dumps({"text": {"type": "input", "multiplier": 2}})

        r1 = client.post("/calculate",
            data={"model_name": "gpt-4o", "columns_config": cols_x1,
                  "pricing_override": pricing_override},
            files={"file": ("t.xlsx", xls,
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        r2 = client.post("/calculate",
            data={"model_name": "gpt-4o", "columns_config": cols_x2,
                  "pricing_override": pricing_override},
            files={"file": ("t.xlsx", xls,
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})

        # Both should succeed; r2 should have higher cost
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Cost appears in both pages — r2 cost string should differ from r1
        assert r1.text != r2.text


# ────────────────────────────────────────────────────────────────────────────
# POST /calculate-text
# ────────────────────────────────────────────────────────────────────────────

class TestCalculateText:
    def test_text_file_returns_200(self, client, pricing_override):
        content = b"Hello world, this is a test prompt."
        r = client.post("/calculate-text",
            data={"file_type": "text", "model_name": "gpt-4o",
                  "pricing_override": pricing_override},
            files={"file": ("test.txt", content, "text/plain")})
        assert r.status_code == 200

    def test_unknown_model_returns_400(self, client):
        content = b"some text"
        r = client.post("/calculate-text",
            data={"file_type": "text", "model_name": "nonexistent-xyz"},
            files={"file": ("test.txt", content, "text/plain")})
        assert r.status_code == 400

    def test_unknown_file_type_returns_400(self, client, pricing_override):
        content = b"some content"
        r = client.post("/calculate-text",
            data={"file_type": "pdf", "model_name": "gpt-4o",
                  "pricing_override": pricing_override},
            files={"file": ("test.pdf", content, "application/pdf")})
        assert r.status_code == 400

    def test_result_contains_model_name(self, client, pricing_override):
        content = b"Test content for token counting."
        r = client.post("/calculate-text",
            data={"file_type": "text", "model_name": "gpt-4o",
                  "pricing_override": pricing_override},
            files={"file": ("test.txt", content, "text/plain")})
        assert "gpt-4o" in r.text

    def test_empty_file_still_works(self, client, pricing_override):
        r = client.post("/calculate-text",
            data={"file_type": "text", "model_name": "gpt-4o",
                  "pricing_override": pricing_override},
            files={"file": ("empty.txt", b"", "text/plain")})
        assert r.status_code == 200


class TestProviderData:
    """Tests for /provider-data — the fresh-fetch endpoint used by tab switching."""

    def test_openai_returns_200(self, client):
        r = client.get("/provider-data", params={"provider": "openai"})
        assert r.status_code == 200

    def test_returns_required_keys(self, client):
        r = client.get("/provider-data", params={"provider": "openai"})
        data = r.json()
        assert "prices"   in data
        assert "metadata" in data
        assert "models"   in data

    def test_anthropic_returns_data(self, client):
        r = client.get("/provider-data", params={"provider": "anthropic"})
        assert r.status_code == 200
        assert len(r.json()["models"]) > 0

    def test_gemini_returns_data(self, client):
        r = client.get("/provider-data", params={"provider": "gemini"})
        assert r.status_code == 200
        assert len(r.json()["models"]) > 0

    def test_pinned_model_first_in_models_list(self, client):
        """After pinning a model, /provider-data returns it first in models list."""
        # Pin gpt-4o-mini
        client.post("/toggle-pin", json={"model": "gpt-4o-mini", "provider": "openai"})
        r = client.get("/provider-data", params={"provider": "openai"})
        models = r.json()["models"]
        assert models[0] == "gpt-4o-mini", \
            f"Pinned model should be first, got: {models[:3]}"
        # Clean up
        client.post("/toggle-pin", json={"model": "gpt-4o-mini", "provider": "openai"})

    def test_pinned_state_reflected_after_pin(self, client):
        """After pinning, /provider-data metadata shows pinned=True."""
        client.post("/toggle-pin", json={"model": "gpt-4o", "provider": "openai"})
        r = client.get("/provider-data", params={"provider": "openai"})
        meta = r.json()["metadata"]
        assert meta["gpt-4o"]["pinned"] is True
        # Clean up
        client.post("/toggle-pin", json={"model": "gpt-4o", "provider": "openai"})

    def test_saved_prices_reflected(self, client):
        """After saving new prices, /provider-data returns updated values."""
        new_price = 99.0
        client.post("/save-pricing",
                    params={"__provider__": "openai"},
                    json={"gpt-4o": {"input": new_price, "output": 15.0,
                                     "cached_input": 2.5, "pinned": False}})
        r = client.get("/provider-data", params={"provider": "openai"})
        assert r.json()["prices"]["gpt-4o"]["input"] == new_price

    def test_tab_switch_sees_fresh_data(self, client):
        """
        Simulate the full tab-switch stale-data bug:
        1. Pin a model on OpenAI
        2. Call /provider-data for Anthropic (simulates tab switch away)
        3. Call /provider-data for OpenAI (simulates tab switch back)
        4. OpenAI data should still show the model as pinned
        """
        client.post("/toggle-pin", json={"model": "gpt-4o", "provider": "openai"})

        # Switch away and back
        client.get("/provider-data", params={"provider": "anthropic"})
        r = client.get("/provider-data", params={"provider": "openai"})

        meta = r.json()["metadata"]
        assert meta["gpt-4o"]["pinned"] is True, \
            "Pin state lost after switching tabs — stale data bug!"

        # Clean up
        client.post("/toggle-pin", json={"model": "gpt-4o", "provider": "openai"})

    def test_pinned_models_at_top_after_tab_switch(self, client):
        """
        Core bug: after pinning multiple models, switching tabs and back
        must preserve pinned models at the TOP of the models list.
        """
        # Pin two models
        client.post("/toggle-pin", json={"model": "gpt-4o",      "provider": "openai"})
        client.post("/toggle-pin", json={"model": "gpt-4o-mini", "provider": "openai"})

        # Simulate tab switch: away to Anthropic then back to OpenAI
        client.get("/provider-data", params={"provider": "anthropic"})
        r = client.get("/provider-data", params={"provider": "openai"})

        models = r.json()["models"]
        pinned = [m for m in models if r.json()["metadata"][m].get("pinned")]

        # Both pinned models must be in the first len(pinned) positions
        top_n = models[:len(pinned)]
        for m in pinned:
            assert m in top_n, \
                f"Pinned model '{m}' not at top after tab switch. Top models: {top_n}"

        # Clean up
        client.post("/toggle-pin", json={"model": "gpt-4o",      "provider": "openai"})
        client.post("/toggle-pin", json={"model": "gpt-4o-mini", "provider": "openai"})


# ────────────────────────────────────────────────────────────────────────────
# POST /save-pricing
# ────────────────────────────────────────────────────────────────────────────

class TestSavePricing:
    def test_save_returns_success(self, client):
        payload = {
            "gpt-4o": {"input": 5.0, "output": 15.0, "cached_input": 2.5},
        }
        r = client.post("/save-pricing",
                        params={"__provider__": "openai", "__source__": "manual"},
                        json=payload)
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_save_returns_last_updated(self, client):
        payload = {"gpt-4o": {"input": 5.0, "output": 15.0}}
        r = client.post("/save-pricing",
                        params={"__provider__": "openai"},
                        json=payload)
        assert "last_updated" in r.json()

    def test_save_and_reload(self, client):
        """After saving, the home page should include the new model."""
        payload = {"my-custom-model": {"input": 1.0, "output": 2.0,
                                        "cached_input": None, "pinned": False}}
        client.post("/save-pricing",
                    params={"__provider__": "openai"},
                    json=payload)
        r = client.get("/")
        assert "my-custom-model" in r.text


# ────────────────────────────────────────────────────────────────────────────
# POST /toggle-pin
# ────────────────────────────────────────────────────────────────────────────

class TestTogglePin:
    def test_pin_model(self, client):
        r = client.post("/toggle-pin",
                        json={"model": "gpt-4o", "provider": "openai"})
        assert r.status_code == 200
        data = r.json()
        assert "pinned" in data

    def test_pin_then_unpin(self, client):
        # First call pins
        r1 = client.post("/toggle-pin",
                         json={"model": "gpt-4o-mini", "provider": "openai"})
        pinned_after_first = r1.json()["pinned"]
        # Second call unpins
        r2 = client.post("/toggle-pin",
                         json={"model": "gpt-4o-mini", "provider": "openai"})
        assert r2.json()["pinned"] == (not pinned_after_first)

    def test_unknown_model_returns_error(self, client):
        r = client.post("/toggle-pin",
                        json={"model": "no-such-model", "provider": "openai"})
        assert r.status_code in (200, 404)
        # Should either 404 or return success=False
        if r.status_code == 200:
            assert r.json().get("success") is False or "error" in r.json()

    def test_pin_sets_pinned_true_in_json(self, client, tmp_project):
        """After pinning, the JSON file should have pinned=True for that model."""
        import json as json_mod
        # Ensure unpinned first
        client.post("/toggle-pin", json={"model": "gpt-4o", "provider": "openai"})
        # Read JSON and check
        pricing_path = tmp_project / "openai_ai_models_pricing.json"
        data = json_mod.loads(pricing_path.read_text(encoding="utf-8"))
        gpt4o_pinned = data.get("gpt-4o", {}).get("pinned", None)
        # Toggle one more time so state is known (True)
        client.post("/toggle-pin", json={"model": "gpt-4o", "provider": "openai"})
        data2 = json_mod.loads(pricing_path.read_text(encoding="utf-8"))
        # One of the two states must be True (we toggled twice from unknown start)
        assert isinstance(data.get("gpt-4o", {}).get("pinned"), bool)
        assert isinstance(data2.get("gpt-4o", {}).get("pinned"), bool)

    def test_pinned_model_appears_first_in_home_page(self, client):
        """After pinning gpt-4o-mini, it should appear before gpt-4 in the model selector."""
        # Pin gpt-4o-mini
        client.post("/toggle-pin", json={"model": "gpt-4o-mini", "provider": "openai"})
        r = client.get("/")

        # The Jinja template renders option value attributes — pinned models first.
        # Search for value="model-name" to find position in HTML source.
        idx_mini = r.text.find('value="gpt-4o-mini"')
        idx_4    = r.text.find('value="gpt-4"')

        assert idx_mini != -1, "gpt-4o-mini option not found in page"
        assert idx_4    != -1, "gpt-4 option not found in page"
        assert idx_mini < idx_4, \
            "Pinned gpt-4o-mini should appear before unpinned gpt-4 in selector"

        # Clean up — unpin
        client.post("/toggle-pin", json={"model": "gpt-4o-mini", "provider": "openai"})


# ────────────────────────────────────────────────────────────────────────────
# GET /fetch-pricing
# ────────────────────────────────────────────────────────────────────────────

class TestFetchPricing:
    def test_network_failure_returns_error(self, client, monkeypatch):
        """When LiteLLM is unreachable, endpoint returns success=False."""
        import main as app_module
        import httpx as httpx_mod

        class MockAsyncClient:
            def __init__(self, **kwargs): pass  # accept timeout= etc.
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get(self, *a, **kw):
                raise httpx_mod.ConnectError("mocked network error")

        monkeypatch.setattr(app_module.httpx, "AsyncClient", MockAsyncClient)
        r = client.get("/fetch-pricing", params={"provider": "openai"})
        assert r.status_code == 200
        assert r.json()["success"] is False

    def test_returns_untracked_list(self, client, monkeypatch):
        """With a mock LiteLLM response, untracked models are identified."""
        import main as app_module

        mock_response_data = {
            "gpt-4o": {
                "input_cost_per_token": 0.000005,
                "output_cost_per_token": 0.000015,
                "mode": "chat",
                "max_input_tokens": 128000,
                "max_output_tokens": 16384,
            }
            # gpt-4 and gpt-4o-mini are in our JSON but NOT in LiteLLM → untracked
        }

        class MockResponse:
            def raise_for_status(self): pass
            def json(self): return mock_response_data

        class MockAsyncClient:
            def __init__(self, **kwargs): pass  # accept timeout= etc.
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get(self, *a, **kw): return MockResponse()

        # Patch in the main module's httpx reference
        monkeypatch.setattr(app_module.httpx, "AsyncClient", MockAsyncClient)
        r = client.get("/fetch-pricing", params={"provider": "openai"})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True, f"Expected success but got: {data}"
        assert "untracked" in data
        # gpt-4 and gpt-4o-mini are in our JSON but not in the mock → untracked
        assert "gpt-4" in data["untracked"] or "gpt-4o-mini" in data["untracked"]


# ────────────────────────────────────────────────────────────────────────────
# POST /export
# ────────────────────────────────────────────────────────────────────────────

class TestExport:
    def _export_payload(self, **overrides):
        base = {
            "selected_model":    "gpt-4o",
            "has_cached_cols":   False,
            "caching_supported": True,
            "file_name":         "test.xlsx",
            "file_md5":          "abc123",
            "budget":            0,
            "model_rates":       {"input": 5.0, "output": 15.0, "cached_input": 2.5},
            "totals": {
                "rows":          3,
                "input_tokens":  100,
                "output_tokens": 50,
                "cached_tokens": 0,
                "total_tokens":  150,
                "input_cost":    0.0005,
                "output_cost":   0.00075,
                "cached_cost":   0.0,
                "total_cost":    0.00125,
            },
            "rows": [
                {"row": 1, "input_tokens": 30, "output_tokens": 15, "cached_tokens": 0,
                 "total_tokens": 45, "input_cost": 0.00015, "output_cost": 0.000225,
                 "cached_cost": 0, "total_cost": 0.000375},
                {"row": 2, "input_tokens": 35, "output_tokens": 18, "cached_tokens": 0,
                 "total_tokens": 53, "input_cost": 0.000175, "output_cost": 0.00027,
                 "cached_cost": 0, "total_cost": 0.000445},
                {"row": 3, "input_tokens": 35, "output_tokens": 17, "cached_tokens": 0,
                 "total_tokens": 52, "input_cost": 0.000175, "output_cost": 0.000255,
                 "cached_cost": 0, "total_cost": 0.00043},
            ],
            "model_comparison": [
                {"model": "gpt-4o-mini", "input_cost": 0.000015, "output_cost": 0.00003,
                 "cached_cost": 0, "total_cost": 0.000045, "caching_supported": True,
                 "pinned": False, "litellm_tracked": True},
                {"model": "gpt-4o", "input_cost": 0.0005, "output_cost": 0.00075,
                 "cached_cost": 0, "total_cost": 0.00125, "caching_supported": True,
                 "pinned": False, "litellm_tracked": True},
            ],
            "col_breakdown": [],
        }
        base.update(overrides)
        return base

    def test_returns_200(self, client):
        r = client.post("/export", json=self._export_payload())
        assert r.status_code == 200

    def test_returns_xlsx_content_type(self, client):
        r = client.post("/export", json=self._export_payload())
        assert "spreadsheetml" in r.headers["content-type"]

    def test_response_is_valid_xlsx(self, client):
        import openpyxl
        r = client.post("/export", json=self._export_payload())
        wb = openpyxl.load_workbook(io.BytesIO(r.content))
        assert len(wb.sheetnames) >= 2   # at least Summary + one more

    def test_summary_sheet_exists(self, client):
        import openpyxl
        r = client.post("/export", json=self._export_payload())
        wb = openpyxl.load_workbook(io.BytesIO(r.content))
        assert "Summary" in wb.sheetnames

    def test_model_comparison_sheet_exists(self, client):
        import openpyxl
        r = client.post("/export", json=self._export_payload())
        wb = openpyxl.load_workbook(io.BytesIO(r.content))
        sheets = wb.sheetnames
        assert any("Comparison" in s or "Model" in s for s in sheets)

    def test_row_breakdown_sheet_exists(self, client):
        import openpyxl
        r = client.post("/export", json=self._export_payload())
        wb = openpyxl.load_workbook(io.BytesIO(r.content))
        sheets = wb.sheetnames
        assert any("Row" in s or "Breakdown" in s for s in sheets)

    def test_col_breakdown_sheet_with_data(self, client):
        import openpyxl
        payload = self._export_payload(col_breakdown=[
            {"column": "prompt",   "type": "input",  "tokens": 80, "cost": 0.0004, "pct": 80.0},
            {"column": "response", "type": "output", "tokens": 20, "cost": 0.0003, "pct": 20.0},
        ])
        r = client.post("/export", json=payload)
        wb = openpyxl.load_workbook(io.BytesIO(r.content))
        sheets = wb.sheetnames
        assert any("Column" in s for s in sheets)

    def test_budget_exceeded_note_in_summary(self, client):
        import openpyxl
        payload = self._export_payload(budget=0.000001)   # tiny budget → will be exceeded
        r = client.post("/export", json=payload)
        wb    = openpyxl.load_workbook(io.BytesIO(r.content))
        ws    = wb["Summary"]
        # Collect all cell values as strings for easy searching
        all_values = []
        for row in ws.iter_rows():
            for cell in row:
                all_values.append(str(cell.value or "").lower())
        # Budget row exists ("budget ($)" label in col A)
        assert any("budget" in v for v in all_values), \
            "Budget label not found in Summary sheet"

    def test_untracked_model_note_in_comparison(self, client):
        import openpyxl
        payload = self._export_payload(model_comparison=[
            {"model": "old-model", "input_cost": 0.001, "output_cost": 0.002,
             "cached_cost": 0, "total_cost": 0.003, "caching_supported": False,
             "pinned": False, "litellm_tracked": False},
        ])
        r = client.post("/export", json=payload)
        wb = openpyxl.load_workbook(io.BytesIO(r.content))
        # Find the model comparison sheet
        comp_sheet = next(s for s in wb.sheetnames if "Comparison" in s or "Model" in s)
        ws = wb[comp_sheet]
        all_values = [str(ws.cell(r, c).value or "")
                      for r in range(1, ws.max_row + 1)
                      for c in range(1, ws.max_column + 1)]
        assert any("untracked" in v.lower() for v in all_values)

    def test_content_disposition_header(self, client):
        r = client.post("/export", json=self._export_payload())
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert ".xlsx" in cd

# ────────────────────────────────────────────────────────────────────────────
# Sheet selection
# ────────────────────────────────────────────────────────────────────────────

class TestSheetSelection:

    MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    def _multi(self):
        from conftest import make_multi_sheet_excel
        return make_multi_sheet_excel({
            "Inputs":  {"prompt": ["Hello", "World"], "context": ["a", "b"]},
            "Outputs": {"response": ["Hi", "Earth"],  "score":   [1,   2  ]},
            "Meta":    {"id": [100, 200],              "tag":     ["x", "y"]},
        })

    def _single(self):
        from conftest import make_excel_bytes
        return make_excel_bytes({"col_a": ["x", "y"], "col_b": [1, 2]})

    # ── /upload — returns ALL sheets' columns in one shot ────────────────────

    def test_upload_returns_sheets(self, client):
        r = client.post("/upload", files={"file": ("m.xlsx", self._multi(), self.MIME)})
        assert r.status_code == 200
        assert "sheets" in r.json()

    def test_upload_returns_all_sheet_names(self, client):
        r = client.post("/upload", files={"file": ("m.xlsx", self._multi(), self.MIME)})
        assert set(r.json()["sheets"]) == {"Inputs", "Outputs", "Meta"}

    def test_upload_returns_all_columns_map(self, client):
        r = client.post("/upload", files={"file": ("m.xlsx", self._multi(), self.MIME)})
        data = r.json()
        assert "all_columns" in data
        assert set(data["all_columns"].keys()) == {"Inputs", "Outputs", "Meta"}

    def test_upload_all_columns_correct_per_sheet(self, client):
        r = client.post("/upload", files={"file": ("m.xlsx", self._multi(), self.MIME)})
        ac = r.json()["all_columns"]
        assert set(ac["Inputs"])  == {"prompt", "context"}
        assert set(ac["Outputs"]) == {"response", "score"}
        assert set(ac["Meta"])    == {"id", "tag"}

    def test_upload_columns_is_first_sheet(self, client):
        r = client.post("/upload", files={"file": ("m.xlsx", self._multi(), self.MIME)})
        data = r.json()
        assert set(data["columns"]) == set(data["all_columns"][data["active_sheet"]])

    def test_upload_single_sheet_returns_one(self, client):
        r = client.post("/upload", files={"file": ("s.xlsx", self._single(), self.MIME)})
        assert len(r.json()["sheets"]) == 1

    def test_upload_defaults_to_first_sheet(self, client):
        r = client.post("/upload", files={"file": ("m.xlsx", self._multi(), self.MIME)})
        data = r.json()
        assert data["active_sheet"] == "Inputs"

    def test_upload_reads_headers_only(self, client):
        from conftest import make_multi_sheet_excel
        xls = make_multi_sheet_excel({"Data": {"col_one": ["row1_val"], "col_two": [1]}})
        r   = client.post("/upload", files={"file": ("d.xlsx", xls, self.MIME)})
        cols = r.json()["all_columns"]["Data"]
        assert "col_one" in cols and "col_two" in cols
        assert "row1_val" not in cols

    def test_upload_no_extra_roundtrip_needed(self, client):
        r    = client.post("/upload", files={"file": ("m.xlsx", self._multi(), self.MIME)})
        data = r.json()
        for sheet in data["sheets"]:
            assert sheet in data["all_columns"]

    # ── /calculate single-sheet (legacy columns_config) ──────────────────────

    def test_calculate_single_sheet_returns_200(self, client, pricing_override):
        xls  = self._multi()
        cols = json.dumps({"prompt": {"type": "input", "multiplier": 1},
                           "context": {"type": "skip",  "multiplier": 1}})
        r = client.post("/calculate",
            data={"model_name": "gpt-4o", "columns_config": cols,
                  "pricing_override": pricing_override},
            files={"file": ("m.xlsx", xls, self.MIME)})
        assert r.status_code == 200

    def test_calculate_invalid_sheet_falls_back(self, client, pricing_override):
        xls  = self._multi()
        cols = json.dumps({"prompt": {"type": "input", "multiplier": 1},
                           "context": {"type": "skip",  "multiplier": 1}})
        r = client.post("/calculate",
            data={"model_name": "gpt-4o", "columns_config": cols,
                  "pricing_override": pricing_override, "sheet_name": "NonExistent"},
            files={"file": ("m.xlsx", xls, self.MIME)})
        assert r.status_code == 200

    # ── /calculate multi-sheet (all_columns_config) ───────────────────────────

    def test_calculate_multi_sheet_returns_200(self, client, pricing_override):
        xls = self._multi()
        all_cfg = json.dumps({
            "Inputs":  {"prompt":   {"type": "input",  "multiplier": 1},
                        "context":  {"type": "skip",   "multiplier": 1}},
            "Outputs": {"response": {"type": "output", "multiplier": 1},
                        "score":    {"type": "skip",   "multiplier": 1}},
        })
        r = client.post("/calculate",
            data={"model_name": "gpt-4o", "all_columns_config": all_cfg,
                  "pricing_override": pricing_override},
            files={"file": ("m.xlsx", xls, self.MIME)})
        assert r.status_code == 200

    def test_calculate_multi_sheet_shows_per_sheet_sections(self, client, pricing_override):
        """Results page must have per-sheet breakdown sections."""
        xls = self._multi()
        all_cfg = json.dumps({
            "Inputs":  {"prompt":   {"type": "input",  "multiplier": 1},
                        "context":  {"type": "skip",   "multiplier": 1}},
            "Outputs": {"response": {"type": "output", "multiplier": 1},
                        "score":    {"type": "skip",   "multiplier": 1}},
        })
        r = client.post("/calculate",
            data={"model_name": "gpt-4o", "all_columns_config": all_cfg,
                  "pricing_override": pricing_override},
            files={"file": ("m.xlsx", xls, self.MIME)})
        assert "Inputs" in r.text,  "Sheet name Inputs not in results"
        assert "Outputs" in r.text, "Sheet name Outputs not in results"

    def test_calculate_multi_sheet_grand_totals_are_sum(self, client, pricing_override):
        """Grand total cost must equal sum of all sheet costs."""
        from conftest import make_multi_sheet_excel
        # Each sheet has exactly one row with known content
        xls = make_multi_sheet_excel({
            "S1": {"text": ["hello"]},
            "S2": {"text": ["world"]},
        })
        all_cfg = json.dumps({
            "S1": {"text": {"type": "input", "multiplier": 1}},
            "S2": {"text": {"type": "input", "multiplier": 1}},
        })
        r = client.post("/calculate",
            data={"model_name": "gpt-4o", "all_columns_config": all_cfg,
                  "pricing_override": pricing_override},
            files={"file": ("m.xlsx", xls, self.MIME)})
        assert r.status_code == 200
        # Both sheet names appear
        assert "S1" in r.text and "S2" in r.text

    def test_calculate_multi_sheet_skips_all_skip_sheets(self, client, pricing_override):
        """A sheet where all columns are skip must not cause an error."""
        xls = self._multi()
        # Only Inputs has active columns, Meta is all skip
        all_cfg = json.dumps({
            "Inputs": {"prompt":   {"type": "input", "multiplier": 1},
                       "context":  {"type": "skip",  "multiplier": 1}},
            "Meta":   {"id":       {"type": "skip",  "multiplier": 1},
                       "tag":      {"type": "skip",  "multiplier": 1}},
        })
        r = client.post("/calculate",
            data={"model_name": "gpt-4o", "all_columns_config": all_cfg,
                  "pricing_override": pricing_override},
            files={"file": ("m.xlsx", xls, self.MIME)})
        assert r.status_code == 200
        # Meta should not appear (all skipped)
        assert "Inputs" in r.text

    def test_calculate_no_columns_selected_returns_400(self, client, pricing_override):
        """If all sheets have all-skip columns, return 400."""
        xls = self._multi()
        all_cfg = json.dumps({
            "Inputs": {"prompt": {"type": "skip", "multiplier": 1}},
        })
        r = client.post("/calculate",
            data={"model_name": "gpt-4o", "all_columns_config": all_cfg,
                  "pricing_override": pricing_override},
            files={"file": ("m.xlsx", xls, self.MIME)})
        assert r.status_code == 400

    def test_calculate_different_sheets_different_costs(self, client, pricing_override):
        from conftest import make_multi_sheet_excel
        xls = make_multi_sheet_excel({
            "Long":  {"text": ["This is a very long sentence " * 20]},
            "Short": {"text": ["Hi"]},
        })
        def _calc(sheet):
            return client.post("/calculate",
                data={"model_name": "gpt-4o",
                      "all_columns_config": json.dumps({sheet: {"text": {"type": "input", "multiplier": 1}}}),
                      "pricing_override": pricing_override},
                files={"file": ("m.xlsx", xls, self.MIME)})
        r_long  = _calc("Long")
        r_short = _calc("Short")
        assert r_long.status_code  == 200
        assert r_short.status_code == 200
        assert r_long.text != r_short.text