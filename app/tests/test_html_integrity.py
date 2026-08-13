"""
HTML & JS integrity tests.
These catch the class of bugs we've been hitting:
 - truncated files
 - missing script blocks
 - syntax errors in JS
 - missing critical DOM IDs
 - leftover template builder code
"""
import os
import re
import sys
import subprocess
import pytest

SRC_DIR = os.path.dirname(os.path.dirname(__file__))
INDEX_PATH  = os.path.join(SRC_DIR, "templates", "index.html")
SUMMARY_PATH = os.path.join(SRC_DIR, "templates", "summary.html")


def load(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def extract_main_script(content):
    """Return the largest inline <script> block (the main JS)."""
    scripts = re.findall(
        r'<script(?![^>]*(?:src|type="application))[^>]*>(.*?)</script>',
        content, re.DOTALL)
    if not scripts:
        return ""
    return max(scripts, key=len)


def js_syntax_ok(script_text):
    """Return (ok, error_message) by running node --check on a temp file."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js",
                                     delete=False, encoding="utf-8") as f:
        f.write(script_text)
        tmp = f.name
    try:
        node = subprocess.run(
            ["node", "--check", tmp],
            capture_output=True, text=True)
        return node.returncode == 0, node.stderr.strip()
    finally:
        os.unlink(tmp)


# ────────────────────────────────────────────────────────────────────────────
# index.html
# ────────────────────────────────────────────────────────────────────────────

class TestIndexHTML:

    @pytest.fixture(autouse=True)
    def content(self):
        self.html = load(INDEX_PATH)
        self.script = extract_main_script(self.html)

    # File completeness
    def test_file_not_empty(self):
        assert len(self.html) > 10_000, "index.html seems truncated"

    def test_has_closing_body(self):
        assert "</body>" in self.html

    def test_has_closing_html(self):
        assert "</html>" in self.html

    def test_has_closing_script(self):
        assert "</script>" in self.html

    # JS syntax
    def test_main_script_exists(self):
        assert len(self.script) > 5_000, "Main JS block is too small — likely missing"

    def test_js_syntax_valid(self):
        ok, err = js_syntax_ok(self.script)
        assert ok, f"JS syntax error in index.html:\n{err}"

    # Critical JS functions
    @pytest.mark.parametrize("fn", [
        "handleFile",
        "renderColumnsTable",
        "safeId",
        "selectFileType",
        "setCalcStep",
        "switchProvider",
        "rebuildPricingTable",
        "attachPinHandlers",
        "openPanel",
        "closePanel",
        "buildPanelHTML",
        "applyTheme",
        "toggleTheme",
        "showAlert",
        "hideFetchSummary",
        "getPricingFromTable",
        "onTypeChange",         # badge updater (was updateBadge in v1)
        "reorderModelSelector", # moves pinned models to top of dropdown + updates ⭐
        "reorderPinnedRows",    # moves pinned rows to top of pricing table
        "renderSheetTabs",        # builds sheet tab buttons after Excel upload
        "selectSheet",            # switches sheet and restores config
        "saveCurrentSheetConfig", # saves current sheet column selections
        "restoreSheetConfig",     # restores column selections when switching sheets
        "updateSheetTabBadges",   # shows active column count on each sheet tab
    ])
    def test_function_defined(self, fn):
        assert f"function {fn}" in self.script, f"Missing JS function: {fn}"

    # Critical upload flow
    def test_fetch_upload(self):
        assert "fetch('/upload'" in self.script

    def test_columns_section_shown(self):
        assert "columnsSection" in self.script

    def test_calc_section_shown(self):
        assert "calcSection" in self.script

    # Calculate form
    def test_calculateBtn_listener(self):
        assert "calculateBtn" in self.script

    def test_columns_object_initialised(self):
        assert "const columns = {};" in self.script

    def test_excel_form_action(self):
        assert "form.action = '/calculate';" in self.script

    def test_text_form_action(self):
        assert "form.action = '/calculate-text';" in self.script

    # Provider tabs
    def test_switchProvider_called(self):
        assert "switchProvider" in self.html

    def test_all_providers_in_html(self):
        assert "openai"    in self.html.lower()
        assert "anthropic" in self.html.lower()
        assert "gemini"    in self.html.lower()

    # Theme
    def test_theme_toggle_in_html(self):
        assert "themeToggle" in self.html

    def test_local_storage_theme(self):
        assert "llm-calc-theme" in self.script

    # Critical DOM IDs
    @pytest.mark.parametrize("el_id", [
        "uploadZone", "fileInput", "fileName",
        "columnsSection", "columnsTable",
        "calcSection", "modelSelect", "calculateBtn",
        "pricingTableBody", "fetchBtn", "savePricingBtn",
        "fetchStatus", "fetchError", "saveAlert", "fetchSummary",
        "pricingCollapsible",
        "sheetSection",
        "sheetTabs",
    ])
    def test_dom_id_present(self, el_id):
        assert el_id in self.html, f"Missing DOM id: {el_id}"

    # No template builder leftovers
    def test_no_template_builder_js(self):
        assert "toggleTemplateBuilder" not in self.html
        assert "_templateActive"       not in self.html
        assert "templateSection"       not in self.html
        assert "templateBody"          not in self.html

    def test_reorder_model_selector_updates_star(self):
        """reorderModelSelector must update ⭐ prefix — the bug we fixed."""
        assert "reorderModelSelector" in self.script
        # The fix: option textContent is updated based on pinned state
        assert "o.textContent" in self.script or "opt.textContent" in self.script

    def test_sheet_selector_css_present(self):
        assert ".sheet-tab" in self.html, "Missing .sheet-tab CSS class"


    def test_sheet_name_sent_with_calculate(self):
        assert "sheet_name" in self.script,             "sheet_name must be sent as hidden field in calculate form"

    def test_current_sheet_state_variable(self):
        assert "currentSheet" in self.script, \
            "currentSheet state variable must track the active sheet"

    def test_all_columns_cache_state_variable(self):
        assert "allColumnsCache" in self.script, \
            "allColumnsCache must be populated from /upload response"

    def test_all_sheet_columns_config_state_variable(self):
        assert "allSheetColumnsConfig" in self.script, \
            "allSheetColumnsConfig must store per-sheet column selections"

    def test_all_columns_config_sent_to_calculate(self):
        assert "all_columns_config" in self.script, \
            "all_columns_config must be sent to /calculate for multi-sheet support"

    def test_sheet_tab_badge_css(self):
        assert "sheet-col-badge" in self.html, \
            "Missing .sheet-col-badge CSS for active column count on sheet tabs"

    def test_select_sheet_uses_cache_not_network(self):
        assert "sheet-columns" not in self.script, \
            "selectSheet should use allColumnsCache, not re-fetch from server"

    # ── Upload progress bar ──────────────────────────────────────────────────

    def test_upload_progress_bar_html_present(self):
        assert "uploadProgress"  in self.html
        assert "progressFill"    in self.html
        assert "progressLabel"   in self.html

    def test_upload_progress_bar_css_present(self):
        assert "upload-progress"    in self.html
        assert "progress-bar-fill"  in self.html

    def test_upload_progress_shown_and_hidden(self):
        assert "progress.classList.add('show')"    in self.script
        assert "progress.classList.remove('show')" in self.script

    # ── Calculation progress bar ─────────────────────────────────────────────

    def test_calc_progress_bar_html_present(self):
        assert "calcProgress"      in self.html
        assert "calcProgressFill"  in self.html
        assert "calcProgressLabel" in self.html

    def test_calc_progress_bar_css_present(self):
        assert "calc-progress" in self.html

    def test_calculate_uses_fetch_not_form_submit(self):
        """Calculate must use fetch() so progress bar can animate during server work."""
        assert "fetch(endpoint" in self.script, \
            "Calculate must use fetch() to keep page alive for progress bar"
        assert "form.submit()" not in self.script, \
            "form.submit() blocks JS — progress bar cannot animate"

    def test_calculate_progress_ticker(self):
        """Progress must animate continuously while server processes."""
        assert "ticker"              in self.script
        assert "clearInterval(ticker)" in self.script

    def test_calculate_button_restored_on_error(self):
        assert "btn.disabled    = false" in self.script \
            or "btn.disabled = false"    in self.script

    def test_calculate_error_shown_to_user(self):
        assert "Network error" in self.script or "Calculation failed" in self.script


        """switchProvider must fetch /provider-data on every switch — not use stale cache."""
        assert "provider-data" in self.script, \
            "switchProvider must fetch /provider-data to avoid stale pin/price state"
        assert "ALL_PROVIDERS[provider] = pd" in self.script, \
            "switchProvider must update the local cache after fetching fresh data"

    def test_rebuild_pricing_table_uses_ordered_models(self):
        """rebuildPricingTable must use the server-sorted models array (pinned first),
        not Object.entries(prices) which uses JSON insertion order."""
        assert "orderedModels" in self.script, \
            "rebuildPricingTable must use orderedModels array to preserve pinned-first order"
        assert "const orderedModels = models || Object.keys(prices)" in self.script

    # Cloudflare obfuscation check
    def test_no_cf_email_obfuscation(self):
        assert "cdn-cgi/l/email-protection" not in self.html

    def test_email_present(self):
        assert "hasan.zamzam@gmail.com" in self.html


# ────────────────────────────────────────────────────────────────────────────
# summary.html
# ────────────────────────────────────────────────────────────────────────────

class TestSummaryHTML:

    @pytest.fixture(autouse=True)
    def content(self):
        self.html = load(SUMMARY_PATH)
        self.script = extract_main_script(self.html)

    def test_file_not_empty(self):
        assert len(self.html) > 5_000

    def test_has_closing_body(self):
        assert "</body>" in self.html

    def test_has_closing_html(self):
        assert "</html>" in self.html

    def test_js_syntax_valid(self):
        if len(self.script) < 50:
            pytest.skip("No meaningful JS in summary.html")
        ok, err = js_syntax_ok(self.script)
        assert ok, f"JS syntax error in summary.html:\n{err}"

    # Jinja template variables
    @pytest.mark.parametrize("var", [
        "selected_model", "rows", "total_input_tokens",
        "total_output_tokens", "dataset_total_tokens",
        "total_input_cost", "total_output_cost", "dataset_total_cost",
        "model_comparison", "has_cached_cols", "col_breakdown",
        "file_name", "file_md5", "budget",
    ])
    def test_template_variable_present(self, var):
        assert var in self.html, f"Missing Jinja variable: {var}"

    # Key sections
    def test_stats_grid(self):
        assert "stats-grid" in self.html

    def test_model_comparison_section(self):
        assert "Model Cost Comparison" in self.html

    def test_row_breakdown_section(self):
        assert "Row-by-Row Breakdown" in self.html

    def test_projection_section(self):
        assert "Cost Projection" in self.html or "proj" in self.html.lower()

    def test_col_breakdown_section(self):
        assert "Token Breakdown by Column" in self.html

    def test_multi_sheet_conditional_present(self):
        assert "multi_sheet" in self.html, \
            "summary.html must have multi_sheet conditional for per-sheet results"

    def test_per_sheet_results_section(self):
        assert "sheet_results" in self.html, \
            "summary.html must iterate sheet_results for per-sheet breakdown"

    def test_sheet_badge_css_in_summary(self):
        assert "sheet-badge" in self.html, "Missing .sheet-badge CSS in summary.html"

    def test_export_button(self):
        assert "exportBtn" in self.html or "Export to Excel" in self.html

    def test_untracked_badge_defined(self):
        assert "untracked by LiteLLM" in self.html

    # export JS
    def test_export_function(self):
        assert "exportToExcel" in self.html

    def test_export_fetches_export_endpoint(self):
        assert "'/export'" in self.html or '"/export"' in self.html

    def test_export_data_script_tag(self):
        assert "exportData" in self.html

    def test_theme_toggle(self):
        assert "themeToggle" in self.html
        assert "applyTheme"  in self.html

    # No Cloudflare obfuscation
    def test_no_cf_email_obfuscation(self):
        assert "cdn-cgi/l/email-protection" not in self.html

    def test_email_present(self):
        assert "hasan.zamzam@gmail.com" in self.html


# ────────────────────────────────────────────────────────────────────────────
# Pricing JSON files
# ────────────────────────────────────────────────────────────────────────────

class TestPricingJSON:

    @pytest.mark.parametrize("fname", [
        "openai_ai_models_pricing.json",
        "anthropic_ai_models_pricing.json",
        "google_ai_models_pricing.json",
    ])
    def test_file_exists(self, fname):
        path = os.path.join(SRC_DIR, fname)
        assert os.path.exists(path), f"Missing pricing file: {fname}"

    @pytest.mark.parametrize("fname", [
        "openai_ai_models_pricing.json",
        "anthropic_ai_models_pricing.json",
        "google_ai_models_pricing.json",
    ])
    def test_valid_json(self, fname):
        path = os.path.join(SRC_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_openai_has_gpt4o(self):
        path = os.path.join(SRC_DIR, "openai_ai_models_pricing.json")
        with open(path) as f:
            data = json.load(f)
        assert "gpt-4o" in data

    def test_all_models_have_pinned_field(self):
        """Every model entry must have the pinned field (needed for pinning feature)."""
        for fname in ["openai_ai_models_pricing.json",
                      "anthropic_ai_models_pricing.json",
                      "google_ai_models_pricing.json"]:
            path = os.path.join(SRC_DIR, fname)
            with open(path) as f:
                data = json.load(f)
            for key, val in data.items():
                if key in ("last_updated", "last_updated_source"):
                    continue
                if isinstance(val, dict):
                    assert "pinned" in val, \
                        f"{fname}: model '{key}' missing 'pinned' field"

    def test_no_nan_in_json(self):
        """JSON must not contain NaN/Infinity (breaks JSON.parse in browser)."""
        import math
        for fname in ["openai_ai_models_pricing.json",
                      "anthropic_ai_models_pricing.json",
                      "google_ai_models_pricing.json"]:
            path = os.path.join(SRC_DIR, fname)
            with open(path) as f:
                data = json.load(f)
            for key, val in data.items():
                if not isinstance(val, dict):
                    continue
                for field, v in val.items():
                    if isinstance(v, float):
                        assert not math.isnan(v),  f"{fname}/{key}/{field} = NaN"
                        assert not math.isinf(v),  f"{fname}/{key}/{field} = Inf"


import json  # needed by TestPricingJSON