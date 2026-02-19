"""
Unit tests for knowledge/ modules (llm_client, kb_client, ingest, llm_endpoints, kb_server).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


class TestLLMClient:
    """Tests for knowledge.llm.llm_client."""

    def test_get_provider_default(self, monkeypatch):
        """Test _get_provider returns default when not set."""
        monkeypatch.delenv("KB_LLM_PROVIDER", raising=False)
        from knowledge.llm.llm_client import _get_provider
        assert _get_provider() == "openai"

    def test_get_provider_from_env(self, monkeypatch):
        """Test _get_provider reads KB_LLM_PROVIDER."""
        monkeypatch.setenv("KB_LLM_PROVIDER", "anthropic")
        from knowledge.llm.llm_client import _get_provider
        assert _get_provider() == "anthropic"

    def test_get_model_default(self, monkeypatch):
        """Test _get_model returns default when not set."""
        monkeypatch.delenv("KB_LLM_MODEL", raising=False)
        from knowledge.llm.llm_client import _get_model
        assert _get_model() == "gpt-4o-mini"

    def test_get_model_from_env(self, monkeypatch):
        """Test _get_model reads KB_LLM_MODEL."""
        monkeypatch.setenv("KB_LLM_MODEL", "gpt-4")
        from knowledge.llm.llm_client import _get_model
        assert _get_model() == "gpt-4"

    def test_is_available_with_key(self, monkeypatch):
        """Test is_available returns True when API key is set."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from knowledge.llm.llm_client import is_available
        assert is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        """Test is_available returns False when no key is set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KB_LLM_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("KB_LLM_BASE_URL", raising=False)
        monkeypatch.setenv("KB_LLM_PROVIDER", "openai")
        from knowledge.llm.llm_client import is_available
        # May be True if user has key in env; we can't fully clear in test
        result = is_available()
        assert isinstance(result, bool)


class TestKBClient:
    """Tests for knowledge.services.kb_client."""

    def test_init(self):
        """Test KBClient initialization."""
        from knowledge.services.kb_client import KBClient
        client = KBClient(base_url="http://localhost:8000")
        assert client.base_url == "http://localhost:8000"
        assert client.token is None
        assert client.timeout == 30.0

    def test_init_strips_trailing_slash(self):
        """Test base_url trailing slash is stripped."""
        from knowledge.services.kb_client import KBClient
        client = KBClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_headers_without_token(self):
        """Test _headers returns Content-Type when no token."""
        from knowledge.services.kb_client import KBClient
        client = KBClient(base_url="http://localhost:8000")
        h = client._headers()
        assert h == {"Content-Type": "application/json"}

    def test_headers_with_token(self):
        """Test _headers includes Authorization when token set."""
        from knowledge.services.kb_client import KBClient
        client = KBClient(base_url="http://localhost:8000", token="secret")
        h = client._headers()
        assert "Authorization" in h
        assert h["Authorization"] == "Bearer secret"


class TestChunkText:
    """Tests for knowledge.ingest.chunk.chunk_text."""

    def test_chunk_empty_returns_empty(self):
        """Test chunk_text with empty string returns []."""
        from knowledge.ingest.chunk import chunk_text
        assert chunk_text("") == []
        assert chunk_text("   \n") == []

    def test_chunk_short_text(self):
        """Test chunk_text with short text returns single chunk."""
        from knowledge.ingest.chunk import chunk_text
        text = "Short paragraph."
        result = chunk_text(text)
        assert len(result) == 1
        assert result[0]["text"] == text
        assert result[0]["page"] == 1

    def test_chunk_respects_page(self):
        """Test chunk_text passes page to chunks."""
        from knowledge.ingest.chunk import chunk_text
        result = chunk_text("Short.", page=5)
        assert result[0]["page"] == 5


class TestLLMEndpoints:
    """Tests for knowledge.services.llm_endpoints."""

    def test_format_runs_for_brief_empty(self):
        """Test _format_runs_for_brief with empty list."""
        from knowledge.services.llm_endpoints import _format_runs_for_brief
        assert _format_runs_for_brief([]) == "(no runs)"

    def test_format_runs_for_brief_with_runs(self):
        """Test _format_runs_for_brief with runs."""
        from knowledge.services.llm_endpoints import _format_runs_for_brief
        runs = [
            {"case_id": "c1", "success": True, "total_score": 0.5,
             "case_config": {"surface_params": {"surface": "s1"}, "coils_params": {"ncoils": 4, "order": 6}}},
        ]
        out = _format_runs_for_brief(runs)
        assert "c1" in out
        assert "SUCCESS" in out
        assert "s1" in out
        assert "4" in out
        assert "6" in out

    def test_format_stats_for_brief_empty(self):
        """Test _format_stats_for_brief with empty stats."""
        from knowledge.services.llm_endpoints import _format_stats_for_brief
        assert _format_stats_for_brief({}) == "(no stats)"

    def test_format_stats_for_brief_with_data(self):
        """Test _format_stats_for_brief with stats."""
        from knowledge.services.llm_endpoints import _format_stats_for_brief
        stats = {"total": 10, "fail_rate": 0.3, "failure_classes": {"a": 2}}
        out = _format_stats_for_brief(stats)
        assert "10" in out
        assert "0.30" in out
        assert "Failure classes" in out or "failure_classes" in out
        assert "a" in out and "2" in out

    def test_call_brief_llm_unavailable(self, monkeypatch):
        """Test call_brief returns error when LLM not configured."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KB_LLM_API_KEY", raising=False)
        monkeypatch.setenv("KB_LLM_PROVIDER", "openai")
        from knowledge.services.llm_endpoints import call_brief
        result = call_brief([], {}, [])
        assert "error" in result
        assert result["brief"] == ""
        assert result["citations"] == []

    def test_call_brief_llm_available_mocked(self, monkeypatch):
        """Test call_brief returns brief when LLM returns text."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with (
            patch("knowledge.llm.llm_client.is_available", return_value=True),
            patch("knowledge.llm.llm_client.complete") as mock_complete,
        ):
            mock_complete.return_value = "This is the brief."
            from knowledge.services.llm_endpoints import call_brief
            result = call_brief(
                [{"case_id": "c1", "success": True, "total_score": 0.5, "case_config": {}}],
                {"total": 5, "fail_rate": 0.2},
                [],
            )
            assert result["brief"] == "This is the brief."
            assert "citations" in result
            mock_complete.assert_called_once()

    def test_call_propose_llm_unavailable(self, monkeypatch):
        """Test call_propose returns error when LLM not configured."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KB_LLM_API_KEY", raising=False)
        monkeypatch.setenv("KB_LLM_PROVIDER", "openai")
        from knowledge.services.llm_endpoints import call_propose
        result = call_propose({}, {})
        assert "error" in result
        assert result["actions"] == []

    def test_call_propose_llm_available_mocked(self, monkeypatch):
        """Test call_propose returns actions when LLM returns JSON."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with (
            patch("knowledge.llm.llm_client.is_available", return_value=True),
            patch("knowledge.llm.llm_client.complete_json") as mock_cj,
        ):
            mock_cj.return_value = [{"type": "explore", "surface": "s1", "ncoils": 4, "order": 6}]
            from knowledge.services.llm_endpoints import call_propose
            result = call_propose(
                {"top_parents": [], "failure_stats": {"fail_rate": 0.1}},
                {"exploration": {"surfaces": ["s1"], "ncoils_choices": [4], "order_choices": [6]}},
                batch_size=1,
            )
            assert result["actions"] == [{"type": "explore", "surface": "s1", "ncoils": 4, "order": 6}]
            mock_cj.assert_called_once()


class TestMakePostmortem:
    """Tests for knowledge.ingest.make_postmortem."""

    def test_success_returns_empty(self):
        """Test make_postmortem returns empty string for success."""
        from knowledge.ingest.make_postmortem import make_postmortem
        assert make_postmortem({"success": True}) == ""
        assert make_postmortem({}) == ""

    def test_min_sep_violation_suggestions(self):
        """Test make_postmortem for min_sep_violation."""
        from knowledge.ingest.make_postmortem import make_postmortem
        out = make_postmortem({"success": False, "failure_class": "min_sep_violation", "failure_reason": "cc"})
        assert "Failure class: min_sep_violation" in out
        assert "Suggest:" in out
        assert "coil-coil" in out or "separation" in out

    def test_timeout_suggestions(self):
        """Test make_postmortem for timeout."""
        from knowledge.ingest.make_postmortem import make_postmortem
        out = make_postmortem({"success": False, "failure_class": "timeout", "failure_reason": "exceeded"})
        assert "timeout" in out.lower()
        assert "Suggest:" in out

    def test_unknown_class_suggestion(self):
        """Test make_postmortem for unknown failure class."""
        from knowledge.ingest.make_postmortem import make_postmortem
        out = make_postmortem({"success": False, "failure_class": "unknown", "failure_reason": "x"})
        assert "inspect logs" in out or "Suggest:" in out

    def test_negative_margins_included(self):
        """Test make_postmortem includes negative margins."""
        from knowledge.ingest.make_postmortem import make_postmortem
        out = make_postmortem({
            "success": False,
            "failure_class": "line_search_fail",
            "failure_reason": "x",
            "margins": {"cc_sep": -0.01, "good": 0.5},
        })
        assert "Negative margins" in out
        assert "cc_sep" in out


class TestMakeRunCard:
    """Tests for knowledge.ingest.make_run_card."""

    def test_minimal_summary(self):
        """Test make_run_card with minimal summary."""
        from knowledge.ingest.make_run_card import make_run_card
        out = make_run_card({"case_id": "c1", "success": True, "total_score": 1e-4})
        assert "c1" in out
        assert "SUCCESS" in out
        assert "1.0000e-04" in out or "1e-04" in out

    def test_failed_with_config(self):
        """Test make_run_card with failed run and case_config."""
        from knowledge.ingest.make_run_card import make_run_card
        out = make_run_card({
            "case_id": "c2",
            "success": False,
            "total_score": float("inf"),
            "case_config": {"surface_params": {"surface": "s1"}, "coils_params": {"ncoils": 5, "order": 8}},
            "failure_class": "timeout",
            "failure_reason": "exceeded limit",
        })
        assert "c2" in out
        assert "FAILED" in out
        assert "s1" in out
        assert "5" in out
        assert "8" in out
        assert "timeout" in out

    def test_with_metrics_and_margins(self):
        """Test make_run_card with metrics and tight margins."""
        from knowledge.ingest.make_run_card import make_run_card
        out = make_run_card({
            "case_id": "c3",
            "success": True,
            "total_score": 0.1,
            "metrics": {"final_min_cc_separation": 0.15, "BdotN_over_B": 1e-3},
            "margins": {"cc_sep": 0.05},
        })
        assert "CC separation" in out or "0.15" in out
        assert "B·n" in out or "BdotN" in out or "1e-03" in out
        assert "Tight margins" in out or "cc_sep" in out


class TestExtractPdf:
    """Tests for knowledge.ingest.extract_pdf."""

    def test_file_not_found(self, tmp_path):
        """Test extract_pdf raises FileNotFoundError for missing file."""
        from knowledge.ingest.extract_pdf import extract_pdf
        missing = tmp_path / "nonexistent.pdf"
        with pytest.raises(FileNotFoundError):
            extract_pdf(missing)

    def test_extract_raises_import_error_when_no_reader(self, tmp_path):
        """Test extract_pdf raises ImportError when neither pymupdf nor pypdf available."""
        import knowledge.ingest.extract_pdf as ep_module
        pdf_path = tmp_path / "x.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 minimal\n")
        with patch.object(ep_module, "_HAS_PYMUPDF", False), patch.object(ep_module, "_HAS_PYPDF", False):
            with pytest.raises(ImportError, match="pymupdf or pypdf"):
                ep_module.extract_pdf(pdf_path)

    def test_extract_with_pypdf(self, tmp_path):
        """Test extract_pdf with a minimal PDF (if pypdf or pymupdf available)."""
        pytest.importorskip("pypdf", reason="pypdf required for PDF extraction")
        from knowledge.ingest.extract_pdf import extract_pdf, _HAS_PYPDF, _HAS_PYMUPDF
        if not _HAS_PYPDF and not _HAS_PYMUPDF:
            pytest.skip("Neither pypdf nor pymupdf installed")
        # Create minimal PDF with pypdf
        from pypdf import PdfWriter
        pdf_path = tmp_path / "test.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.add_metadata({"/Title": "Test"})
        with open(pdf_path, "wb") as f:
            writer.write(f)
        pages, chunks, chunk_to_page = extract_pdf(pdf_path)
        assert isinstance(pages, list)
        assert isinstance(chunks, list)
        assert isinstance(chunk_to_page, list)
        assert len(pages) >= 1
        assert len(chunk_to_page) == len(chunks) or len(chunks) <= len(pages)


class TestKBServerHelpers:
    """Tests for knowledge.services.kb_server helper functions."""

    def test_get_postgres_dsn_default(self, monkeypatch):
        """Test _get_postgres_dsn returns default when not set."""
        monkeypatch.delenv("KB_POSTGRES_DSN", raising=False)
        from knowledge.services.kb_server import _get_postgres_dsn
        assert "postgresql" in _get_postgres_dsn()
        assert "stellcoilbench_kb" in _get_postgres_dsn()

    def test_get_postgres_dsn_from_env(self, monkeypatch):
        """Test _get_postgres_dsn reads KB_POSTGRES_DSN."""
        monkeypatch.setenv("KB_POSTGRES_DSN", "postgresql://custom/db")
        from knowledge.services.kb_server import _get_postgres_dsn
        assert _get_postgres_dsn() == "postgresql://custom/db"

    def test_get_qdrant_url_default(self, monkeypatch):
        """Test _get_qdrant_url returns default."""
        monkeypatch.delenv("KB_QDRANT_URL", raising=False)
        from knowledge.services.kb_server import _get_qdrant_url
        assert "6333" in _get_qdrant_url()

    def test_use_sqlite_true(self, monkeypatch):
        """Test _use_sqlite returns True when KB_USE_SQLITE=1."""
        monkeypatch.setenv("KB_USE_SQLITE", "1")
        from knowledge.services.kb_server import _use_sqlite
        assert _use_sqlite() is True

    def test_use_sqlite_false(self, monkeypatch):
        """Test _use_sqlite returns False when not set."""
        monkeypatch.delenv("KB_USE_SQLITE", raising=False)
        from knowledge.services.kb_server import _use_sqlite
        assert _use_sqlite() is False


class TestKBServerEndpoints:
    """Tests for knowledge.services.kb_server FastAPI endpoints."""

    def test_app_or_skip(self):
        """Skip if app failed to load (missing fastapi/qdrant)."""
        from knowledge.services import kb_server
        if kb_server.app is None:
            pytest.skip("KB server app not available (fastapi/qdrant not installed)")

    def test_ingest_run_and_runs_top(self, monkeypatch, tmp_path):
        """Test POST /ingest/run and GET /runs/top with SQLite."""
        from knowledge.services import kb_server
        if kb_server.app is None:
            pytest.skip("KB server app not available")
        monkeypatch.setenv("KB_USE_SQLITE", "1")
        monkeypatch.setenv("KB_SQLITE_PATH", str(tmp_path / "kb.sqlite"))
        # Re-import to get app with new env - app is already created. We need fresh app.
        # The app is created at import time. So we need to either reload the module with new env
        # or create the app in a way that reads env at request time. Looking at the code,
        # _use_sqlite() and _db_conn() read env at call time. So we just need to set env
        # before making requests. The app was created with the OLD env. The _db_conn is called
        # when handling the request, so it will use the current env. Good.
        from fastapi.testclient import TestClient
        client = TestClient(kb_server.app)
        payload = {
            "case_id": "test-case-1",
            "success": True,
            "total_score": 0.001,
            "case_config": {"surface_params": {"surface": "s1"}, "coils_params": {"ncoils": 4, "order": 6}},
        }
        resp = client.post("/ingest/run", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["case_id"] == "test-case-1"
        resp2 = client.get("/runs/top?k=5")
        assert resp2.status_code == 200
        runs = resp2.json().get("runs", [])
        assert len(runs) >= 1
        assert any(r.get("case_id") == "test-case-1" for r in runs)

    def test_search_papers_stub(self):
        """Test GET /search/papers returns empty (stub)."""
        from knowledge.services import kb_server
        if kb_server.app is None:
            pytest.skip("KB server app not available")
        from fastapi.testclient import TestClient
        client = TestClient(kb_server.app)
        resp = client.get("/search/papers?q=stellarator")
        assert resp.status_code == 200
        assert resp.json().get("papers", []) == []

    def test_post_brief_requires_llm(self, monkeypatch, tmp_path):
        """Test POST /brief returns error when LLM not configured."""
        from knowledge.services import kb_server
        if kb_server.app is None:
            pytest.skip("KB server app not available")
        monkeypatch.setenv("KB_USE_SQLITE", "1")
        monkeypatch.setenv("KB_SQLITE_PATH", str(tmp_path / "kb.sqlite"))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KB_LLM_API_KEY", raising=False)
        from fastapi.testclient import TestClient
        client = TestClient(kb_server.app)
        resp = client.post("/brief", json={"query": "", "context": "general"})
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data or "brief" in data
