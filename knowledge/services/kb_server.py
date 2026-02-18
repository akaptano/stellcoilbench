#!/usr/bin/env python3
"""
Desktop KB server: Postgres or SQLite (runs) + Qdrant (run card embeddings).

Endpoints:
  POST /ingest/run     — upsert run summary (idempotent by case_id)
  GET  /runs/top       — top runs by score
  GET  /search/runs    — semantic search over run cards
  GET  /search/papers  — semantic search over paper chunks (stub)
  GET  /stats/recent   — failure histogram
  POST /brief          — LLM research brief (runs + papers)
  POST /propose        — LLM mutation actions (strict JSON)

Requires: fastapi, uvicorn, qdrant-client, sentence-transformers
  + psycopg[binary] for Postgres, or use SQLite (no extra deps)

Run: uvicorn knowledge.services.kb_server:app --host 0.0.0.0 --port 8000

Set env: KB_USE_SQLITE=1 to use SQLite (no Docker needed)
         KB_POSTGRES_DSN=postgresql://kb:kb@localhost:5432/stellcoilbench_kb
         KB_QDRANT_URL=http://localhost:6333 (default)
"""
from __future__ import annotations

import json
import os
from collections import Counter
from contextlib import contextmanager
from typing import Any, Generator

app = None


def _get_postgres_dsn() -> str:
    """Return Postgres connection string from KB_POSTGRES_DSN env."""
    return os.environ.get(
        "KB_POSTGRES_DSN",
        "postgresql://kb:kb@localhost:5432/stellcoilbench_kb",
    )


def _get_qdrant_url() -> str:
    """Return Qdrant server URL from KB_QDRANT_URL env."""
    return os.environ.get("KB_QDRANT_URL", "http://localhost:6333")


def _use_sqlite() -> bool:
    """Return True if KB_USE_SQLITE is set (use SQLite instead of Postgres)."""
    return os.environ.get("KB_USE_SQLITE", "").lower() in ("1", "true", "yes")


@contextmanager
def _db_conn() -> Generator[Any, None, None]:
    """Yield a DB connection (Postgres or SQLite)."""
    if _use_sqlite():
        import sqlite3
        db_path = os.environ.get("KB_SQLITE_PATH", "knowledge/kb.sqlite")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.commit()
            conn.close()
    else:
        try:
            import psycopg
        except ImportError:
            raise ImportError("Install psycopg: pip install 'psycopg[binary]'")
        with psycopg.connect(_get_postgres_dsn()) as conn:
            yield conn


def _ensure_schema(conn: Any) -> None:
    """Create runs table if it does not exist (Postgres or SQLite)."""
    if _use_sqlite():
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                case_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                run_card TEXT,
                postmortem TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
    else:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                case_id TEXT PRIMARY KEY,
                summary JSONB NOT NULL,
                run_card TEXT,
                postmortem TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        conn.commit()


def _get_embedder():
    """Return SentenceTransformer embedder (all-MiniLM-L6-v2) or None if not installed."""
    try:
        from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        return None


def _create_app():
    from fastapi import FastAPI, HTTPException

    _app = FastAPI(title="StellCoilBench KB (Desktop)", version="0.1.0")
    _embedder = _get_embedder()
    _qdrant: Any = None
    if _embedder:
        try:
            from qdrant_client import QdrantClient
            _qdrant = QdrantClient(url=_get_qdrant_url())
            _qdrant.get_collection("run_cards")
        except Exception:
            try:
                _qdrant = QdrantClient(url=_get_qdrant_url())
                _qdrant.create_collection(
                    "run_cards",
                    vectors_config={"size": 384, "distance": "Cosine"},
                )
            except Exception:
                _qdrant = None

    def _embed(text: str) -> list[float] | None:
        if _embedder and text:
            return _embedder.encode(text, convert_to_numpy=True).tolist()
        return None

    @_app.post("/ingest/run")
    def ingest_run(payload: dict[str, Any]) -> dict[str, str]:
        """Upsert run summary. Idempotent by case_id."""
        case_id = payload.get("case_id", "")
        if not case_id:
            raise HTTPException(400, "case_id required")

        # Generate run card and postmortem
        sys_path = os.environ.get("PYTHONPATH", "").split(os.pathsep)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if repo_root not in sys_path:
            import sys
            sys.path.insert(0, repo_root)
        try:
            from knowledge.ingest.make_run_card import make_run_card
            from knowledge.ingest.make_postmortem import make_postmortem
            run_card = make_run_card(payload)
            postmortem = make_postmortem(payload)
        except Exception:
            run_card = ""
            postmortem = ""

        with _db_conn() as conn:
            _ensure_schema(conn)
            summary_json = json.dumps(payload)
            if _use_sqlite():
                conn.execute(
                    """
                    INSERT INTO runs (case_id, summary, run_card, postmortem, updated_at)
                    VALUES (?, ?, ?, ?, datetime('now'))
                    ON CONFLICT (case_id) DO UPDATE SET
                        summary = excluded.summary,
                        run_card = excluded.run_card,
                        postmortem = excluded.postmortem,
                        updated_at = datetime('now')
                    """,
                    (case_id, summary_json, run_card, postmortem),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO runs (case_id, summary, run_card, postmortem, updated_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (case_id) DO UPDATE SET
                        summary = EXCLUDED.summary,
                        run_card = EXCLUDED.run_card,
                        postmortem = EXCLUDED.postmortem,
                        updated_at = NOW()
                    """,
                    (case_id, summary_json, run_card, postmortem),
                )
            if not _use_sqlite():
                conn.commit()

        # Store in Qdrant for semantic search
        if _qdrant and run_card:
            vec = _embed(run_card)
            if vec:
                from qdrant_client.models import PointStruct
                _qdrant.upsert(
                    collection_name="run_cards",
                    points=[
                        PointStruct(
                            id=case_id,
                            vector=vec,
                            payload={"case_id": case_id, "run_card": run_card[:1000]},
                        )
                    ],
                )

        return {"status": "ok", "case_id": case_id}

    @_app.get("/runs/top")
    def runs_top(k: int = 10, success: str = "true") -> dict[str, Any]:
        """Top runs by score (for proposer parents)."""
        with _db_conn() as conn:
            _ensure_schema(conn)
            if _use_sqlite():
                success_filter = "AND json_extract(summary, '$.success') = 1" if success == "true" else ""
                cur = conn.execute(
                    f"""
                    SELECT summary FROM runs
                    WHERE 1=1 {success_filter}
                    ORDER BY CAST(json_extract(summary, '$.total_score') AS REAL) ASC
                    LIMIT ?
                    """,
                    (k,),
                )
            else:
                success_filter = "AND summary->>'success' = 'true'" if success == "true" else ""
                cur = conn.execute(
                    f"""
                    SELECT summary FROM runs
                    WHERE 1=1 {success_filter}
                    ORDER BY (summary->>'total_score')::float ASC NULLS LAST
                    LIMIT %s
                    """,
                    (k,),
                )
            rows = cur.fetchall()
        return {"runs": [_row_summary(r) for r in rows]}

    @_app.get("/search/runs")
    def search_runs(q: str = "", k: int = 10) -> dict[str, Any]:
        """Semantic search over run cards."""
        if _qdrant and _embedder and q:
            vec = _embed(q)
            if vec:
                results = _qdrant.search(
                    collection_name="run_cards",
                    query_vector=vec,
                    limit=k,
                )
                case_ids = [r.payload.get("case_id") for r in results if r.payload]
                if case_ids:
                    with _db_conn() as conn:
                        ph = ",".join("?" if _use_sqlite() else "%s" for _ in case_ids)
                        cur = conn.execute(
                            f"SELECT summary FROM runs WHERE case_id IN ({ph})",
                            tuple(case_ids),
                        )
                        rows = cur.fetchall()
                    return {"runs": [_row_summary(r) for r in rows]}
        return {"runs": []}

    @_app.get("/search/papers")
    def search_papers(q: str = "", k: int = 10, tags: str | None = None) -> dict[str, Any]:
        """Semantic search over papers. Stub until papers ingested."""
        return {"papers": []}

    def _row_summary(row: Any) -> dict:
        """Extract summary dict from a row (SQLite Row or tuple)."""
        val = row[0] if hasattr(row, "__getitem__") else getattr(row, "summary", row)
        return json.loads(val) if isinstance(val, str) else val

    @_app.get("/stats/recent")
    def stats_recent(window: int = 30) -> dict[str, Any]:
        """Failure histogram, top runs."""
        with _db_conn() as conn:
            _ensure_schema(conn)
            cur = conn.execute(
                "SELECT summary FROM runs ORDER BY case_id DESC LIMIT " + ("?" if _use_sqlite() else "%s"),
                (window,),
            )
            rows = cur.fetchall()
        runs = [_row_summary(r) for r in rows]
        success_count = sum(1 for r in runs if r.get("success"))
        fail_count = len(runs) - success_count
        reasons: Counter[str] = Counter()
        classes: Counter[str] = Counter()
        for r in runs:
            if not r.get("success"):
                reasons[r.get("failure_reason", "?")[:50]] += 1
                classes[r.get("failure_class", "unknown")] += 1
        return {
            "total": len(runs),
            "success_count": success_count,
            "fail_count": fail_count,
            "fail_rate": fail_count / len(runs) if runs else 0,
            "failure_reasons": dict(reasons.most_common(10)),
            "failure_classes": dict(classes.most_common(10)),
        }

    @_app.post("/brief")
    def post_brief(payload: dict[str, Any]) -> dict[str, Any]:
        """LLM-generated research brief from runs + papers. Requires KB_LLM_* env."""
        from knowledge.services.llm_endpoints import call_brief

        query = payload.get("query", "")
        context = payload.get("context", "general")

        with _db_conn() as conn:
            _ensure_schema(conn)
            if _use_sqlite():
                cur = conn.execute(
                    "SELECT summary FROM runs ORDER BY CAST(json_extract(summary, '$.total_score') AS REAL) ASC LIMIT ?",
                    (15,),
                )
            else:
                cur = conn.execute(
                    "SELECT summary FROM runs ORDER BY (summary->>'total_score')::float ASC NULLS LAST LIMIT %s",
                    (15,),
                )
            rows = cur.fetchall()
            runs = [_row_summary(r) for r in rows]

            cur2 = conn.execute(
                "SELECT summary FROM runs ORDER BY case_id DESC LIMIT " + ("?" if _use_sqlite() else "%s"),
                (30,),
            )
            rows2 = cur2.fetchall()
        stats_runs = [_row_summary(r) for r in rows2]
        success_count = sum(1 for r in stats_runs if r.get("success"))
        fail_count = len(stats_runs) - success_count
        reasons: Counter[str] = Counter()
        classes: Counter[str] = Counter()
        for r in stats_runs:
            if not r.get("success"):
                reasons[r.get("failure_reason", "?")[:50]] += 1
                classes[r.get("failure_class", "unknown")] += 1
        stats = {
            "total": len(stats_runs),
            "fail_rate": fail_count / len(stats_runs) if stats_runs else 0,
            "failure_reasons": dict(reasons.most_common(10)),
            "failure_classes": dict(classes.most_common(10)),
        }

        papers: list[dict] = []
        if _qdrant and query:
            vec = _embed(query)
            if vec:
                try:
                    results = _qdrant.search(
                        collection_name="paper_chunks",
                        query_vector=vec,
                        limit=5,
                    )
                    for r in results:
                        if r.payload:
                            papers.append({
                                "paper_id": r.payload.get("paper_id"),
                                "title": r.payload.get("title", "?"),
                                "text": r.payload.get("chunk_text", ""),
                            })
                except Exception:
                    pass

        return call_brief(runs, stats, papers, query=query, context=context)

    @_app.post("/propose")
    def post_propose(payload: dict[str, Any]) -> dict[str, Any]:
        """LLM-generated mutation actions. Requires KB_LLM_* env."""
        from knowledge.services.llm_endpoints import call_propose

        context = payload.get("context", {})
        policy = payload.get("policy", {})
        batch_size = payload.get("batch_size", 8)
        return call_propose(context, policy, batch_size=batch_size)

    return _app


try:
    app = _create_app()
except ImportError as e:
    app = None
    import sys
    print(f"KB server import failed: {e}", file=sys.stderr)
