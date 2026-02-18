#!/usr/bin/env python3
"""
API client for the StellCoilBench Knowledge Base.

Provides a Python interface to the KB server's REST API for ingesting run
summaries, searching runs and papers, fetching stats, and calling LLM-powered
endpoints (brief, propose).

Usage
-----
    from knowledge.services.kb_client import KBClient
    client = KBClient(base_url="http://localhost:8000", token=None)
    client.ingest_run(summary)
    runs = client.search_runs(q="coil separation", k=5)
    brief = client.brief(query="What surfaces work best?")
    actions = client.propose(context=ctx, policy=policy, batch_size=8)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


class KBClient:
    """HTTP client for the StellCoilBench Knowledge Base API.

    Supports both requests (if installed) and urllib. All methods return
    parsed JSON. Raises on HTTP errors.
    """

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _post(self, path: str, data: dict | list) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode("utf-8")
        if _HAS_REQUESTS:
            r = requests.post(
                url,
                data=body,
                headers=self._headers(),
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json() if r.content else {}
        req = urllib.request.Request(
            url,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}

    def _get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{qs}"
        if _HAS_REQUESTS:
            r = requests.get(
                url,
                headers=self._headers(),
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json() if r.content else {}
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}

    def ingest_run(self, summary: dict[str, Any]) -> dict[str, Any]:
        """POST /ingest/run — upsert a run summary (idempotent by case_id)."""
        return self._post("/ingest/run", summary)

    def search_runs(
        self,
        q: str,
        k: int = 10,
        success: bool | None = None,
        failure_class: str | None = None,
    ) -> list[dict[str, Any]]:
        """GET /search/runs — semantic search over run cards."""
        params: dict[str, str] = {"q": q, "k": str(k)}
        if success is not None:
            params["filter"] = f"success:{str(success).lower()}"
        if failure_class:
            params["filter"] = params.get("filter", "") + f",failure_class:{failure_class}"
        out = self._get("/search/runs", params)
        return out.get("runs", [])

    def search_papers(
        self,
        q: str,
        k: int = 10,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """GET /search/papers — semantic search over paper chunks."""
        params: dict[str, str] = {"q": q, "k": str(k)}
        if tags:
            params["tags"] = ",".join(tags)
        out = self._get("/search/papers", params)
        return out.get("papers", [])

    def stats_recent(self, window_days: int = 30) -> dict[str, Any]:
        """GET /stats/recent — failure histogram, top runs, etc."""
        return self._get("/stats/recent", {"window": str(window_days)})

    def get_top_runs(self, k: int = 10, success_only: bool = True) -> list[dict[str, Any]]:
        """GET /runs/top — top feasible runs by score (for proposer parents)."""
        params: dict[str, str] = {"k": str(k)}
        if success_only:
            params["success"] = "true"
        out = self._get("/runs/top", params)
        return out.get("runs", [])

    def brief(
        self,
        query: str = "",
        context: str = "general",
    ) -> dict[str, Any]:
        """POST /brief — Generate LLM research brief from runs and papers.

        Returns a 2–4 paragraph summary of recent optimization results,
        failure modes, and literature insights.
        """
        return self._post("/brief", {"query": query, "context": context})

    def propose(
        self,
        context: dict[str, Any],
        policy: dict[str, Any],
        batch_size: int = 8,
    ) -> dict[str, Any]:
        """POST /propose — Generate LLM mutation/exploration actions.

        Returns a list of actions (mutate/explore) for the next CI batch.
        """
        return self._post("/propose", {
            "context": context,
            "policy": policy,
            "batch_size": batch_size,
        })


def main() -> int:
    """CLI: ingest a summary file or run a search."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000", help="KB base URL")
    parser.add_argument("--token", default=None, help="Bearer token")
    sub = parser.add_subparsers(dest="cmd")
    ing = sub.add_parser("ingest")
    ing.add_argument("summary_file", type=Path, help="Path to summary.json")
    sr = sub.add_parser("search-runs")
    sr.add_argument("query", help="Search query")
    sr.add_argument("-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()
    client = KBClient(base_url=args.url, token=args.token)
    if args.cmd == "ingest":
        summary = json.loads(args.summary_file.read_text())
        client.ingest_run(summary)
        print("Ingested:", summary.get("case_id", "?"))
    elif args.cmd == "search-runs":
        runs = client.search_runs(q=args.query, k=args.k)
        print(json.dumps(runs, indent=2))
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
