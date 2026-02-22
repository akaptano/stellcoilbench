#!/usr/bin/env python3
"""
Automated PDF fetcher for stellarator coil optimization literature.

This module discovers and downloads open-access PDFs of papers relevant to
stellarator coil design, quasisymmetry, and related fusion research. It
searches arXiv (primary, always free) and optionally Semantic Scholar,
using Unpaywall to resolve publisher-restricted PDFs when available.

Papers are filtered by title and abstract to exclude economics, game theory,
and other non-fusion "equilibrium" literature that commonly matches generic
search terms. Only papers containing stellarator-specific keywords
(stellarator, quasisymmetry, FOCUS, simsopt, etc.) in title or abstract
are retained.

Usage
-----
    cd /path/to/stellcoilbench
    python knowledge/scripts/fetch_papers.py [--max-per-source 100] [--min-year 2010]
    python knowledge/scripts/fetch_papers.py --s2  # Also fetch from Semantic Scholar

Dependencies
------------
- arxiv, requests (pip install arxiv requests)
- Optional: SEMANTIC_SCHOLAR_API_KEY for higher S2 rate limits
- Optional: UNPAYWALL_EMAIL for Unpaywall API (recommended)

Output
------
- PDFs saved to knowledge/papers/
- Manifest written to knowledge/papers_manifest.jsonl (one JSON object per line)
- Duplicate papers (same arXiv id or DOI) are never downloaded twice
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import re
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Default search queries for stellarator coil optimization literature
# Queries are kept specific to avoid economics/game-theory "equilibrium" papers
DEFAULT_QUERIES = [
    "stellarator coil optimization",
    "stellarator coils",
    "quasi-symmetric stellarator",
    "quasisymmetry stellarator",
    "stellarator equilibrium coils",
    "coil winding surface stellarator",
    "landreman paul stellarator",
    "FOCUS stellarator coils",
    "simsopt stellarator",
    "DESC stellarator",
    "winding surface optimization"
    "stellarator permanent magnet optimization"
    "Thea Energy"
    "Proxima Fusion"
    "Type One Energy"
]

# Title must contain at least one of these (case-insensitive) to be accepted
_TITLE_REQUIRED_TERMS = [
    "stellarator",
    "near-axis expansion",
    "quasisymmetry",
    "quasi-symmetry",
    "simsopt",
    "winding surface",
    "coil optimization",
]

# Reject if title contains these (economics, game theory, astronomy, etc.)
_TITLE_EXCLUDE_PATTERNS = [
    # Astronomy: "stellar" without "stellarator" (stellar populations, stellar ages)
    r"\bstellar\s+ages?\b",
    r"\bstellar\s+population",
    r"\bstellar\s+radii\b",
    r"\bstellar\s+metallicity\b",
    r"\bstellar\s+evolution\b",
    r"\bstellar\s+structure\b",
    r"\bstellar\s+atmosphere",
    r"\bstellar\s+wind\b",
    r"\bstellar\s+disk\b",
    r"\bstellar\s+halo\b",
    r"\bstellar\s+cluster",
    r"\bstellar\s+formation\b",
    r"\bstellar\s+abundance",
    r"\bstellar\s+kinematic",
    r"\bstellar\s+feedback\b",
    r"\bstellar\s+mass\b",
    r"\bstellar\s+luminosity\b",
    r"\bstellar\s+rotation\b",
    r"\bstellar\s+activity\b",
    r"\bstellar\s+parameter",
    r"\bstellar\s+model",
    r"\bstellar\s+code\b",
    r"\bgeneral equilibrium\b",
    r"\bsequential equilibrium\b",
    r"\bnash equilibrium\b",
    r"\bradner equilibrium\b",
    r"\bgame theory\b",
    r"\bchemical equilibrium\b",
    r"\bthermodynamic equilibrium\b",
    r"\bclimate change\b",
    r"\bstellar radiation\b",  # astronomy, not fusion
    r"\bdust depletion\b",
    r"\bnon-equilibrium\s+diffuse",  # materials science
    r"\bequivariant equilibrium\b",  # ML
    r"\bconstrained ordered equilibrium\b",
    r"\bcursed sequential\b",
    r"\bconditional strategy equilibrium\b",
    r"\bcomputability of equilibrium\b",
    r"\bconsiderate equilibrium\b",
    r"\brecession phenomenon\b",
    r"\boptimization induced equilibrium\b",  # ML, not fusion
]


def _is_stellarator_relevant(title: str, abstract: str = "") -> bool:
    """Return True if paper title or abstract indicates stellarator/fusion relevance.

    A paper passes if:
    - At least one required term (stellarator, quasisymmetry, FOCUS, etc.) appears
      in the title or abstract.
    - The title does not match any exclude pattern (e.g. general equilibrium,
      game theory, climate change) that indicates non-fusion content.

    Parameters
    ----------
    title : str
        Paper title.
    abstract : str, optional
        Paper abstract. Used to find required terms when title is generic.

    Returns
    -------
    bool
        True if the paper should be included in the stellarator corpus.
    """
    t = (title or "").lower()
    a = (abstract or "").lower()
    combined = f"{t} {a}"
    # Must contain at least one required term (in title or abstract)
    # Use word boundary for "stellarator" to avoid matching astronomy "stellar" papers
    def _has_term(term: str, text: str) -> bool:
        if term == "stellarator":
            return bool(re.search(r"\bstellarator\b", text, re.I))
        return term in text
    if not any(_has_term(term, combined) for term in _TITLE_REQUIRED_TERMS):
        return False
    # Reject if title matches exclude patterns (economics, game theory, etc.)
    for pat in _TITLE_EXCLUDE_PATTERNS:
        if re.search(pat, t, re.I):
            return False
    return True


def _sanitize_filename(s: str) -> str:
    """Make a string safe for use as a filesystem filename.

    Strips non-alphanumeric characters except spaces, hyphens, and periods;
    replaces spaces with underscores; truncates to 80 characters.
    """
    s = re.sub(r"[^\w\s\-\.]", "", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s[:80] if s else "paper"


def _download_pdf(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download a PDF from a URL to the given destination path.

    Parameters
    ----------
    url : str
        Direct URL to the PDF.
    dest : Path
        Local path to write the file.
    timeout : int, optional
        Request timeout in seconds (default 60).

    Returns
    -------
    bool
        True if download succeeded and file has at least 500 bytes; False otherwise.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StellCoilBench/1.0 (fetch_papers)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()
            if len(content) < 500:
                return False
            dest.write_bytes(content)
            return True
    except Exception:
        return False


def _is_direct_pdf_url(url: str) -> bool:
    """Check if a URL points to a direct PDF (not an HTML landing page).

    Returns True if the path ends with .pdf, contains /pdf/, or is an arxiv.org/pdf URL.
    """
    if not url:
        return False
    path = urlparse(url).path.lower()
    return path.endswith(".pdf") or "/pdf/" in path or "arxiv.org/pdf" in url


def _extract_arxiv_id_from_url(url: str) -> str | None:
    """Extract arXiv identifier from an arXiv PDF URL.

    Parameters
    ----------
    url : str
        URL such as https://arxiv.org/pdf/1602.04867.pdf.

    Returns
    -------
    str | None
        Identifier with dots replaced by underscores (e.g. 1602_04867), or None.
    """
    if not url or "arxiv.org" not in url:
        return None
    # e.g. https://arxiv.org/pdf/1602.04867.pdf or .../1602.04867v2.pdf
    m = re.search(r"arxiv\.org/pdf/(\d+\.\d+)(?:v\d+)?", url, re.I)
    if m:
        return m.group(1).replace(".", "_")
    return None


def _ensure_arxiv() -> bool:
    """Ensure the arxiv Python package is available; install via pip if missing.

    Returns
    -------
    bool
        True if arxiv can be imported; False on failure (prints to stderr).
    """
    try:
        import arxiv  # noqa: F401
        return True
    except ImportError:
        pass
    print("Installing arxiv and requests...", file=sys.stderr)
    import subprocess
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "arxiv", "requests", "-q"],
            check=True,
            capture_output=True,
            timeout=120,
        )
        import arxiv  # noqa: F401
        return True
    except Exception as e:
        print(f"Failed to install arxiv: {e}. Run: pip install arxiv requests", file=sys.stderr)
        return False


def fetch_from_arxiv(
    queries: list[str],
    output_dir: Path,
    max_per_query: int = 500,
    delay_seconds: float = 3.0,
    min_year: int | None = 1990,
    title_filter: bool = True,
) -> list[dict[str, Any]]:
    """Search arXiv for stellarator-related papers and download their PDFs.

    For each query, searches arXiv with a date filter, applies title/abstract
    relevance filtering (when enabled), and downloads PDFs. Skips duplicates
    by arXiv id. Results are sorted by relevance.

    Parameters
    ----------
    queries : list[str]
        Search query strings (e.g. "stellarator coil optimization").
    output_dir : Path
        Directory to save PDFs (e.g. knowledge/papers/).
    max_per_query : int, optional
        Maximum results per query (default 15).
    delay_seconds : float, optional
        Delay between API requests to avoid rate limits (default 3.0).
    min_year : int | None, optional
        Only include papers from this year onward; None to disable (default 2010).
    title_filter : bool, optional
        If True, only keep papers with stellarator-relevant title/abstract (default True).

    Returns
    -------
    list[dict]
        Manifest entries with id, path, title, authors, year, arxiv_id, tags.
    """
    if not _ensure_arxiv():
        return []
    import arxiv

    output_dir.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(delay_seconds=delay_seconds, num_retries=3)
    seen_ids: set[str] = set()
    entries: list[dict[str, Any]] = []

    # Date filter: submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]
    date_filter = ""
    if min_year is not None:
        date_filter = f" AND submittedDate:[{min_year}01010000 TO 209912312359]"

    for query in queries:
        full_query = f"({query}){date_filter}" if date_filter else query
        search = arxiv.Search(query=full_query, max_results=max_per_query, sort_by=arxiv.SortCriterion.Relevance)
        for r in client.results(search):
            aid = getattr(r, "get_short_id", lambda: "")()
            if not aid:
                aid = (r.entry_id or "").split("/")[-1]
            if not aid:
                continue
            aid_clean = re.sub(r"v\d+$", "", str(aid)).replace(".", "_")  # e.g. 2101_12345
            if aid_clean in seen_ids:
                continue
            # Filter by title/abstract: must be stellarator/fusion relevant
            abstract = getattr(r, "summary", None) or ""
            if title_filter and not _is_stellarator_relevant(r.title or "", abstract):
                continue
            # Filter by year (API filter may not catch all)
            if min_year is not None and r.published and r.published.year < min_year:
                continue
            seen_ids.add(aid_clean)

            dest = output_dir / f"arxiv_{aid_clean}.pdf"
            if dest.exists():
                rel = dest.relative_to(_REPO_ROOT)
                entries.append({
                    "id": f"arxiv_{aid_clean}",
                    "path": str(rel),
                    "title": r.title or "Unknown",
                    "authors": [a.name for a in (r.authors or [])],
                    "year": r.published.year if r.published else None,
                    "doi": None,
                    "arxiv_id": aid,
                    "tags": ["stellarator", "arxiv"],
                })
                continue

            try:
                r.download_pdf(filename=str(dest))
                if dest.exists():
                    rel = dest.relative_to(_REPO_ROOT)
                    entries.append({
                        "id": f"arxiv_{aid_clean}",
                        "path": str(rel),
                        "title": r.title or "Unknown",
                        "authors": [a.name for a in (r.authors or [])],
                        "year": r.published.year if r.published else None,
                        "doi": None,
                        "arxiv_id": aid,
                        "tags": ["stellarator", "arxiv"],
                    })
                    print(f"Downloaded: {r.title[:60]}...")
            except Exception as e:
                print(f"Skip {aid}: {e}", file=sys.stderr)
        time.sleep(delay_seconds)

    return entries


# S2 rate limit: 100 req/5min without API key. One search per query; 5s between requests.
S2_DELAY_SECONDS = 5.0


def fetch_from_semantic_scholar(
    queries: list[str],
    output_dir: Path,
    max_per_query: int = 10,
    unpaywall_email: str | None = None,
    min_year: int | None = 1990,
    existing_arxiv_ids: set[str] | None = None,
    title_filter: bool = True,
) -> list[dict[str, Any]]:
    """Search Semantic Scholar for open-access PDFs of stellarator papers.

    Skips papers that have an arXiv id (those are fetched from arXiv instead).
    Uses openAccessPdf when available; falls back to Unpaywall for DOI resolution
    when the publisher link is not a direct PDF. Rate-limited (~100 req/5min
    without API key).

    Parameters
    ----------
    queries : list[str]
        Search query strings.
    output_dir : Path
        Directory to save PDFs.
    max_per_query : int, optional
        Maximum results per query (default 10).
    unpaywall_email : str | None, optional
        Email for Unpaywall API (required for DOI→PDF resolution).
    min_year : int | None, optional
        Only include papers from this year onward (default 2010).
    existing_arxiv_ids : set[str] | None, optional
        arXiv ids already fetched; skip S2 papers that point to these.
    title_filter : bool, optional
        If True, only keep stellarator-relevant papers (default True).

    Returns
    -------
    list[dict]
        Manifest entries with id, path, title, authors, year, doi, tags.
    """
    try:
        import requests
    except ImportError:
        print("Install requests: pip install requests", file=sys.stderr)
        return []

    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    output_dir.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    seen_manifest_ids: set[str] = set()  # avoid duplicate manifest entries (same DOI from different S2 papers)
    entries: list[dict[str, Any]] = []
    have_arxiv: set[str] = set(existing_arxiv_ids) if existing_arxiv_ids else set()

    for query in queries:
        time.sleep(S2_DELAY_SECONDS)
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": query, "limit": min(max_per_query, 100), "fields": "title,authors,year,abstract,openAccessPdf,externalIds"}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            print(f"S2 search failed for '{query}': {e}", file=sys.stderr)
            continue

        for p in data.get("data", []):
            pid = p.get("paperId", "")
            if not pid or pid in seen_ids:
                continue
            title = p.get("title", "Unknown")
            abstract = p.get("abstract", "") or ""
            if title_filter and not _is_stellarator_relevant(title, abstract):
                continue
            authors = [a.get("name", "") for a in p.get("authors", []) if a.get("name")]
            year = p.get("year")
            if min_year is not None and year is not None and year < min_year:
                continue
            ext = p.get("externalIds", {})
            arxiv_id = ext.get("ArXiv")
            doi = ext.get("DOI")
            oa = p.get("openAccessPdf")

            # Prefer arXiv if present (we'll get it from arXiv source)
            if arxiv_id:
                seen_ids.add(pid)
                continue

            pdf_url = None
            if oa and "url" in oa:
                oa_url = oa.get("url", "")
                if _is_direct_pdf_url(oa_url):
                    pdf_url = oa_url
                elif doi and unpaywall_email:
                    pdf_url = _unpaywall_resolve(doi, unpaywall_email)

            if not pdf_url:
                seen_ids.add(pid)
                continue

            # Skip if PDF is from arXiv and we already have it (avoid duplicate download)
            arxiv_from_url = _extract_arxiv_id_from_url(pdf_url)
            if arxiv_from_url and (arxiv_from_url in have_arxiv or (output_dir / f"arxiv_{arxiv_from_url}.pdf").exists()):
                seen_ids.add(pid)
                continue

            pdf_id = (doi.replace("/", "_") if doi else pid)[:40]
            safe_id = _sanitize_filename(pdf_id)
            manifest_id = f"s2_{safe_id}"
            dest = output_dir / f"{manifest_id}.pdf"
            if dest.exists():
                if manifest_id not in seen_manifest_ids:
                    rel = dest.relative_to(_REPO_ROOT)
                    entries.append({
                        "id": manifest_id,
                        "path": str(rel),
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "doi": doi,
                        "tags": ["stellarator", "semantic_scholar"],
                    })
                    seen_manifest_ids.add(manifest_id)
                seen_ids.add(pid)
                continue

            if _download_pdf(pdf_url, dest):
                if manifest_id not in seen_manifest_ids:
                    rel = dest.relative_to(_REPO_ROOT)
                    entries.append({
                        "id": manifest_id,
                        "path": str(rel),
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "doi": doi,
                        "tags": ["stellarator", "semantic_scholar"],
                    })
                    seen_manifest_ids.add(manifest_id)
                seen_ids.add(pid)
                print(f"Downloaded (S2): {title[:60]}...")

    return entries


def _unpaywall_resolve(doi: str, email: str) -> str | None:
    """Resolve a DOI to an open-access PDF URL via the Unpaywall API.

    Parameters
    ----------
    doi : str
        Document object identifier (e.g. "10.1234/example").
    email : str
        Email for Unpaywall API (required by their terms of use).

    Returns
    -------
    str | None
        URL to the PDF if found; None otherwise.
    """
    try:
        import requests
    except ImportError:
        return None
    url = f"https://api.unpaywall.org/v2/{doi}"
    try:
        r = requests.get(url, params={"email": email}, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        loc = data.get("best_oa_location") or data.get("oa_locations", [{}])[0]
        return loc.get("url_for_pdf") or loc.get("url")
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        nargs="+",
        default=DEFAULT_QUERIES,
        help="Search queries (default: stellarator-related)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "knowledge" / "papers",
        help="Directory to save PDFs",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "knowledge" / "papers_manifest.jsonl",
        help="Path to papers_manifest.jsonl",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=100,
        help="Max papers per source per query (default 100 → 50 per query; ~200–400 unique)",
    )
    parser.add_argument(
        "--s2",
        action="store_true",
        help="Also fetch from Semantic Scholar (rate-limited; may hit 429 without API key)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to manifest instead of overwriting",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2010,
        metavar="YEAR",
        help="Only include papers from this year onward (default: 2010). Use 0 to disable.",
    )
    parser.add_argument(
        "--no-title-filter",
        action="store_true",
        help="Disable title filtering (include all search results; may add non-stellarator papers)",
    )
    args = parser.parse_args()

    unpaywall_email = os.environ.get("UNPAYWALL_EMAIL", "stellcoilbench@example.com")
    min_year = args.min_year if args.min_year > 0 else None

    all_entries: list[dict[str, Any]] = []
    existing_ids: set[str] = set()
    existing_arxiv_ids: set[str] = set()

    if args.append and args.manifest.exists():
        for line in args.manifest.read_text().strip().splitlines():
            if line.strip():
                try:
                    ent = json.loads(line)
                    existing_ids.add(ent.get("id", ""))
                    if ent.get("id", "").startswith("arxiv_"):
                        existing_arxiv_ids.add(ent["id"].replace("arxiv_", ""))
                except Exception:
                    pass

    title_filter = not args.no_title_filter
    if title_filter:
        print("Filter: only stellarator/fusion-relevant papers (title or abstract; use --no-title-filter to disable)")

    # 1. arXiv (primary, always free)
    print("Fetching from arXiv...")
    arxiv_entries = fetch_from_arxiv(
        args.queries,
        args.output_dir,
        max_per_query=args.max_per_source // 2,
        min_year=min_year,
        title_filter=title_filter,
    )
    for e in arxiv_entries:
        if e["id"] not in existing_ids:
            all_entries.append(e)
            existing_ids.add(e["id"])
        existing_arxiv_ids.add(e["id"].replace("arxiv_", ""))

    # Also include arxiv papers already on disk (from prior runs)
    for f in args.output_dir.glob("arxiv_*.pdf"):
        existing_arxiv_ids.add(f.stem.replace("arxiv_", ""))

    # 2. Semantic Scholar (opt-in; rate-limited ~100 req/5min without API key)
    if args.s2:
        print("Fetching from Semantic Scholar (open-access only)...")
        s2_entries = fetch_from_semantic_scholar(
            args.queries,
            args.output_dir,
            max_per_query=args.max_per_source // 2,
            unpaywall_email=unpaywall_email,
            min_year=min_year,
            existing_arxiv_ids=existing_arxiv_ids,
            title_filter=title_filter,
        )
        for e in s2_entries:
            if e["id"] not in existing_ids:
                all_entries.append(e)
                existing_ids.add(e["id"])

    # Merge with existing manifest if appending
    if args.append and args.manifest.exists():
        for line in args.manifest.read_text().strip().splitlines():
            if line.strip():
                try:
                    ent = json.loads(line)
                    if ent.get("id") not in {e["id"] for e in all_entries}:
                        all_entries.insert(0, ent)
                except Exception:
                    pass

    # Write manifest
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w") as f:
        for e in all_entries:
            f.write(json.dumps(e) + "\n")

    print(f"Manifest updated: {len(all_entries)} papers in {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
