#!/usr/bin/env python3
"""
Search PDFs for Zenodo links, download coil optimization data, and compile all coil solutions.

Searches through stellarator coil optimization PDFs (from knowledge/papers/ and
papers_manifest.jsonl) for Zenodo record links. Downloads each record's files,
extracts coil solutions (any JSON with coils/BiotSavart structure), validates
with simsopt when possible, and saves to knowledge/zenodo/{record_id}/{config_name}/coils.json.

Record-specific filtering (final solutions only):
- Input coil files are always ignored (coil_inputs, coil_input.json, input_coils, etc.).
- Augmented Lagrangian / Reactor-scale (14934092): Excludes coil_pareto_plots/SMF/*
  (hash-named Pareto exploration points) and inner_loop/coil_inputs intermediates.
- Single-stage optimization (7655077): Only one final coil set (biot_savart_opt.json
  from results/) with final plasma surface (wout_final.nc). Excludes coil_inputs and
  biot_savart_inner_loop*. Direct JSON files are skipped (zip only).

Metadata is taken from the citing paper so it accurately reflects the publication.

Usage
-----
    cd /path/to/stellcoilbench
    python knowledge/scripts/fetch_zenodo_coils.py [--papers-dir knowledge/papers] [--output knowledge/zenodo]
    python knowledge/scripts/fetch_zenodo_coils.py --manifest knowledge/papers_manifest.jsonl
    python knowledge/scripts/fetch_zenodo_coils.py --limit 50  # Faster: search only first 50 PDFs

Dependencies
-----------
- pymupdf or pypdf for PDF text extraction (pip install pypdf)
- simsopt (for coil validation/serialization; falls back to raw JSON if unavailable)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Zenodo record ID to skip: QUASR (QUAsisymmetric Stellarator Repository) does not
# contain simsopt-style coil JSON; its data format is incompatible.
QUASR_RECORD_ID = "10050656"

# Record IDs with "final-only" filtering (exclude intermediate/checkpoint solutions)
# Augmented Lagrangian paper (Reactor-scale stellarators, Kaptanoglu et al.): exclude
# coil_pareto_plots/SMF/* (hash-named Pareto exploration points) and any path matching
# intermediate patterns like inner_loop, coil_inputs.
AUGMENTED_LAGRANGIAN_RECORD_IDS = frozenset({"14934092"})

# Single-stage optimization (Jorge et al.): only biot_savart_opt.json from results/,
# exclude coil_inputs/ and biot_savart_inner_loop*. Also extract wout_final.nc (final
# VMEC equilibrium surface) alongside each coil solution.
SINGLE_STAGE_RECORD_IDS = frozenset({"7655077"})

# Zenodo URL patterns: record ID is numeric
_ZENODO_RECORD_RE = re.compile(
    r"zenodo\.org/record/(\d+)|"
    r"doi\.org/10\.5281/zenodo\.(\d+)|"
    r"zenodo\.org/records/(\d+)",
    re.IGNORECASE,
)


def _is_coil_json(data: Any) -> bool:
    """Return True if data looks like a simsopt/stellcoilbench coil configuration.

    Accepts any dict that contains coil-related structure: a "coils" key (list of
    curves), or "BiotSavart"/"CurveXYZFourier"/"SerretFrenet" in the serialized form.
    Does not restrict to specific filenames.
    """
    if not isinstance(data, dict):
        return False
    if "coils" in data and isinstance(data["coils"], list):
        return True
    s = json.dumps(data)
    return any(
        marker in s
        for marker in ("BiotSavart", "CurveXYZFourier", "SerretFrenet", "CurveRZFourier")
    )


def _extract_zenodo_ids_from_text(text: str) -> set[str]:
    """Extract unique Zenodo record IDs from text.

    Matches zenodo.org/record/{id}, doi.org/10.5281/zenodo.{id}, and
    zenodo.org/records/{id}.
    """
    ids: set[str] = set()
    for m in _ZENODO_RECORD_RE.finditer(text):
        for g in m.groups():
            if g:
                ids.add(g)
                break
    return ids


def _load_paper_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Load papers manifest JSONL into a dict: paper_id -> paper metadata.

    Returns
    -------
    dict
        Keys are paper IDs (e.g. arxiv_2412_13937). Values include title, authors,
        year, arxiv_id, doi, tags.
    """
    out: dict[str, dict[str, Any]] = {}
    if not manifest_path.exists():
        return out
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                pid = entry.get("id")
                if pid:
                    out[pid] = entry
            except json.JSONDecodeError:
                continue
    return out


def _get_pdf_paths(papers_dir: Path, manifest_path: Path | None) -> list[Path]:
    """Return list of PDF paths to search.

    Uses manifest if provided; otherwise scans papers_dir for *.pdf.
    """
    paths: list[Path] = []
    if manifest_path and manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    rel = entry.get("path", "")
                    if rel.endswith(".pdf"):
                        p = _REPO_ROOT / rel
                        if p.exists():
                            paths.append(p)
                except json.JSONDecodeError:
                    continue
    if not paths and papers_dir.exists():
        paths = sorted(papers_dir.glob("*.pdf"))
    return paths


def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract full text from PDF for Zenodo link search."""
    try:
        from knowledge.ingest.extract_pdf import extract_pdf
    except ImportError:
        sys.path.insert(0, str(_REPO_ROOT))
        from knowledge.ingest.extract_pdf import extract_pdf

    pages, _, _ = extract_pdf(pdf_path)
    return "\n".join(pages)


def _fetch_zenodo_record(record_id: str) -> dict[str, Any] | None:
    """Fetch Zenodo record metadata from API."""
    url = f"https://zenodo.org/api/records/{record_id}"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        print(f"  Warning: Failed to fetch Zenodo record {record_id}: {e}", file=sys.stderr)
        return None


def _download_file(url: str, dest: Path) -> bool:
    """Download file from URL to dest."""
    req = Request(url, headers={"User-Agent": "StellCoilBench/1.0"})
    try:
        with urlopen(req, timeout=120) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.read())
            return True
    except (HTTPError, URLError, OSError) as e:
        print(f"  Warning: Failed to download {url}: {e}", file=sys.stderr)
        return False


def _sanitize_config_name(name: str, max_len: int = 80) -> str:
    """Sanitize a path/name for use as a directory name.

    Keeps alphanumeric, underscore, hyphen. Replaces other chars with underscore.
    Truncates to max_len.
    """
    sanitized = re.sub(r"[^\w\-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized[:max_len] if sanitized else "coils"


def _should_include_coil_path(
    zip_path: str,
    json_filename: str,
    record_id: str,
) -> bool:
    """Return True if this coil JSON should be included (final solutions only for filtered records).

    Globally: exclude input coil files (coil_inputs, coil_input.json, input_coils, etc.).

    For augmented Lagrangian records (14934092): exclude coil_pareto_plots/SMF/* (hash-named
    Pareto exploration points), inner_loop, coil_inputs, and any path with hash-like
    directory names (e.g. ff58810532fc04b19...).

    For single-stage records (7655077): only biot_savart_opt.json from results/;
    exclude coil_inputs/ and biot_savart_inner_loop* (intermediate optimization steps).
    """
    path_lower = zip_path.replace("\\", "/").lower()
    name_lower = json_filename.lower()

    # Exclude input coil files globally (coil_inputs, coil_input.json, input_coils, etc.)
    if "coil_input" in path_lower or "input_coil" in path_lower:
        return False

    if record_id in AUGMENTED_LAGRANGIAN_RECORD_IDS:
        # Exclude Pareto exploration (coil_pareto_plots/SMF, coil_pareto_plots/SMT, etc.)
        if "coil_pareto_plots" in path_lower:
            return False
        if "inner_loop" in path_lower or "coil_inputs" in path_lower:
            return False
        # Exclude any path with hash-like dir (32 hex chars, e.g. ffdc71b0f23b4641915467c4d0188b74)
        if re.search(r"/[a-f0-9]{32}/", path_lower):
            return False
        # Also exclude if parent dir itself is a hash (config_name would be the hash)
        parts = zip_path.replace("\\", "/").split("/")
        parent_dir = parts[-2] if len(parts) >= 2 else ""
        if re.fullmatch(r"[a-f0-9]{32}", parent_dir.lower()):
            return False
        return True

    if record_id in SINGLE_STAGE_RECORD_IDS:
        # Only final coils (biot_savart_opt.json) from results/; exclude coil_inputs and inner_loop
        if "coil_inputs" in path_lower:
            return False
        if "inner_loop" in name_lower:
            return False
        if name_lower != "biot_savart_opt.json":
            return False
        # Must be in results/ subdir (e.g. results/config/... or x/results/config/...)
        if "results/" not in path_lower:
            return False
        return True

    return True


def _extract_all_coils_from_zip(
    zip_path: Path,
    record_id: str,
) -> list[tuple[str, dict[str, Any], list[str]]]:
    """Extract coil JSONs from a zip archive, with record-specific filtering.

    For augmented Lagrangian records: only final solutions (exclude Pareto hash folders,
    inner_loop, coil_inputs). For single-stage: only biot_savart_opt.json from results/,
    and returns associated wout_final.nc paths for extraction.

    Returns
    -------
    list of (config_name, data, extra_zip_paths)
        config_name is sanitized for use as a subdirectory. data is the parsed JSON.
        extra_zip_paths: paths within the zip to extract alongside (e.g. wout_final.nc).
    """
    results: list[tuple[str, dict[str, Any], list[str]]] = []
    seen_data_hashes: set[str] = set()

    try:
        with zipfile.ZipFile(zip_path) as zf:
            all_names = set(zf.namelist())

            for name in zf.namelist():
                if not name.endswith(".json") or "__MACOSX" in name or name.startswith("._"):
                    continue
                p = Path(name)
                if not _should_include_coil_path(name, p.name, record_id):
                    continue
                try:
                    with zf.open(name) as f:
                        data = json.loads(f.read().decode())
                except (json.JSONDecodeError, KeyError):
                    continue
                if not _is_coil_json(data):
                    continue
                # Deduplicate by content hash
                data_str = json.dumps(data, sort_keys=True)
                h = str(hash(data_str))
                if h in seen_data_hashes:
                    continue
                seen_data_hashes.add(h)

                # Config name from path
                parent_name = p.parent.name if p.parent.name else p.stem
                config_name = _sanitize_config_name(parent_name)
                if not config_name:
                    config_name = _sanitize_config_name(p.stem)

                # For single-stage: find wout_final.nc in same result folder
                extra_paths: list[str] = []
                if record_id in SINGLE_STAGE_RECORD_IDS:
                    # Result folder is parent of coils/; wout_final.nc is in result root
                    result_dir = p.parent.parent
                    wout_name = str(result_dir / "wout_final.nc").replace("\\", "/")
                    if wout_name in all_names:
                        extra_paths.append(wout_name)

                # Ensure uniqueness
                used_names = {cn for cn, _, _ in results}
                base = config_name
                c = 1
                while config_name in used_names:
                    config_name = f"{base}_{c}"
                    c += 1
                used_names.add(config_name)
                results.append((config_name, data, extra_paths))

                # Single-stage: only one final coil set with final plasma surface
                if record_id in SINGLE_STAGE_RECORD_IDS:
                    break
    except zipfile.BadZipFile as e:
        print(f"  Warning: Bad zip {zip_path}: {e}", file=sys.stderr)

    return results


def _extract_all_coils_from_direct_json(file_path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Extract coil data from a single JSON file (non-zip).

    Returns a list of one element if the file is a valid coil JSON, else empty.
    """
    try:
        data = json.loads(file_path.read_text())
        if _is_coil_json(data):
            name = _sanitize_config_name(file_path.stem) or "coils"
            return [(name, data)]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _validate_and_save_coils(data: dict[str, Any], out_path: Path) -> bool:
    """Validate coils with simsopt and save to out_path.

    If simsopt is available, loads and re-saves for format normalization.
    If simsopt fails (e.g. version mismatch), saves raw JSON.
    """
    try:
        from simsopt import load, save
    except ImportError:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2))
        return True

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(data, tf, indent=2)
        tmp_path = Path(tf.name)
    try:
        obj = load(str(tmp_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save(obj, str(out_path))
        return True
    except Exception as e:
        print(f"  Warning: simsopt load/save failed: {e}; saving raw JSON", file=sys.stderr)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2))
        return True
    finally:
        tmp_path.unlink(missing_ok=True)


def _process_zenodo_record(
    record_id: str,
    output_dir: Path,
    record_dir: Path,
    paper_sources: list[str],
    paper_manifest: dict[str, dict[str, Any]],
) -> int:
    """Download record files, extract all coil solutions, save to output_dir/record_id/{config}/coils.json.

    Metadata is taken from the citing paper (first in paper_sources) so it accurately
    reflects the publication. Returns the number of coil solutions saved.
    """
    meta = _fetch_zenodo_record(record_id)
    if not meta:
        return 0

    files = meta.get("files", [])
    if not files:
        print(f"  Record {record_id}: no files")
        return 0

    record_out = output_dir / record_id
    record_out.mkdir(parents=True, exist_ok=True)

    # Paper metadata from the citing paper (first source)
    paper_info: dict[str, Any] = {}
    if paper_sources:
        first_paper_id = paper_sources[0]
        paper_info = paper_manifest.get(first_paper_id, {})
    paper_meta = {
        "paper_id": paper_sources[0] if paper_sources else "",
        "paper_title": paper_info.get("title", ""),
        "paper_authors": paper_info.get("authors", []),
        "paper_year": paper_info.get("year"),
        "arxiv_id": paper_info.get("arxiv_id"),
    }

    # Record metadata: Zenodo info + citing paper info
    meta_slim = {
        "zenodo_id": record_id,
        "zenodo_title": meta.get("metadata", {}).get("title", ""),
        "zenodo_doi": meta.get("doi", ""),
        "zenodo_publication_date": meta.get("metadata", {}).get("publication_date", ""),
        "citing_paper": paper_meta,
        "paper_sources": paper_sources,
    }
    (record_out / "record_metadata.json").write_text(json.dumps(meta_slim, indent=2))

    saved_count = 0

    # 1. Direct JSON files (not in zip) — skip for single-stage (we use zip only)
    if record_id not in SINGLE_STAGE_RECORD_IDS:
        for f in files:
            fname = f.get("key", "")
            if not fname.endswith(".json"):
                continue
            # Ignore input coil files
            fname_lower = fname.lower()
            if "coil_input" in fname_lower or "input_coil" in fname_lower:
                continue
            url = f.get("links", {}).get("self", "")
            if not url:
                continue
            dest = record_dir / fname
            if _download_file(url, dest):
                for config_name, data in _extract_all_coils_from_direct_json(dest):
                    coils_out = record_out / config_name / "coils.json"
                    if _validate_and_save_coils(data, coils_out):
                        saved_count += 1
                        print(f"  Record {record_id}: saved {config_name}/coils.json from {fname}")

    # 2. Zip archives: extract all coil JSONs
    for f in files:
        fname = f.get("key", "")
        if not fname.lower().endswith(".zip"):
            continue
        url = f.get("links", {}).get("self", "")
        if not url:
            continue
        dest = record_dir / fname
        if not _download_file(url, dest):
            continue
        coils_list = _extract_all_coils_from_zip(dest, record_id)
        for config_name, data, extra_paths in coils_list:
            config_dir = record_out / config_name
            coils_out = config_dir / "coils.json"
            if _validate_and_save_coils(data, coils_out):
                saved_count += 1
                # Extract extra files (e.g. wout_final.nc for single-stage)
                if extra_paths:
                    try:
                        with zipfile.ZipFile(dest) as zf:
                            for ep in extra_paths:
                                if ep in zf.namelist():
                                    out_file = config_dir / Path(ep).name
                                    out_file.parent.mkdir(parents=True, exist_ok=True)
                                    out_file.write_bytes(zf.read(ep))
                    except (zipfile.BadZipFile, KeyError):
                        pass
                print(f"  Record {record_id}: saved {config_name}/coils.json from {fname}")

    if saved_count == 0:
        print(f"  Record {record_id}: no coil JSON found in {len(files)} file(s)")

    return saved_count


def main() -> int:
    """Search PDFs for Zenodo links, download records, and compile all coil solutions."""
    parser = argparse.ArgumentParser(
        description="Search PDFs for Zenodo links, download coil solutions, compile to knowledge/zenodo/",
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=_REPO_ROOT / "knowledge" / "papers",
        help="Directory containing PDFs to search",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "knowledge" / "papers_manifest.jsonl",
        help="Papers manifest JSONL (paths + metadata for citing papers)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "knowledge" / "zenodo",
        help="Output directory for compiled coils (default: knowledge/zenodo/)",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Temporary directory for Zenodo downloads (default: output/._downloads)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report found Zenodo IDs, do not download",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of PDFs to search (for faster testing)",
    )
    args = parser.parse_args()

    output_dir = args.output.resolve()
    download_dir = args.download_dir or (output_dir / "._downloads")
    download_dir = download_dir.resolve()

    paper_manifest = _load_paper_manifest(args.manifest)

    pdf_paths = _get_pdf_paths(args.papers_dir, args.manifest)
    if args.limit:
        pdf_paths = pdf_paths[: args.limit]
    if not pdf_paths:
        print(
            "No PDFs found in papers directory or manifest. "
            "Run fetch_papers.py to download papers to knowledge/papers/ first.",
            file=sys.stderr,
        )
        return 1

    all_ids: set[str] = set()
    paper_sources: dict[str, list[str]] = {}

    for pdf_path in pdf_paths:
        try:
            text = _extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"Skip {pdf_path.name}: {e}", file=sys.stderr)
            continue
        ids = _extract_zenodo_ids_from_text(text)
        for rid in ids:
            all_ids.add(rid)
            paper_sources.setdefault(rid, []).append(pdf_path.stem)

    # Skip QUASR
    all_ids.discard(QUASR_RECORD_ID)
    if QUASR_RECORD_ID in paper_sources:
        print(f"Skipping QUASR (record {QUASR_RECORD_ID}): incompatible data format")

    if not all_ids:
        print("No Zenodo links found in PDFs.")
        return 0

    print(f"Found {len(all_ids)} Zenodo record(s): {sorted(all_ids)}")
    for rid in sorted(all_ids):
        print(f"  {rid} (from: {', '.join(paper_sources.get(rid, []))})")

    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    total_coils = 0
    manifest_entries: list[dict[str, Any]] = []

    for record_id in sorted(all_ids):
        record_dl = download_dir / record_id
        record_dl.mkdir(parents=True, exist_ok=True)
        count = _process_zenodo_record(
            record_id,
            output_dir,
            record_dl,
            paper_sources.get(record_id, []),
            paper_manifest,
        )
        if count > 0:
            total_coils += count
            configs = sorted((output_dir / record_id).iterdir())
            config_dirs = [c.name for c in configs if c.is_dir()]
            manifest_entries.append({
                "record_id": record_id,
                "coils_paths": [f"{record_id}/{d}/coils.json" for d in config_dirs],
                "coil_count": count,
                "sources": paper_sources.get(record_id, []),
            })

    if manifest_entries:
        (output_dir / "manifest.json").write_text(json.dumps(manifest_entries, indent=2))

    print(f"\nCompiled {total_coils} coil solution(s) from {len(manifest_entries)} record(s) to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
