# Persistent Knowledge Layer — Design & Integration Assessment

This document assesses what it would take to add a persistent knowledge layer (vector DB + run DB) to StellCoilBench, per the design outlined in the user query. It maps the proposed architecture to the existing repo and identifies concrete implementation steps.

---

## 1. Current State vs. Proposed Design

### 1.1 What Already Exists

| Proposed Component | Current State | Notes |
|-------------------|---------------|-------|
| **Run summaries** | `cases/done/<case_id>/summary.json` | Written by `run-ci-case` (success or failure) |
| **Summary fields** | `case_id`, `success`, `total_score`, `iterations_used`, `walltime_sec`, `failure_reason`, `failure_class`, `metrics`, `case_config`, `tags`, `parent_ids`, `random_seed` | `failure_class` is `type(exc).__name__` on failure; empty on success |
| **Proposer context** | `tools/build_context.py` | Reads `cases/done/*/summary.json`, computes failure stats, top parents, config hashes |
| **Config hash** | `_config_hash()` in build_context | Used for novelty/dedup; not stored in summary |
| **CI output** | `cases/done/` committed to git | Autopilot workflow commits after each batch |
| **Guardrails** | `policy/proposer_policy.yaml` | `failure_class`, `sliding_window`, `critical_failure_classes` already used |

### 1.2 Gaps in summary.json (for KB queryability)

The design calls for these fields in `summary.json`; some are missing or need refinement:

| Field | Status | Action |
|-------|--------|--------|
| `failure_class` | Present | Canonicalize: map raw exception names → `vmec_nonconverged`, `nan_in_objective`, `timeout`, `line_search_fail`, `min_sep_violation`, etc. |
| `margins` | Missing | Add dict of constraint margins (e.g. `cc_margin`, `cs_margin`, …) from `metrics` vs thresholds |
| `coil_fingerprint` | Missing | Add hash of coil geometry (e.g. Fourier coefficients) for solution dedup |
| `config_hash` | Missing | Add `_config_hash(case_config)` for dedup in KB and proposer |

### 1.3 Metrics Already Available

From `results_dict` (e.g. `optimize_coils` returns), the summary already gets `metrics` with:

- `final_min_cc_separation`, `final_min_cs_separation`
- `cc_threshold`, `cs_threshold`, `msc_threshold`, `curvature_threshold`, etc.
- `final_squared_flux`, `iterations_used`, `walltime_sec`, `BdotN`, etc.

So margins can be derived as `(current_value - threshold)` or `(current_value / threshold - 1)` for soft constraints.

---

## 2. Repo Layout for Knowledge Layer

### 2.1 Proposed `knowledge/` Structure

```
knowledge/
  papers/                     # (optional) small curated PDFs; big corpora via Git LFS or external bucket
  papers_manifest.jsonl        # list of paper ids + URLs/paths + tags
  run_schema.json              # defines summary.json fields your CI must output
  ingest/
    extract_pdf.py             # PDF→(pages, text) + chunk→page map
    chunk.py                   # chunking policy
    make_run_card.py           # run summary→short text card
    make_postmortem.py         # deterministic postmortem (or LLM optional)
  services/
    kb_client.py               # API client to your KB service (desktop or cloud)
  docs/
    kb_design.md               # this file
```

This fits cleanly alongside the existing repo. No changes to `cases/`, `policy/`, or `tools/` structure beyond what’s needed for the new fields.

### 2.2 CI Output Location

Current: `cases/done/<case_id>/summary.json` — already committed.

Optional: `cases/done/<case_id>/fingerprint.json` (tiny, e.g. `{"coil_hash": "..."}`) — can be added as a separate file for KB dedup.

---

## 3. KB Storage: Cloud vs Desktop

### 3.1 Cloud KB (recommended for scale)

- **Postgres** (runs table) + **Qdrant** (or managed vector store) + **S3/GCS** for PDFs
- CI pushes run summaries via `curl -H "Authorization: Bearer $KB_TOKEN" -d @summary.json https://kb.example.com/ingest/run`
- **Requires**: KB service deployed and reachable from CI; `KB_TOKEN` secret

### 3.2 Desktop KB (recommended for local control)

- **Postgres + Qdrant** on desktop (Docker)
- **FastAPI** service with `/ingest/run`, `/search/papers`, `/search/runs`, etc.
- CI **cannot** reach your laptop directly → **pull-based**: CI commits `summary.json` to git; desktop runs `git pull` + local ingest script.

### 3.3 Hybrid

- Cloud KB for ingestion + retrieval
- Desktop mirror/cache for offline use

### 3.4 Recommendation for StellCoilBench

**Desktop KB** is used. Postgres + Qdrant run via Docker. CI commits `cases/done/` to git; a local script (`knowledge/scripts/kb_updater.py`) runs `git pull` and ingests summaries into the local KB. The proposer can use `--kb-url http://localhost:8000` when the KB server is running.

---

## 4. API Contract (KB Service)

Whether cloud or desktop, a minimal contract:

```yaml
# Ingest
POST /ingest/run     # body: summary.json (+ optional fingerprint.json)
POST /ingest/paper   # body: PDF bytes or URL + metadata
POST /ingest/paper_chunks  # optional, if pre-extracted

# Query
GET /search/papers?q=...&k=...&tags=...
GET /search/runs?q=...&k=...&filter=success:true,failure_class:...
GET /stats/recent?window=30

# Agent helpers (optional)
POST /brief   # → research brief JSON with citations
POST /propose # → mutation actions JSON (not raw configs)
```

---

## 5. CI Integration

### 5.1 Workflow 1 — Run Cases (existing)

Already:

- Writes `cases/done/<case_id>/summary.json`
- Commits `cases/done/` in the autopilot workflow

**Additions:**

1. **Enrich summary.json** (see §6) with `config_hash`, `margins`, `coil_fingerprint`, canonical `failure_class`.
2. **Optional**: `fingerprint.json` in `cases/done/<case_id>/`.
3. **Cloud KB**: If `KB_TOKEN` is set, add a step after commit:

   ```yaml
   - name: Upload run to KB
     if: env.KB_TOKEN != ''
     run: |
       for f in cases/done/*/summary.json; do
         if [ -f "$f" ]; then
           curl -sS -H "Authorization: Bearer $KB_TOKEN" -d @"$f" "$KB_URL/ingest/run" || true
         fi
       done
   ```

   (Or only upload new runs since last ingest to avoid duplicates.)

### 5.2 Workflow 2 — Ingest Knowledge (new)

**Trigger:**

- `when: cases/done/**/summary.json changes` (or `knowledge/papers_manifest.jsonl` changes)
- Or nightly schedule

**Actions:**

1. Find new run summaries since last ingest (or all if first run).
2. Generate run cards + postmortems via `knowledge/ingest/make_run_card.py` and `make_postmortem.py`.
3. Push to KB (Postgres + Qdrant).
4. Idempotent: same `run_id` (case_id) → upsert.

**Note:** Desktop KB uses a separate local script that runs `git pull` and ingests; no new workflow needed for that path.

---

## 6. Implementation Priority (Low Risk, High ROI)

### Phase 1 — Enrich summary.json (no KB yet)

1. **Add `config_hash`** — `_config_hash(case_config)` in `cli.py` when writing summary.
2. **Canonicalize `failure_class`** — map `TimeoutError`, `ValueError` (e.g. NaN), `VMECException`, etc. to `timeout`, `nan_in_objective`, `vmec_nonconverged`, etc.
3. **Add `margins`** — dict of `(metric - threshold)` or `(metric / threshold - 1)` for soft constraints (cc, cs, msc, curvature, flux, force, torque).
4. **Add `coil_fingerprint`** — hash of coil Fourier coefficients (or similar) from `results_dict` when available. Write to `summary.json` or `fingerprint.json`.

**Files to touch:**

- `src/stellcoilbench/cli.py` — `run_ci_case` and `_write_ci_summary`
- `knowledge/run_schema.json` — schema for `summary.json` (for documentation and validation)

### Phase 2 — Run card + postmortem generators

1. **`knowledge/ingest/make_run_card.py`** — turn summary into 10–20 line text card (success/fail, score, key metrics, tightest margins, notable settings).
2. **`knowledge/ingest/make_postmortem.py`** — rule-based postmortem:
   - `min_sep_violation` → suggest `sep penalty ↑`, `step scale ↓`
   - `line_search_fail` → suggest `trust region or step scale ↓`, restart plan
   - etc.

Store these as text for later embedding. No KB yet.

### Phase 3 — Desktop KB + ingest

1. **`knowledge/` folder** — layout as above.
2. **`knowledge/run_schema.json`** — JSON schema for `summary.json`.
3. **FastAPI service** — `POST /ingest/run`, `GET /search/runs`, `GET /stats/recent`.
4. **Postgres** — runs table (config, score, metrics, margins, failure_class, coil_fingerprint, config_hash, git_commit).
5. **Qdrant** — run cards + postmortems as embeddings.
6. **Local ingest script** — `git pull` → `cases/done/*/summary.json` → `make_run_card` → `make_postmortem` → POST to local KB.

### Phase 4 — Proposer integration

1. **`build_context`** — optionally call KB for:
   - `top feasible runs` (instead of or in addition to local `cases/done/`)
   - `failure histogram`, `common failures`
   - `similar runs` by embedding search
2. **`propose_batch`** — use KB context when available; fall back to `build_context(done_dir)` when KB is offline.

### Phase 5 — Papers + LLM (optional)

1. **`knowledge/papers_manifest.jsonl`** — paper IDs, URLs, tags.
2. **`knowledge/ingest/extract_pdf.py`**, **`chunk.py`** — PDF → chunks + embeddings.
3. **`POST /brief`** — LLM-generated research brief with citations.
4. **`POST /propose`** — LLM-generated mutation actions (strict JSON).

---

## 7. Sample `run_schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CI Run Summary",
  "description": "Schema for cases/done/<case_id>/summary.json",
  "type": "object",
  "required": ["case_id", "success", "total_score", "iterations_used", "walltime_sec", "case_config"],
  "properties": {
    "case_id": { "type": "string" },
    "success": { "type": "boolean" },
    "total_score": { "type": "number" },
    "iterations_used": { "type": "integer" },
    "walltime_sec": { "type": "number" },
    "failure_reason": { "type": "string" },
    "failure_class": { "type": "string", "enum": ["", "timeout", "vmec_nonconverged", "nan_in_objective", "line_search_fail", "min_sep_violation", "validation", "unknown"] },
    "config_hash": { "type": "string" },
    "margins": {
      "type": "object",
      "additionalProperties": { "type": "number" }
    },
    "coil_fingerprint": { "type": "string" },
    "metrics": { "type": "object" },
    "case_config": { "type": "object" },
    "tags": { "type": "array", "items": { "type": "string" } },
    "parent_ids": { "type": "array", "items": { "type": "string" } },
    "random_seed": { "type": "integer" }
  }
}
```

---

## 8. Summary

| Component | Effort | Dependencies |
|-----------|--------|---------------|
| Enrich summary.json | Low | None |
| make_run_card | Low | summary.json |
| make_postmortem | Low | summary.json |
| Desktop KB (FastAPI + Postgres + Qdrant) | Medium | Docker, Python deps |
| Proposer KB integration | Medium | KB service |
| Papers ingest | Medium | PDF libs, embedding model |
| LLM brief/propose | Higher | LLM API, prompt engineering |

**Recommended first step:** Phase 1 (enrich summary.json) — small, backward-compatible, and unlocks all later phases.

---

## 9. LLM and PDF Papers Setup

### 9.1 LLM Configuration

The `/brief` and `/propose` endpoints use an LLM. Configure via environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `KB_LLM_PROVIDER` | `openai`, `anthropic`, or `openai_compatible` | `openai` |
| `KB_LLM_MODEL` | Model name | `gpt-4o-mini`, `claude-3-5-haiku` |
| `OPENAI_API_KEY` | OpenAI API key (for `openai` provider) | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key (for `anthropic` provider) | `sk-ant-...` |
| `KB_LLM_API_KEY` | Fallback API key | |
| `KB_LLM_BASE_URL` | For `openai_compatible` (Ollama, vLLM, etc.) | `http://localhost:11434/v1` |

**OpenAI:**
```bash
export KB_LLM_PROVIDER=openai
export KB_LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...
```

**Anthropic:**
```bash
export KB_LLM_PROVIDER=anthropic
export KB_LLM_MODEL=claude-3-5-haiku-20241022
export ANTHROPIC_API_KEY=sk-ant-...
```

**Local (Ollama):**
```bash
export KB_LLM_PROVIDER=openai_compatible
export KB_LLM_MODEL=llama3.2
export KB_LLM_BASE_URL=http://localhost:11434/v1
# No API key needed for local Ollama
```

**Using the LLM proposer:**
```bash
python tools/propose_batch.py --kb-url http://127.0.0.1:8000 --llm --dry-run
```

### 9.2 PDF Papers Base

To add a corpus of PDF papers for `/brief` and semantic search:

**1. Fetch papers automatically (recommended)**

```bash
python knowledge/scripts/fetch_papers.py [--max-per-source 20]
```

This searches **arXiv** (primary, always free) and **Semantic Scholar** (broader, open-access only), using **Unpaywall** to resolve PDFs when publisher links are restricted. Default queries target stellarator coil optimization.

**Options:**
- `--s2` — also fetch from Semantic Scholar (default: arXiv only)
- `--append` — append to existing manifest
- `--queries "stellarator" "coil optimization"` — custom search terms
- `UNPAYWALL_EMAIL` — set for Unpaywall API (recommended)
- `SEMANTIC_SCHOLAR_API_KEY` — optional, for higher S2 rate limits

**2. Or add papers manually**

Edit `knowledge/papers_manifest.jsonl` (one JSON object per line):

```jsonl
{"id":"landreman_2021","path":"knowledge/papers/landreman_2021.pdf","title":"Landreman & Paul 2021","authors":["Landreman","Paul"],"year":2021,"tags":["stellarator","coils","QA"]}
```

**3. Place PDFs**

Put PDF files under `knowledge/papers/` (or paths referenced in the manifest).

**3. Ingest papers into the KB**

```bash
cd /path/to/stellcoilbench
conda activate stellcoilbench_comprehensive
pip install pymupdf  # or pypdf for PDF extraction

# Ensure KB server is running (with Qdrant)
uvicorn knowledge.services.kb_server:app --host 127.0.0.1 --port 8000  # in another terminal

python knowledge/scripts/ingest_papers.py
```

**4. Run the ingest script**

```bash
python knowledge/scripts/ingest_papers.py
```

The script reads `knowledge/papers_manifest.jsonl`, extracts and chunks each PDF, embeds with sentence-transformers, and upserts to Qdrant `paper_chunks`. Requires Qdrant running (e.g. `docker compose -f knowledge/docker-compose.yml up -d`).

The KB server's `/brief` searches `paper_chunks` when `query` is provided. If no papers are ingested, `/brief` still works using runs only.

**5. Optional: paper_chunks collection**

The KB server expects a Qdrant collection `paper_chunks` with payload:

- `paper_id`, `title`, `chunk_text`, `page`

Create it on first ingest (same vector size as `run_cards` — 384 for all-MiniLM-L6-v2).
