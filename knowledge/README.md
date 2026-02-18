# StellCoilBench Desktop Knowledge Base

Local KB for run summaries and (optionally) papers. Uses Postgres or SQLite + Qdrant + FastAPI.

## Quick start (no Docker — SQLite)

### 1. Install Python deps

```bash
pip install -r knowledge/requirements.txt
```

### 2. Start the KB server (SQLite, no Docker)

```bash
KB_USE_SQLITE=1 python -m uvicorn knowledge.services.kb_server:app --host 0.0.0.0 --port 8000
```

Or use the run script (from repo root):

```bash
KB_USE_SQLITE=1 ./knowledge/scripts/run_kb_server.sh
```

Data is stored in `knowledge/kb.sqlite`. Semantic search (Qdrant) is skipped without Docker.

## Full setup (Docker — Postgres + Qdrant)

### 1. Start Postgres and Qdrant

```bash
docker compose -f knowledge/docker-compose.yml up -d
```

### 2. Install Python deps

```bash
pip install -r knowledge/requirements.txt
```

### 3. Start the KB server

```bash
python -m uvicorn knowledge.services.kb_server:app --host 0.0.0.0 --port 8000
```

### 5. Ingest run summaries

From the repo root, after `git pull` (or with `--no-pull` to skip):

```bash
python knowledge/scripts/kb_updater.py
```

### 6. Use with proposer (optional)

When the KB is running, the proposer can use it for context:

```bash
python tools/propose_batch.py --kb-url http://localhost:8000 --dry-run
```

Or set env vars: `KB_URL=http://localhost:8000`

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `KB_POSTGRES_DSN` | `postgresql://kb:kb@localhost:5432/stellcoilbench_kb` | Postgres connection |
| `KB_QDRANT_URL` | `http://localhost:6333` | Qdrant URL |

## Endpoints

- `POST /ingest/run` — upsert run summary (idempotent)
- `GET /runs/top?k=10&success=true` — top runs by score
- `GET /search/runs?q=...&k=10` — semantic search over run cards
- `GET /stats/recent?window=30` — failure stats

## Data flow

1. CI runs cases → writes `cases/done/*/summary.json`
2. CI **ingests** summaries into KB (if KB server is running on the runner)
3. CI **proposer** uses KB for top runs and failure stats (if KB server is running)
4. CI commits results and proposed cases to git

**On the runner:** Start the KB server as a persistent service (systemd, screen, etc.). Then CI will automatically ingest and use it. If the KB is not running, CI still works (ingest is skipped, proposer falls back to local files).
