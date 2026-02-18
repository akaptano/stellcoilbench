-- StellCoilBench desktop KB: runs table
-- Run after Postgres is up: psql -h localhost -U kb -d stellcoilbench_kb -f knowledge/db/schema.sql

CREATE TABLE IF NOT EXISTS runs (
    case_id TEXT PRIMARY KEY,
    summary JSONB NOT NULL,
    run_card TEXT,
    postmortem TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_runs_success ON runs ((summary->>'success'));
CREATE INDEX IF NOT EXISTS idx_runs_total_score ON runs (((summary->>'total_score')::float));
CREATE INDEX IF NOT EXISTS idx_runs_failure_class ON runs ((summary->>'failure_class'));
