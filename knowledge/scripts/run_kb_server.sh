#!/bin/bash
# Run the KB server. Uses python -m uvicorn so uvicorn need not be on PATH.
# Usage: ./knowledge/scripts/run_kb_server.sh
# Or: KB_USE_SQLITE=1 ./knowledge/scripts/run_kb_server.sh

set -e
cd "$(dirname "$0")/../.."
python -m uvicorn knowledge.services.kb_server:app --host 127.0.0.1 --port 8000 "$@"
