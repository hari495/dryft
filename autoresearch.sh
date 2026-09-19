#!/usr/bin/env bash
# Canonical benchmark: one official Dryft run of the current engine/ tree.
#
# There is no local H100 and this round only accepts official runs, so the
# platform is the bench. Cheap local gates run first so a broken engine never
# burns a 10-15 minute GPU slot:
#   1. CPU correctness vs the HF reference on a tiny Qwen3 (agent/verify.py --tiny --all)
#   2. the platform's own archive lint (bin/dryft validate engine)
#   3. snapshot the working tree onto origin/main, push, wait, print METRIC lines
#
# Env: DRYFT_TOKEN / DRYFT_API from ./.env (gitignored). AUTORESEARCH_DESC names
# the pushed commit. Primary metric: score (hidden-workload geomean, native=100).
set -euo pipefail
cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a; . ./.env; set +a
fi
: "${DRYFT_TOKEN:?set DRYFT_TOKEN in .env}"
export DRYFT_API="${DRYFT_API:-https://htn.dryft.ai}"
PY=.venv/bin/python

echo "[autoresearch] gate 1/3: CPU correctness (tiny Qwen3 vs HF)" >&2
$PY agent/verify.py --tiny --all 2>&1 | grep -v "Sliding Window" | tail -4 >&2
test "${PIPESTATUS[0]}" -eq 0

echo "[autoresearch] gate 2/3: platform lint" >&2
./bin/dryft validate engine 2>&1 | tail -1 >&2
test "${PIPESTATUS[0]}" -eq 0

echo "[autoresearch] gate 3/3: official run" >&2
exec $PY agent/official_run.py --branch main --timeout "${AUTORESEARCH_TIMEOUT:-2700}"
