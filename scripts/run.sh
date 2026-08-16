#!/usr/bin/env bash
# One button. Launches the console and opens it.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${KESSLER_PORT:-8760}"

[ -d .venv ] || { echo "run ./scripts/bootstrap.sh first"; exit 1; }

# Nemotron if a local NIM is up, otherwise whatever is configured.
if [ -z "${KESSLER_BACKEND:-}" ] && curl -sf -m 2 "http://localhost:8000/v1/models" >/dev/null 2>&1; then
  export KESSLER_BACKEND=nemotron
  export KESSLER_BASE_URL="http://localhost:8000/v1"
  echo "→ local Nemotron NIM detected, using it"
fi

echo "→ http://localhost:${PORT}"
( sleep 4; command -v xdg-open >/dev/null && xdg-open "http://localhost:${PORT}" \
  || command -v open >/dev/null && open "http://localhost:${PORT}" ) >/dev/null 2>&1 &

exec .venv/bin/streamlit run app.py \
  --server.port "${PORT}" --server.address 0.0.0.0 --server.headless true
