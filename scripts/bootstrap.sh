#!/usr/bin/env bash
# Project Kessler -- one-shot setup for the Acer Veriton GN100 (GB10 Grace Blackwell).
#
#   ./scripts/bootstrap.sh
#
# Detects the platform, installs the right numerical stack, warms the orbital
# catalogue, and runs preflight. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[38;5;79m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[38;5;214m!\033[0m %s\n" "$*"; }
die()  { printf "  \033[38;5;203m✗\033[0m %s\n" "$*"; exit 1; }

bold "Project Kessler — bootstrap"
echo

# ---------------------------------------------------------------- platform
ARCH="$(uname -m)"; OS="$(uname -s)"
ok "platform: $OS/$ARCH"

CUDA_MAJOR=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  CUDA_VER="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -1 || true)"
  CUDA_MAJOR="${CUDA_VER%%.*}"
  ok "GPU: ${GPU_NAME:-unknown}  (CUDA ${CUDA_VER:-unknown})"
else
  warn "no nvidia-smi — will install the CPU stack; everything still runs, slower"
fi

# ---------------------------------------------------------------- python
command -v python3 >/dev/null 2>&1 || die "python3 not found"
PY_VER="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
ok "python: $PY_VER"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ok "created .venv"
else
  ok "reusing .venv"
fi
PIP=".venv/bin/pip"
$PIP install -q --upgrade pip setuptools wheel

# ---------------------------------------------------------------- deps
bold "Installing dependencies"
$PIP install -q -r requirements.txt
ok "base stack (skyfield, numpy, plotly, streamlit)"

$PIP install -q openai mcp 2>/dev/null && ok "agent stack (openai-compatible client, MCP)" \
  || warn "agent extras failed — deterministic solver will still run"

# CuPy: the wheel has to match the CUDA major version on the box.
if [ -n "$CUDA_MAJOR" ]; then
  CUPY_PKG="cupy-cuda${CUDA_MAJOR}x"
  if $PIP install -q "$CUPY_PKG" 2>/dev/null; then
    ok "installed $CUPY_PKG"
  elif $PIP install -q cupy-cuda12x 2>/dev/null; then
    ok "installed cupy-cuda12x (fallback)"
  else
    warn "no CuPy wheel matched — screening will run on NumPy"
    warn "on DGX/GB10 images CuPy often ships preinstalled system-wide; try:"
    warn "  .venv/bin/pip install --no-build-isolation cupy"
  fi
fi

# ---------------------------------------------------------------- data
bold "Warming the orbital catalogue"
if .venv/bin/python -c "
from kessler.catalog import load_full_catalog
c = load_full_catalog()
print(f'  fetched {len(c):,} objects')
" 2>/dev/null; then
  ok "catalogue cached to data/ — the demo now runs offline"
else
  warn "catalogue fetch failed (no network?). The demo needs data/*.tle present."
fi

echo
bold "Preflight"
.venv/bin/python scripts/preflight.py || true

echo
bold "Ready"
echo "  Launch:  ./scripts/run.sh"
echo "  Console: .venv/bin/python run_demo.py"
