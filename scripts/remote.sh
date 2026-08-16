#!/usr/bin/env bash
# Drive Project Kessler on the GN100 from your laptop.
#
#   ./scripts/remote.sh setup   user@gn100     # clone + bootstrap + preflight
#   ./scripts/remote.sh bench   user@gn100     # CPU vs GPU numbers for the pitch
#   ./scripts/remote.sh run     user@gn100     # launch the UI, tunnel it to localhost:8760
#   ./scripts/remote.sh shell   user@gn100
#
# `run` opens an SSH tunnel, so you browse http://localhost:8760 on your laptop
# while every calculation happens on the GB10. Nothing is exposed to the venue
# network.
set -euo pipefail

CMD="${1:-help}"
HOST="${2:-}"
REPO="${KESSLER_REPO:-https://github.com/qimmune/project-kessler.git}"
DIR="${KESSLER_DIR:-~/project-kessler}"
PORT="${KESSLER_PORT:-8760}"

need_host() { [ -n "$HOST" ] || { echo "usage: $0 $CMD user@host"; exit 1; }; }
bold() { printf "\033[1m%s\033[0m\n" "$*"; }

case "$CMD" in
  setup)
    need_host
    bold "Setting up on $HOST"
    ssh "$HOST" "set -e
      if [ -d $DIR/.git ]; then
        cd $DIR && git pull --ff-only
      else
        git clone $REPO $DIR && cd $DIR
      fi
      ./scripts/bootstrap.sh"
    ;;

  bench)
    need_host
    bold "Benchmarking on $HOST"
    ssh "$HOST" "cd $DIR && .venv/bin/python scripts/benchmark.py"
    ;;

  run)
    need_host
    bold "Starting on $HOST, tunnelling to http://localhost:$PORT"
    echo "  Ctrl-C stops the tunnel and the server."
    # -t forces a TTY so Ctrl-C reaches streamlit and does not orphan it
    ssh -t -L "${PORT}:localhost:${PORT}" "$HOST" \
      "cd $DIR && KESSLER_PORT=$PORT ./scripts/run.sh"
    ;;

  shell)  need_host; ssh -t "$HOST" "cd $DIR && exec \$SHELL -l" ;;

  *)
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
