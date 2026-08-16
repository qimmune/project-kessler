#!/usr/bin/env bash
# Run every suite and report honestly.
cd "$(dirname "$0")/.."
pass=0; fail=0
for t in tests/test_*.py; do
  printf "  %-32s " "$(basename "$t")"
  if out=$(.venv/bin/python "$t" 2>&1); then
    echo "PASS"; pass=$((pass+1))
  else
    echo "FAIL"; echo "$out" | tail -12 | sed 's/^/      /'; fail=$((fail+1))
  fi
done
echo
echo "  $pass passed, $fail failed"
[ "$fail" -eq 0 ]
