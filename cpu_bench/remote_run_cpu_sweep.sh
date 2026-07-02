#!/usr/bin/env bash
# =============================================================================
# remote_run_cpu_sweep.sh — runs on the RUBIK Pi 3 board. Reads a manifest CSV
# (n,t,variant,reps) from stdin, runs ~/cpu_bench/matmul_cpu_bench that many
# times per row, and prints one RESULT line per rep:
#   RESULT,<n>,<t>,<variant>,<rep>,<ms>,<pass>
# =============================================================================
set -u
cd "$HOME/cpu_bench" || exit 1
DEFAULT_REPS="${REPS:-3}"

while IFS=, read -r n t variant reps; do
  [ -z "$n" ] && continue
  reps="${reps:-$DEFAULT_REPS}"
  for rep in $(seq 1 "$reps"); do
    out=$(./matmul_cpu_bench -n "$n" -t "$t" -m "$variant" 2>&1)
    ms=$(echo "$out" | grep -oE 'CPU wall ms: *[0-9.]+' | grep -oE '[0-9.]+$')
    pass=$(echo "$out" | grep -qE '^Result: PASS' && echo 1 || echo 0)
    [ -z "$ms" ] && ms=0
    echo "RESULT,$n,$t,$variant,$rep,$ms,$pass"
  done
done
