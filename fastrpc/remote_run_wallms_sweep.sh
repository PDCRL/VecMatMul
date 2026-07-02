#!/usr/bin/env bash
# =============================================================================
# remote_run_wallms_sweep.sh — runs on the RUBIK Pi 3 board. Reads a manifest
# CSV (n,k,t,variant,reps) from stdin, runs ~/matmul-fastrpc/matmul that many
# times per row, and prints one RESULT line per rep with wall-clock ms
# (for the DSP-vs-ARM-CPU comparison; separate from remote_run_sweep.sh which
# is pcycles-only and feeds the existing real_exp*.csv pipeline):
#   RESULT,<n>,<k>,<t>,<variant>,<rep>,<ms>,<pass>
# =============================================================================
set -u
cd "$HOME/matmul-fastrpc" || exit 1
DEFAULT_REPS="${REPS:-3}"

while IFS=, read -r n k t variant reps; do
  [ -z "$n" ] && continue
  reps="${reps:-$DEFAULT_REPS}"
  for rep in $(seq 1 "$reps"); do
    out=$(sudo ./matmul -n "$n" -k "$k" -t "$t" -d 3 -U 1 -m "$variant" 2>&1)
    ms=$(echo "$out" | grep -oE 'DSP call wall ms: *[0-9.]+' | grep -oE '[0-9.]+$')
    pass=$(echo "$out" | grep -qE '^Result: PASS' && echo 1 || echo 0)
    [ -z "$ms" ] && ms=0
    echo "RESULT,$n,$k,$t,$variant,$rep,$ms,$pass"
  done
done
