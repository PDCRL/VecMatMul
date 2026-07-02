#!/usr/bin/env bash
# =============================================================================
# remote_run_sweep.sh — runs on the RUBIK Pi 3 board. Reads a manifest CSV
# (n,k,t,variant,exp,reps) from stdin — reps is per-row, so slow scalar
# kernels at large n can use fewer reps than the fast vectorized ones — runs
# ~/matmul-fastrpc/matmul that many times per row, and prints one RESULT line
# per rep:
#   RESULT,<exp>,<n>,<k>,<t>,<variant>,<rep>,<cycles>,<pass>
# If a row omits the 6th (reps) column, falls back to $REPS (default 3).
# =============================================================================
set -u
cd "$HOME/matmul-fastrpc" || exit 1
DEFAULT_REPS="${REPS:-3}"

while IFS=, read -r n k t variant exp reps; do
  [ -z "$n" ] && continue
  reps="${reps:-$DEFAULT_REPS}"
  for rep in $(seq 1 "$reps"); do
    out=$(sudo ./matmul -n "$n" -k "$k" -t "$t" -d 3 -U 1 -m "$variant" 2>&1)
    cycles=$(echo "$out" | grep -oE 'DSP wall pcycles: [0-9]+' | grep -oE '[0-9]+')
    pass=$(echo "$out" | grep -qE '^Result: PASS' && echo 1 || echo 0)
    [ -z "$cycles" ] && cycles=0
    echo "RESULT,$exp,$n,$k,$t,$variant,$rep,$cycles,$pass"
  done
done
