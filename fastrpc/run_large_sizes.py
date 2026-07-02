#!/usr/bin/env python3
"""
run_large_sizes.py — extends real_exp1_vs_size.csv with n=1024,2048.

Vectorized variants (HVX, HVX+MT, HVX+MT+Static, QHL SDK) use the usual 3
reps. Scalar variants (Sequential, Static, Dynamic) use 1 rep at these sizes
-- Sequential alone took 6.5 minutes for a single n=1024 run (severe cache
thrashing in the naive scalar triple-loop, not just O(n^3) growth), so 3 reps
of all 3 scalar kernels at both sizes would take hours.

Appends new rows to results_real_device/real_exp1_vs_size.csv (does not
touch existing rows) and regenerates the plots.
"""
import csv
import subprocess
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
OUT_DIR = REPO_ROOT / "results_real_device"
CSV_PATH = OUT_DIR / "real_exp1_vs_size.csv"

VARIANTS = [
    ("seq", "Sequential"),
    ("static", "Static"),
    ("dynamic", "Dynamic"),
    ("hvx", "HVX"),
    ("hvx_mt", "HVX+MT"),
    ("hvx_mt_static", "HVX+MT+Static"),
    ("qhl_sdk", "QHL SDK"),
]
SCALAR = {"seq", "static", "dynamic"}

LARGE_SIZES = [1024, 2048]
K, T = 32, 4
HOST = "ubuntu@172.24.132.68"


def build_manifest():
    rows = []
    for n in LARGE_SIZES:
        for key, _ in VARIANTS:
            reps = 1 if key in SCALAR else 3
            rows.append((n, K, T, key, "exp1", reps))
    return rows


def main():
    manifest = build_manifest()
    manifest_text = "\n".join(f"{n},{k},{t},{v},{exp},{reps}" for n, k, t, v, exp, reps in manifest) + "\n"

    print(f"Deploying sweep script to {HOST} ...", file=sys.stderr)
    subprocess.run(["scp", "-q", str(HERE / "remote_run_sweep.sh"), f"{HOST}:~/remote_run_sweep.sh"], check=True)

    total_runs = sum(r[5] for r in manifest)
    print(f"Running {len(manifest)} combos, {total_runs} total runs on {HOST} "
          f"(sizes={LARGE_SIZES}, scalar reps=1, vectorized reps=3) ...", file=sys.stderr)
    proc = subprocess.run(
        ["ssh", HOST, "bash ~/remote_run_sweep.sh"],
        input=manifest_text, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"remote sweep failed (exit {proc.returncode})")

    label_of = dict(VARIANTS)
    raw = {}
    for line in proc.stdout.splitlines():
        if not line.startswith("RESULT,"):
            if line.strip():
                print("  [remote]", line, file=sys.stderr)
            continue
        _, exp, n, k, t, variant, rep, cycles, ok = line.split(",")
        key = (int(n), int(k), int(t), variant)
        raw.setdefault(key, []).append((int(cycles), ok == "1"))

    new_rows = []
    fails = []
    for (n, k, t, variant), reps_data in raw.items():
        cycles_list = [c for c, ok in reps_data if ok and c > 0]
        if not cycles_list:
            fails.append((n, k, t, variant))
            continue
        new_rows.append({
            "size": n, "grid_size": k, "threads": t, "method": label_of[variant],
            "avg_pcycles": mean(cycles_list),
            "min_pcycles": min(cycles_list),
            "max_pcycles": max(cycles_list),
        })

    if fails:
        print(f"WARNING: {len(fails)} combos had no passing rep:", file=sys.stderr)
        for f in fails:
            print("  ", f, file=sys.stderr)

    cols = ["size", "grid_size", "threads", "method", "avg_pcycles", "min_pcycles", "max_pcycles"]
    existing_rows = []
    if CSV_PATH.exists():
        with CSV_PATH.open() as f:
            existing_rows = list(csv.DictReader(f))

    all_rows = existing_rows + [{k: str(v) for k, v in r.items()} for r in new_rows]
    all_rows = sorted(all_rows, key=lambda r: (r["method"], int(r["size"]), int(r["grid_size"]), int(r["threads"])))

    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"Wrote {CSV_PATH} ({len(all_rows)} rows total, {len(new_rows)} new)")


if __name__ == "__main__":
    main()
