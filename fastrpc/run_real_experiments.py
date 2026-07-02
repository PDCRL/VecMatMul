#!/usr/bin/env python3
"""
run_real_experiments.py — real-CDSP counterpart of run_all_experiments.ipynb.

Runs the same 3 sweeps (vs size, vs k, vs threads) as the Hexagon-simulator
notebook, but against the FastRPC `matmul` binary on a real RUBIK Pi 3 board
over ssh, for all 7 kernel variants. Reports DSP wall pcycles only (no
L1D/L2 cache-miss counters -- those remain simulator-only).

Usage: ./run_real_experiments.py [--host ubuntu@172.24.132.68] [--reps 3]
Writes real_exp1_vs_size.csv, real_exp2_vs_k.csv, real_exp3_vs_threads.csv
to results_real_device/ at the repo root.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
OUT_DIR = REPO_ROOT / "results_real_device"

VARIANTS = [
    ("seq", "Sequential"),
    ("static", "Static"),
    ("dynamic", "Dynamic"),
    ("hvx", "HVX"),
    ("hvx_mt", "HVX+MT"),
    ("hvx_mt_static", "HVX+MT+Static"),
    ("qhl_sdk", "QHL SDK"),
]

EXP1_SIZES = [32, 64, 96, 128, 196, 256, 512, 1024, 2048]
EXP1_K, EXP1_T = 32, 4

EXP2_N, EXP2_T = 256, 4
EXP2_KS = [128, 64, 32]

EXP3_N, EXP3_K = 256, 32
EXP3_THREADS = list(range(1, 11))


def build_manifest():
    rows = []
    for n in EXP1_SIZES:
        for key, _ in VARIANTS:
            rows.append((n, EXP1_K, EXP1_T, key, "exp1"))
    for k in EXP2_KS:
        for key, _ in VARIANTS:
            rows.append((EXP2_N, k, EXP2_T, key, "exp2"))
    for t in EXP3_THREADS:
        for key, _ in VARIANTS:
            rows.append((EXP3_N, EXP3_K, t, key, "exp3"))
    return rows


def run_sweep(host, reps):
    manifest = build_manifest()
    manifest_text = "\n".join(f"{n},{k},{t},{v},{exp}" for n, k, t, v, exp in manifest) + "\n"

    print(f"Deploying sweep script to {host} ...", file=sys.stderr)
    subprocess.run(["scp", "-q", str(HERE / "remote_run_sweep.sh"), f"{host}:~/remote_run_sweep.sh"], check=True)

    print(f"Running {len(manifest)} combos x {reps} reps = {len(manifest) * reps} runs on {host} ...", file=sys.stderr)
    proc = subprocess.run(
        ["ssh", host, f"REPS={reps} bash ~/remote_run_sweep.sh"],
        input=manifest_text, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"remote sweep failed (exit {proc.returncode})")

    results = {"exp1": [], "exp2": [], "exp3": []}
    label_of = dict(VARIANTS)
    raw = {}  # (exp,n,k,t,variant) -> list of (cycles, pass)
    fails = []

    for line in proc.stdout.splitlines():
        if not line.startswith("RESULT,"):
            if line.strip():
                print("  [remote]", line, file=sys.stderr)
            continue
        _, exp, n, k, t, variant, rep, cycles, ok = line.split(",")
        key = (exp, int(n), int(k), int(t), variant)
        raw.setdefault(key, []).append((int(cycles), ok == "1"))

    for (exp, n, k, t, variant), reps_data in raw.items():
        cycles_list = [c for c, ok in reps_data if ok and c > 0]
        if not cycles_list:
            fails.append((exp, n, k, t, variant))
            continue
        results[exp].append({
            "size": n, "grid_size": k, "threads": t, "method": label_of[variant],
            "avg_pcycles": mean(cycles_list),
            "min_pcycles": min(cycles_list),
            "max_pcycles": max(cycles_list),
        })

    if fails:
        print(f"WARNING: {len(fails)} combos had no passing rep:", file=sys.stderr)
        for f in fails:
            print("  ", f, file=sys.stderr)

    return results


def write_csvs(results):
    specs = [
        ("exp1", "real_exp1_vs_size.csv", ["size", "grid_size", "threads", "method", "avg_pcycles", "min_pcycles", "max_pcycles"]),
        ("exp2", "real_exp2_vs_k.csv", ["size", "grid_size", "threads", "method", "avg_pcycles", "min_pcycles", "max_pcycles"]),
        ("exp3", "real_exp3_vs_threads.csv", ["size", "grid_size", "threads", "method", "avg_pcycles", "min_pcycles", "max_pcycles"]),
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for exp, fname, cols in specs:
        rows = sorted(results[exp], key=lambda r: (r["method"], r["size"], r["grid_size"], r["threads"]))
        out_path = OUT_DIR / fname
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {out_path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="ubuntu@172.24.132.68")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    results = run_sweep(args.host, args.reps)
    write_csvs(results)


if __name__ == "__main__":
    main()
