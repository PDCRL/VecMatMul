# VecMatMul — Vectorized Matrix Multiplication on the Hexagon DSP

This artifact contains a set of single-precision (`float32`) **N×N matrix
multiplication** kernels for the **Qualcomm Hexagon DSP**, running under the
**QuRT** real-time OS and executed on the cycle-approximate **Hexagon
simulator**. It accompanies our paper and is meant to be built and reproduced
on any machine with the Hexagon SDK installed.

The kernels span scalar baselines, hand-written **HVX** (Hexagon Vector
eXtensions) vectorized kernels, single- vs. multi-threaded variants with both
static and dynamic work distribution, and the vendor **QHL HVX BLAS** routine
as a reference point.

Everything is self-contained: matrices are generated deterministically
in-memory, so no data files are needed. Each run reports the matmul **wall
cycles**, a **checksum** (for correctness cross-checking), and the simulator's
**PMU statistics** (instructions, packets, pcycles).

---

## 1. Kernels

| Source file | `make` target | Vectorized | Threading | Description |
|---|---|:---:|---|---|
| `matmul_sequential.c`        | `matmul_sequential`         | —   | single   | Scalar triple-loop baseline (no tiling). |
| `matmul_experiment.c`        | `matmul_experiment`         | —   | static   | Scalar, rows split statically across threads. |
| `matmul_dynamic.c`           | `matmul_dynamic`            | —   | dynamic  | Scalar, atomic-counter dynamic task allocation. |
| `matmul_vectorized.c`        | `matmul_vectorized`         | HVX | single   | HVX kernel, one thread (32 floats/vector). |
| `matmul_vectorized_mt.c`     | `matmul_vectorized_mt`      | HVX | dynamic  | HVX + atomic-counter dynamic grid allocation. |
| `matmul_vectorized_mt_static.c` | `matmul_vectorized_mt_static` | HVX | static | HVX + statically assigned grids per thread. |
| `matmul_qhl_sdk.c`           | `matmul_qhl_sdk`            | HVX | single   | Vendor reference: `qhblas_hvx_matrix_matrix_mpy_af` from the SDK's QHL HVX BLAS. |

All kernels compute `C = A × B` for square `N×N` matrices in `float32`. The HVX
kernels use 128-byte vectors (`-mhvx-length=128B`) with QFloat
(`-mhvx-qfloat`), processing 32 floats per vector lane.

---

## 2. Requirements

- **Qualcomm Hexagon SDK 6.5.0.0** (other 6.x versions are likely to work).
  Download from the [Qualcomm developer site](https://developer.qualcomm.com/software/hexagon-dsp-sdk).
- The SDK's **HEXAGON_Tools** toolchain (this artifact was validated with
  `hexagon-clang` **19.0.07**), providing `hexagon-clang` and `hexagon-sim`.
- A **Linux x86-64** host (the SDK simulator is Linux-only).
- `make`.

The `matmul_qhl_sdk` target additionally needs the prebuilt **QHL** and
**QHL-HVX** libraries that ship with the SDK
(`libs/qhl{,_hvx}/prebuilt/hexagon_toolv<major>_<hexver>/`). The other six
kernels need only the toolchain and the QuRT runtime — no extra libraries.

---

## 3. Setup

Point the build at your SDK and put the toolchain on `PATH`. The easiest way is
to source the SDK environment script, which sets `HEXAGON_SDK_ROOT` and adds
`hexagon-clang`/`hexagon-sim` to `PATH`:

```bash
source /path/to/Qualcomm/Hexagon_SDK/6.5.0.0/setup_sdk_env.source
```

Alternatively, set the variables yourself:

```bash
export HEXAGON_SDK_ROOT=/path/to/Qualcomm/Hexagon_SDK/6.5.0.0
export PATH="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.07/Tools/bin:$PATH"
```

You can also pass `HEXAGON_SDK_ROOT=...` directly on the `make` command line.
The Makefile auto-detects the `HEXAGON_Tools` version under the SDK; override it
with `HEXAGON_TOOLS_ROOT=...` if you have several installed.

Verify the toolchain is reachable:

```bash
hexagon-clang --version    # should print "Hexagon Clang version 19.x"
hexagon-sim --version
```

---

## 4. Build

```bash
make all                       # build every kernel into build/hexagon/
make matmul_vectorized_mt      # build a single kernel
make clean                     # remove build/
make help                      # list all targets and current settings
```

Binaries are written to `build/hexagon/`. The harmless linker warning
`Undefined weak symbol __dl_cxa_refcount ...` comes from the SDK's libc and can
be ignored.

To target a different Hexagon architecture (must match an installed QuRT image,
and — for `matmul_qhl_sdk` — an available QHL prebuilt):

```bash
make all HEX_VERSION=v68
```

---

## 5. Run on the simulator

Each `sim-<kernel>` target builds the kernel, writes the run parameters to
`build/hexagon/input.txt`, runs the binary on `hexagon-sim` under QuRT, and
prints the result:

```bash
make sim-sequential
make sim-vectorized
make sim-vectorized-mt
make sim-vectorized-mt-static
make sim-dynamic
make sim-experiment
make sim-qhl-sdk

make sim-all                   # run all seven in sequence
```

### Run parameters

Three integers control each run; override any of them on the command line:

| Var | Meaning | Used by |
|----|---------|---------|
| `N` | Matrix dimension (computes `N×N`) | all kernels |
| `K` | Grid / block size for tiling | tiled & multithreaded kernels |
| `T` | Number of QuRT threads | multithreaded kernels |

```bash
make sim-vectorized-mt N=256 K=32 T=4
make sim-vectorized    N=128            # single-threaded: K, T ignored
```

> **Note on `N`:** the HVX kernels process 32 floats per vector. Use an `N`
> that is a multiple of 32 (e.g. 32, 64, 128, 256) for full-vector throughput.
> For the multithreaded kernels, `N` should be divisible by `K`, and `K` chosen
> so the grid count is divisible among `T` threads.

---

## 6. Input / output

These are handled automatically by the `sim-*` targets; this section documents
the mechanism for anyone running the binaries by hand.

- **Input** — each kernel reads `input.txt` (from the simulator's mounted
  filesystem, `build/hexagon/`) containing one line: `N K T`. The `make input`
  target regenerates it from the current `N`/`K`/`T` values.
- **Matrices** — `A` and `B` are generated deterministically in-memory
  (`A[i] = (i % 100) * 0.1`, `B[i] = ((i*7+13) % 100) * 0.1`), so every run is
  reproducible and no data files are needed.
- **Output** — the result dimension is written to `build/hexagon/output.txt`,
  and the program prints to the console:
  - `Total wall cycles` — cycles spent in the matmul region (the headline
    number), measured with the QuRT cycle counter.
  - `Checksum` — sum of all elements of `C`, used to confirm correctness across
    kernels (e.g. `matmul_vectorized` and `matmul_qhl_sdk` print the **same**
    checksum for a given `N`).
- **PMU statistics** — the simulator writes `build/hexagon/pmu_stats.txt` and
  prints a per-thread `Insns / Packets / Pcycles` summary.

---

## 7. Reproducing a sweep

To sweep a parameter (e.g. matrix size) for one kernel, loop over `make`:

```bash
for n in 32 64 128 256 512; do
  echo "=== N=$n ==="
  make sim-vectorized N=$n
done
```

Thread scaling for a fixed problem size:

```bash
for t in 1 2 3 4; do
  make sim-vectorized-mt N=256 K=32 T=$t
done
```

Parse the `Total wall cycles:` line from each run for timing, and
`pmu_stats.txt` for microarchitectural counters.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `'hexagon-clang' not found` | Source `setup_sdk_env.source` or add the toolchain `bin` to `PATH`; or pass `CC_HEXAGON=/full/path/to/hexagon-clang`. |
| `hexagon-sim not found` | Same as above for `HEXAGON_SIM`. |
| `QHL prebuilt not found at ...` | Your `HEX_VERSION` has no matching QHL prebuilt. Choose one that exists under `libs/qhl_hvx/prebuilt/` (e.g. `v68`, `v69`, `v73`), or skip `matmul_qhl_sdk`. |
| Wrong SDK picked up | Set `HEXAGON_SDK_ROOT` explicitly; check `make help` echoes the right paths. |
| `Undefined weak symbol __dl_cxa_refcount` warning | Benign SDK libc warning — ignore. |
| `Failed to unlock ETM base TLB entry` at sim start | Benign simulator message — ignore. |

`make help` prints the resolved `HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS_ROOT`, and
`HEX_VERSION`, which is the quickest way to confirm your environment.

---

## 9. Repository layout

```
VecMatMul/
├── Makefile                        # standalone build/run for all kernels
├── README.md                       # this file
├── matmul_sequential.c             # scalar single-threaded baseline
├── matmul_experiment.c             # scalar static multithreading
├── matmul_dynamic.c                # scalar dynamic (atomic) multithreading
├── matmul_vectorized.c             # HVX single-threaded
├── matmul_vectorized_mt.c          # HVX dynamic multithreading
├── matmul_vectorized_mt_static.c   # HVX static multithreading
├── matmul_qhl_sdk.c                # vendor QHL HVX BLAS reference
└── build/hexagon/                  # build output (created by make; git-ignored)
```
