# FastRPC concepts, script internals, and extending this port

For copy-paste run commands, see [`README.md`](README.md). This file is the
longer explanation: terminology for newcomers, what each script does
internally, how to add a new kernel variant, and the simulator-vs-real-silicon
framing.

---

## 1. Concepts, for anyone new to FastRPC

| Term | Meaning |
|---|---|
| **DSP** | A separate physical chip on the SoC (the Hexagon Compute DSP, "CDSP"). Runs its own tiny RTOS (**QuRT**), not Linux. The ARM CPU cannot directly call functions inside it — hence FastRPC. |
| **FastRPC** | Qualcomm's RPC (remote procedure call) framework: lets ARM-side code call a function that actually executes on the DSP, moving arguments and results across the CPU↔DSP boundary. |
| **IDL** (`matmul.idl`) | A hand-written spec file declaring the RPC interface: function names, parameters, and each parameter's direction (`in`/`out`/`rout`). Contains no logic, just signatures. |
| **`in` / `out` / `rout`** | `in` = caller sends this to the DSP. `out` = DSP sends this back. `rout` = "return out" — an output buffer/array allocated by the caller but filled by the DSP. |
| **`qaic`** | Qualcomm's IDL compiler. Reads `matmul.idl` and code-generates the marshalling C source for both sides. You never hand-edit its output. |
| **Marshalling** | Packing function arguments into a byte format that can cross the CPU↔DSP boundary, and unpacking on the other side. |
| **Stub** (`gen/matmul_stub.c`, generated) | CPU-side generated code. Turns your normal-looking function call (e.g. `matmul_mt(...)`) into an actual FastRPC message sent to the DSP. |
| **Skeleton / skel** (`gen/matmul_skel.c`, generated) | DSP-side generated code. Receives the FastRPC message, unpacks the arguments, and calls your real implementation function in `src/matmul_imp.c`. Compiled together with `matmul_imp.c` into one `.so` — "the skel" colloquially refers to this whole compiled package (`libmatmul_skel.so`). |
| **`src/matmul_imp.c`** | Your real DSP-side implementation — the actual matmul compute. You write/edit this. |
| **`src/matmul_main.c`** | Your real CPU-side program — normal ARM `main()`, parses flags, allocates buffers, calls the generated stub functions. You write/edit this. |
| **Handle** (`remote_handle64`) | An opaque ID returned by `matmul_open()` representing one active RPC session. Passed to every subsequent call, analogous to a file descriptor. |
| **Domain** | Which physical DSP to talk to, when a SoC has more than one (ADSP = audio, CDSP = compute, SDSP = sensors). We always use `-d 3` = CDSP. |
| **PD** (Protection Domain) | An isolated execution context on the DSP, analogous to a process — memory-isolates your skel from other DSP clients and the DSP's core system code. |
| **Signed / unsigned PD** | Code-signing requirement for DSP-loaded modules. "Unsigned PD" (`-U 1`) relaxes this so we can load our own skel without an official Qualcomm/OEM signature — normally only available on dev/engineering images. |
| **`rpcmem`** | Special shared-memory allocator (`rpcmem_alloc`/`rpcmem_free`) for buffers the DSP can access directly (via ION/DMA-BUF), avoiding an extra copy for large arrays like our matrices. |
| **Shell** (`fastrpc_shell_*`) | A small pre-loaded binary on the DSP that spawns a new PD and gives it a minimal runtime (libc, thread support) before our skel is loaded into it. Lives in `/usr/lib/dsp/cdsp/` on the device. |

---

## 2. Layout

```
fastrpc/
├── matmul.idl               # IDL: one method per kernel variant
├── src/matmul_imp.c         # DSP side: all 7 compute implementations + RPC entry points
├── src/matmul_main.c        # CPU side: rpcmem buffers, unsigned PD, -m dispatch, verify
├── gen/                      # qaic-generated matmul.h / matmul_stub.c / matmul_skel.c
├── build_skel.sh             # host: regenerate stubs + build the skel (all methods, incl. QHL link)
├── remote_run_sweep.sh       # device: batch-run a manifest of (n,k,t,variant) combos, pcycles
├── remote_run_wallms_sweep.sh# device: same, but records wall-clock ms instead of pcycles
├── run_real_experiments.py   # host: drives remote_run_sweep.sh for the 3 standard sweeps
├── run_large_sizes.py        # host: extends the size sweep to n=1024,2048
└── build/hexagon/            # skel build output (libmatmul_skel.so)
```

---

## 3. What each `.sh` file actually does, line by line

### `build_skel.sh` — host, builds `libmatmul_skel.so`

```bash
SDK="${HEXAGON_SDK_ROOT:-/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.5.0.0}"
ARCH="${HEX_ARCH:-v68}"
```
Reads the SDK path and target Hexagon arch from env vars, with defaults. Every
path below is built from these two.

```bash
TOOLS="$(ls -d "$SDK"/tools/HEXAGON_Tools/* | sort | tail -1)/Tools"
```
Hexagon SDKs bundle multiple toolchain versions (e.g. `19.0.07`) under
`tools/HEXAGON_Tools/`; this picks the **highest-numbered one** by sorting
directory names and taking the last (`tail -1`).

```bash
CLANG="$TOOLS/bin/hexagon-clang"
QAIC="$SDK/ipc/fastrpc/qaic/Ubuntu/qaic"
```
Locates the two binaries we need: the Hexagon cross-compiler, and the IDL
compiler.

```bash
TOOL_MAJOR="$(basename "$(dirname "$TOOLS")" | cut -d. -f1)"
QHL_HVX_DIR="$SDK/libs/qhl_hvx/prebuilt/hexagon_toolv${TOOL_MAJOR}_${ARCH}"
```
The vendor QHL BLAS library (used by the `qhl_sdk` method) ships prebuilt
per-toolchain-version-per-arch, e.g. `hexagon_toolv19_v68/`. This derives that
directory name from the toolchain version found above (`19.0.07` → `19`) and
`ARCH` (`v68`), so switching `HEX_ARCH` automatically picks the matching QHL
build.

```bash
"$QAIC" -mdll -o "$HERE/gen" \
  -I"$SDK/incs" -I"$SDK/incs/stddef" -I"$SDK/ipc/fastrpc/incs" \
  "$HERE/matmul.idl"
```
**Step 1 of the actual build**: runs `qaic` on `matmul.idl`, writing
`gen/matmul.h`, `gen/matmul_stub.c`, `gen/matmul_skel.c` — this is where every
`-m <variant>` method you declared in the IDL gets turned into real C
dispatch code. `-mdll` tells `qaic` to generate for a dynamically-loaded
skel (as opposed to a statically-linked one).

```bash
"$CLANG" $CFLAGS $INCS -c "$HERE/gen/matmul_skel.c" -o "$BD/matmul_skel.o"
"$CLANG" $CFLAGS $INCS -c "$HERE/src/matmul_imp.c"  -o "$BD/matmul_imp.o"
```
**Step 2**: compiles the generated skel dispatch code and your real
implementation (`matmul_imp.c`) into separate `.o` object files, each with
`-mhvx -mhvx-length=128B -mhvx-qfloat` so HVX intrinsics are available to
every method (even the scalar ones — the flag just enables the instruction
set, it doesn't force its use).

**This is where the actual cross-compilation happens**, via two things
together — neither alone is enough:
1. **`$CLANG` itself** (`$TOOLS/bin/hexagon-clang`) is not your host's normal
   `clang`/`gcc`. It's a *target-specific* compiler binary shipped inside the
   Hexagon SDK: it runs as an x86_64 process on your host, but it only knows
   how to emit **Hexagon DSP machine code**, never x86. (Same pattern as e.g.
   `arm-linux-gnueabi-gcc` — the binary's name/location encodes the target,
   unlike plain `gcc` which always targets the machine it runs on.)
2. **`-m$ARCH`** in `CFLAGS` (expands to `-mv68`, `-mv73`, etc.) tells that
   compiler *which Hexagon core generation* to target — this is the
   Hexagon-specific equivalent of `-march=` on x86/ARM gcc. Without it,
   `hexagon-clang` would still cross-compile (it can't do anything else) but
   wouldn't know which instruction-set variant/QuRT ABI to emit for.

So "cross-compiling" here isn't one flag you pass to a normal compiler — it's
*which compiler binary you invoke* (`hexagon-clang`, always cross) combined
with `-m$ARCH` picking the specific DSP generation. `-G0` (small-data
threshold, a Hexagon/QuRT ABI option) and `-fpic` (position-independent code,
required since the output is a `.so`) are additional target/output-format
flags on top, not cross-compile flags themselves.

```bash
"$CLANG" -m$ARCH -G0 -fpic -Wl,-Bsymbolic \
  -shared -o "$BD/libmatmul_skel.so" -Wl,-soname,libmatmul_skel.so \
  "$BD/matmul_skel.o" "$BD/matmul_imp.o" \
  "$QHL_HVX_DIR/libqhblas_hvx.a" "$QHL_HVX_DIR/libqhdsp_hvx.a" "$QHL_HVX_DIR/libqhmath_hvx.a" \
  -L"$LP" -Wl,--no-as-needed -lc
```
**Step 3**: links both `.o` files plus the three QHL static archives (only
the `qhl_sdk` method's symbols actually get pulled in from these) into the
final shared object. `-Wl,--no-as-needed -lc` forces an explicit `NEEDED
libc.so` tag — without it, the QHL archives satisfy enough libc-like symbols
(e.g. `memcpy`) statically that the linker's default implicit libc pull
disappears, and the skel then fails to resolve `malloc`/`free`/`memset` at
PD-load time on the device.

```bash
readelf -dW "$BD/libmatmul_skel.so" | grep -E "NEEDED|SONAME"
```
Final sanity check, printed for you to eyeball: confirms `libc.so` and
`libgcc.so` are both present as `NEEDED`.

### `remote_run_sweep.sh` / `remote_run_wallms_sweep.sh` — device, batch runner

Both scripts have the identical structure; only what they extract from the
output differs (DSP pcycles vs. DSP-call wall-clock ms). Walking through
`remote_run_sweep.sh`:

```bash
cd "$HOME/matmul-fastrpc" || exit 1
DEFAULT_REPS="${REPS:-3}"
```
Moves into the directory containing the already-built `matmul` binary, and
sets a fallback repeat count (overridable via the `REPS` env var when
invoking the script).

```bash
while IFS=, read -r n k t variant exp reps; do
```
Reads **one CSV row at a time from stdin** — the "manifest" — splitting each
line on commas into the 6 named variables. This is why every Python driver
(`run_real_experiments.py` etc.) builds a manifest string and passes it via
`subprocess.run(..., input=manifest_text)` piped over `ssh`, rather than
looping and re-`ssh`ing per combination — one persistent SSH connection
runs the whole sweep, which is much faster than reconnecting per run.

```bash
  reps="${reps:-$DEFAULT_REPS}"
  for rep in $(seq 1 "$reps"); do
```
Per-row reps override the default — this is what lets a size-1024 scalar
kernel run once while a fast HVX kernel at the same size runs 3 times, all
in the same manifest (see `run_large_sizes.py`).

```bash
    out=$(sudo ./matmul -n "$n" -k "$k" -t "$t" -d 3 -U 1 -m "$variant" 2>&1)
    cycles=$(echo "$out" | grep -oE 'DSP wall pcycles: [0-9]+' | grep -oE '[0-9]+')
    pass=$(echo "$out" | grep -qE '^Result: PASS' && echo 1 || echo 0)
    [ -z "$cycles" ] && cycles=0
    echo "RESULT,$exp,$n,$k,$t,$variant,$rep,$cycles,$pass"
```
Runs the actual DSP call, then **greps the printed output** for the
`DSP wall pcycles: <n>` line and the `Result: PASS` line — there's no
structured (e.g. JSON) output format, just line-oriented text scraping. Emits
one `RESULT,...` CSV line to stdout per repetition; every caller then filters
stdout for lines starting with `RESULT,` (ignoring the human-readable
progress lines the `matmul` binary also prints) and parses the rest with
Python's `csv`/`str.split(",")`.

`remote_run_wallms_sweep.sh` is byte-for-byte the same shape, just greps
`DSP call wall ms: <float>` instead of `DSP wall pcycles: <int>`, and its
manifest has no `exp` column (it's used only for the DSP-vs-CPU comparison,
not the 3-experiment sweep pipeline, so there's no exp1/exp2/exp3 label to
carry through).

`VecMatMul/cpu_bench/remote_run_cpu_sweep.sh` (the ARM-CPU-only counterpart)
follows the exact same pattern, reading `(n,t,variant,reps)` and grepping
`CPU wall ms: <float>` from `matmul_cpu_bench`'s output instead.

---

## 4. Adding a brand-new kernel variant

1. **Write the compute** in `src/matmul_imp.c` (or split into a new `.c` file
   if it's large, and add it to `build_skel.sh`'s compile step). Follow the
   pattern of the existing methods: a private `static` compute function, plus
   a public RPC entry point named `matmul_<method>` that validates arguments
   (`AEE_EBADPARM`/`AEE_EBADSIZE`) and calls the compute function.
2. **Declare the method** in `matmul.idl` — add a new `long <method>(...)`
   line inside the `interface matmul : remote_handle64 { ... }` block,
   matching the exact parameter list/types your entry point needs.
3. **Rebuild via `./build_skel.sh`** — this reruns `qaic`, so
   `gen/matmul.h`/`matmul_stub.c`/`matmul_skel.c` regenerate automatically
   with your new method's declaration. You never hand-edit `gen/`.
4. **Add a case to `matmul_main.c`'s `-m` dispatch** (and to
   `variant_label()` if you want it labeled in sweep CSVs) calling the new
   generated stub function.
5. **Redeploy**: copy the skel + updated `gen/`/`src/matmul_main.c` to the
   device, rebuild the CPU binary, run — see `README.md`'s one-time setup
   steps.
6. **Smoke-test at a small size first** (`-n 64`) and check `Result: PASS`
   before trusting any larger sweep.

---

## 5. Simulator vs. real hardware

The Hexagon **simulator config used for `run_all_experiments.ipynb` targets
v69** (no v69 silicon is available to us); the **RUBIK Pi 3's real CDSP is
v68** — a different, available chip in the same architecture family. Real
hardware here validates that the kernels' *design principles* (HVX beats
scalar, dynamic dispatch beats static, threading saturates once vector units
are exhausted) hold on real silicon, not that raw pcycle counts should match
the v69 simulator 1:1 — the two are different chips.

One concrete, explainable difference: the **v69 simulator config models 4 HVX
vector units; real v68 CDSP silicon has only 2**. `qurt_hvx_lock` blocks
whichever worker threads can't grab one of the available contexts, so on real
hardware `HVX+MT`/`HVX+MT+Static` throughput flattens out at `t=2` — visible
directly in `results_real_device/sim_vs_real_exp3_vs_threads.png` — while the
simulator keeps improving up to `t=4`. Single-threaded `HVX` (only ever needs
1 unit) and `QHL SDK` track the simulator closely regardless.

See `results_real_device/` and `results_dsp_vs_cpu/` at the repo root for the
full real-hardware sweeps, plots, and the DSP-vs-ARM-CPU-only comparison.

---

## 6. Status

- ✅ Skel builds for v68 (`libmatmul_skel.so`, standard SDK link), all 7 methods.
- ✅ CPU app builds natively on the device and links the real `libcdsprpc`.
- ✅ **All 7 variants run on the real CDSP and PASS** (RUBIK Pi 3, domain 3,
  unsigned PD) against a CPU reference checksum, for n up to 2048 (see
  `results_real_device/` at the repo root for the full sweep).
- The former platform blocker (`AEE_EUNABLETOLOAD` 0x80000600 at dynamic PD
  creation) was a broken firmware↔shell pairing in the device's
  `linux-firmware-dragonwing` image; fixed by installing upstream
  `qcm6490/cdsp.mbn` (c3-00134) plus the matching shell set — see
  [`../../hexagon_rubikpi3_setup.md`](../../hexagon_rubikpi3_setup.md).
- Known limitation: matrix sizes that are **not a multiple of 32** silently
  compute the *wrong result* on the HVX-based methods (`hvx`, `hvx_mt`,
  `hvx_mt_static`) on real hardware — Hexagon's HVX vmem load/store rounds
  unaligned addresses down instead of faulting. Always use `n % 32 == 0` for
  those three methods.
