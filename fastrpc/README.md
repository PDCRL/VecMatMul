# matmul FastRPC — run steps

Runs the VecMatMul kernels on the real Hexagon CDSP (RUBIK Pi 3 / QCS6490,
v68) via FastRPC. All commands below are fully written out (device IP
`172.24.132.68`, SDK at `/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.5.0.0`) —
copy-paste as-is, or substitute your own paths/IP.

## One-time setup (after editing any `.c`/`.idl` file)

**1. Build the DSP skel — on the x86_64 host:**
```bash
cd /home/vishnu/experiment4/VecMatMul/fastrpc
HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.5.0.0 ./build_skel.sh
```
Produces `build/hexagon/libmatmul_skel.so`.

**2. Copy the skel and CPU-side generated files to the device:**
```bash
scp build/hexagon/libmatmul_skel.so ubuntu@172.24.132.68:~/libmatmul_skel.so.new
scp gen/matmul.h gen/matmul_stub.c ubuntu@172.24.132.68:~/matmul-fastrpc/gen/
scp src/matmul_main.c ubuntu@172.24.132.68:~/matmul-fastrpc/src/
```

**3. Deploy the skel where the DSP shell looks for it — on the device:**
```bash
ssh ubuntu@172.24.132.68 'sudo cp ~/libmatmul_skel.so.new /usr/lib/dsp/cdsp/libmatmul_skel.so'
```

**4. Rebuild the CPU binary — on the device:**
```bash
ssh ubuntu@172.24.132.68 '
cd ~/matmul-fastrpc
gcc -O2 -o matmul src/matmul_main.c gen/matmul_stub.c \
  -Iinc/incs -Iinc/incs/stddef -Iinc/fastrpc -Iinc/rpcmem -Igen -Isrc \
  -D_GNU_SOURCE -lcdsprpc -ldl -lpthread
'
```

## Run each variant manually

SSH into the device first:
```bash
ssh ubuntu@172.24.132.68
cd ~/matmul-fastrpc
```

Then run any of the 7 variants (example: `n=256`, `k=32`, `t=4`):

```bash
# Sequential — scalar, single-threaded (k/t ignored)
sudo ./matmul -n 256 -k 32 -t 4 -d 3 -U 1 -m seq

# Static — scalar, grid work statically striped across threads
sudo ./matmul -n 256 -k 32 -t 4 -d 3 -U 1 -m static

# Dynamic — scalar, grid work pulled from a shared atomic counter
sudo ./matmul -n 256 -k 32 -t 4 -d 3 -U 1 -m dynamic

# HVX — hand-vectorized, single-threaded (k/t ignored; n must be a multiple of 32)
sudo ./matmul -n 256 -k 32 -t 4 -d 3 -U 1 -m hvx

# HVX+MT — HVX + dynamic atomic-counter multithreading (best variant so far)
sudo ./matmul -n 256 -k 32 -t 4 -d 3 -U 1 -m hvx_mt

# HVX+MT+Static — HVX + statically striped multithreading
sudo ./matmul -n 256 -k 32 -t 4 -d 3 -U 1 -m hvx_mt_static

# QHL SDK — vendor reference GEMM, single-threaded (k/t ignored)
sudo ./matmul -n 256 -k 32 -t 4 -d 3 -U 1 -m qhl_sdk
```

Each prints:
```
DSP wall pcycles: 2492323
DSP call wall ms: 2.672
Checksum (DSP):   ...
Checksum (CPU):   ...
Result: PASS
```
Always check the last line says `PASS` before trusting a number — sizes that
aren't a multiple of 32 silently corrupt results on `hvx`/`hvx_mt`/`hvx_mt_static`.

## Run the pure ARM CPU baseline (no DSP)

**Build (one-time, on the device):**
```bash
ssh ubuntu@172.24.132.68 '
cd ~/cpu_bench
gcc -O2 -o matmul_cpu_bench matmul_cpu_bench.c -lpthread
'
```

**Run:**
```bash
ssh ubuntu@172.24.132.68 'cd ~/cpu_bench && ./matmul_cpu_bench -n 256 -m seq'
ssh ubuntu@172.24.132.68 'cd ~/cpu_bench && ./matmul_cpu_bench -n 256 -t 4 -m static'
ssh ubuntu@172.24.132.68 'cd ~/cpu_bench && ./matmul_cpu_bench -n 256 -t 4 -m dynamic'
```
Prints `CPU wall ms: <float>` and `Result: PASS/FAIL`.

## Batch-sweep many sizes at once (instead of manual one-off runs)

```bash
cd /home/vishnu/experiment4/VecMatMul/fastrpc
python3 run_real_experiments.py --host ubuntu@172.24.132.68 --reps 3
```
Writes `results_real_device/real_exp{1,2,3}_vs_{size,k,threads}.csv` at the
repo root.

## Flag reference

| Flag | Meaning |
|---|---|
| `-n` | Matrix dimension N (N×N). Use a multiple of 32 for `hvx`/`hvx_mt`/`hvx_mt_static`. |
| `-k` | Grid/tile size. Ignored by `seq`/`hvx`/`qhl_sdk`. |
| `-t` | Thread count. Ignored by `seq`/`hvx`/`qhl_sdk`. |
| `-d 3` | FastRPC domain = CDSP (compute DSP). Always 3 here. |
| `-U 1` | Request an unsigned PD (required — our skel isn't Qualcomm-signed). |
| `-m` | Which kernel variant: `seq`, `static`, `dynamic`, `hvx`, `hvx_mt`, `hvx_mt_static`, `qhl_sdk`. |

## If something breaks

- `AEE_EUNABLETOLOAD` (`0x80000600`) at PD load → firmware/shell pairing
  issue, see `../../hexagon_rubikpi3_setup.md`.
- Skel fails to resolve `malloc`/`free`/`memset` at load → rebuild with
  `readelf -dW build/hexagon/libmatmul_skel.so | grep NEEDED` and confirm
  both `libc.so` and `libgcc.so` are listed; if not, see `build_skel.sh`'s
  link step (`-Wl,--no-as-needed -lc`).
- For what each script/term actually does under the hood (IDL, stub, skel,
  PD, etc.) and how to add a brand-new kernel variant, see
  [`CONCEPTS.md`](CONCEPTS.md).
