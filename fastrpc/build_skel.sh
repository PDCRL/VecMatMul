#!/usr/bin/env bash
# =============================================================================
# build_skel.sh — regenerate FastRPC stubs and build the Hexagon CDSP skel
# (libmatmul_skel.so) for the matmul FastRPC port.
#
# Runs on the x86_64 host with the Hexagon SDK (hexagon-clang). Standard SDK
# link: libc.so/libgcc.so stay as DT_NEEDED and are provided in-PD by the
# fastrpc shell. (Requires a shell/firmware pair that actually matches — see
# ../../hexagon_rubikpi3_setup.md; the older "self-contained 0 DT_NEEDED +
# static libc.a" workaround breaks on current shells, whose loader does not
# export sys_sbrk.)
#
#   export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.5.0.0   # or use the default
#   ./build_skel.sh                                        # v68 (QCS6490)
#   HEX_ARCH=v73 ./build_skel.sh                           # other archs
# =============================================================================
set -euo pipefail

SDK="${HEXAGON_SDK_ROOT:-/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.5.0.0}"
ARCH="${HEX_ARCH:-v68}"
TOOLS="$(ls -d "$SDK"/tools/HEXAGON_Tools/* | sort | tail -1)/Tools"
CLANG="$TOOLS/bin/hexagon-clang"
QAIC="$SDK/ipc/fastrpc/qaic/Ubuntu/qaic"
LP="$TOOLS/target/hexagon/lib/$ARCH/G0/pic"      # static PIC C runtime archives
Q="$SDK/rtos/qurt/compute$ARCH/include"          # QuRT headers (threads/HVX lock)
HERE="$(cd "$(dirname "$0")" && pwd)"
BD="$HERE/build/hexagon"; mkdir -p "$BD" "$HERE/gen"

# QHL HVX BLAS prebuilt (needed by the matmul_qhl_sdk method).
TOOL_MAJOR="$(basename "$(dirname "$TOOLS")" | cut -d. -f1)"
QHL_HVX_DIR="$SDK/libs/qhl_hvx/prebuilt/hexagon_toolv${TOOL_MAJOR}_${ARCH}"
QHL_HVX_INC="$SDK/libs/qhl_hvx/inc/qhblas_hvx"

echo "SDK=$SDK  ARCH=$ARCH  QHL_HVX_DIR=$QHL_HVX_DIR"

# 1. Generate stub/skel/header from the IDL.
"$QAIC" -mdll -o "$HERE/gen" \
  -I"$SDK/incs" -I"$SDK/incs/stddef" -I"$SDK/ipc/fastrpc/incs" \
  "$HERE/matmul.idl"

INCS="-I$HERE/gen -I$SDK/incs -I$SDK/incs/stddef -I$SDK/ipc/fastrpc/incs -I$Q -I$Q/qurt -I$Q/posix -I$QHL_HVX_INC"
CFLAGS="-m$ARCH -G0 -fpic -O2 -mhvx -mhvx-length=128B -mhvx-qfloat -Wall"

# 2. Compile the skel dispatch + the ported kernel implementations.
"$CLANG" $CFLAGS $INCS -c "$HERE/gen/matmul_skel.c" -o "$BD/matmul_skel.o"
"$CLANG" $CFLAGS $INCS -c "$HERE/src/matmul_imp.c"  -o "$BD/matmul_imp.o"

# 3. Standard SDK shared link: libc.so/libgcc.so as DT_NEEDED (shell-provided).
#    QuRT/HVX/HAP symbols stay undefined and are resolved by the DSP shell.
#    QHL HVX BLAS is linked in statically (matmul_qhl_sdk method).
#    -lc is explicit + --no-as-needed: the QHL archives satisfy some libc-ish
#    symbols (memcpy etc.) statically, which makes the linker's default
#    (implicit) libc.so pull disappear — force the NEEDED tag back so the
#    shell resolves malloc/free/memset/etc. at PD load time (see README).
"$CLANG" -m$ARCH -G0 -fpic -Wl,-Bsymbolic \
  -shared -o "$BD/libmatmul_skel.so" -Wl,-soname,libmatmul_skel.so \
  "$BD/matmul_skel.o" "$BD/matmul_imp.o" \
  "$QHL_HVX_DIR/libqhblas_hvx.a" "$QHL_HVX_DIR/libqhdsp_hvx.a" "$QHL_HVX_DIR/libqhmath_hvx.a" \
  -L"$LP" -Wl,--no-as-needed -lc

echo "Built $BD/libmatmul_skel.so"
readelf -dW "$BD/libmatmul_skel.so" | grep -E "NEEDED|SONAME"
