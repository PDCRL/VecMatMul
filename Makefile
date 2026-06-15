# =============================================================================
# VecMatMul - Standalone Makefile
# Vectorized (HVX) matrix-multiplication kernels for the Qualcomm Hexagon DSP
# running under the QuRT RTOS, built and run on the Hexagon simulator.
#
# This Makefile is self-contained: it builds every .c file in this folder and
# runs it on the Hexagon simulator. All matrices are generated in-memory; the
# only run-time input is the triple "N K T" (see RUN PARAMETERS below).
#
# Quick start:
#   export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.5.0.0   # or pass on cmdline
#   make all                  # build every kernel
#   make sim-vectorized       # build + run the single-threaded HVX kernel
#   make sim-vectorized-mt N=128 K=16 T=4   # run MT kernel, 128x128, 4 threads
#   make help                 # full list of targets
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Toolchain / SDK discovery (override any of these on the command line)
# -----------------------------------------------------------------------------

# Path to the installed Hexagon SDK. Honour the environment variable if set,
# otherwise fall back to a common default. Override with:
#   make HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.5.0.0 <target>
HEXAGON_SDK_ROOT ?= /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.5.0.0

# Hexagon architecture version (v68, v69, v73, ...). Must match the QuRT image
# and, for the qhl_sdk target, an available QHL prebuilt (see section 4).
HEX_VERSION ?= v69

# Auto-detect the HEXAGON_Tools directory inside the SDK (e.g. .../19.0.07).
# Override with: make HEXAGON_TOOLS_ROOT=/path/to/HEXAGON_Tools/<version>
HEXAGON_TOOLS_ROOT ?= $(firstword $(sort $(wildcard $(HEXAGON_SDK_ROOT)/tools/HEXAGON_Tools/*)))

# Major tool version (e.g. 19 from "19.0.07"), used to locate QHL prebuilts.
TOOL_MAJOR := $(firstword $(subst ., ,$(notdir $(HEXAGON_TOOLS_ROOT))))

# Compiler / simulator. If these are not on PATH, point at them explicitly:
#   make CC_HEXAGON=$(HEXAGON_TOOLS_ROOT)/Tools/bin/hexagon-clang ...
CC_HEXAGON ?= hexagon-clang
HEXAGON_SIM ?= hexagon-sim

# -----------------------------------------------------------------------------
# 2. QuRT RTOS paths
# -----------------------------------------------------------------------------
QURT_DIR    = $(HEXAGON_SDK_ROOT)/rtos/qurt/compute$(HEX_VERSION)
QURT_LIB    = $(QURT_DIR)/lib
TARGET_DIR  = $(HEXAGON_TOOLS_ROOT)/Tools/target/hexagon/lib/$(HEX_VERSION)/G0
QCC_DIR     = $(TARGET_DIR)

# -----------------------------------------------------------------------------
# 3. Compiler / linker flags
# -----------------------------------------------------------------------------
CFLAGS_HEXAGON  = -O2 -m$(HEX_VERSION) -g
CFLAGS_HEXAGON += -Wall -Wextra -Wno-unused-parameter
CFLAGS_HEXAGON += -ffast-math
CFLAGS_HEXAGON += -I$(QURT_DIR)/include
CFLAGS_HEXAGON += -I$(QURT_DIR)/include/qurt
CFLAGS_HEXAGON += -I$(QURT_DIR)/include/posix

# HVX vector extensions (128-byte vectors, QFloat) for the vectorized kernels.
CFLAGS_HVX = $(CFLAGS_HEXAGON) -mhvx -mhvx-length=128B -mhvx-qfloat

# QuRT executable linker flags.
LDFLAGS_HEXAGON  = -m$(HEX_VERSION) -g
LDFLAGS_HEXAGON += -nodefaultlibs -nostdlib
LDFLAGS_HEXAGON += -Wl,--section-start -Wl,.interp=0x23000000
LDFLAGS_HEXAGON += -Wl,--dynamic-linker= -Wl,--force-dynamic
LDFLAGS_HEXAGON += -Wl,-E -Wl,-z -Wl,muldefs
LDFLAGS_HEXAGON += -Wl,--whole-archive

# QuRT libraries (order matters!).
QURT_INIT_LIBS  = $(TARGET_DIR)/init.o
QURT_LINK_LIBS  = $(QURT_LIB)/crt1.o
QURT_LINK_LIBS += $(QURT_LIB)/debugmon.o
QURT_LINK_LIBS += $(QURT_LIB)/libqurt.a
QURT_LINK_LIBS += $(TARGET_DIR)/libc.a
QURT_LINK_LIBS += $(QCC_DIR)/libqcc.a
QURT_LINK_LIBS += $(TARGET_DIR)/libhexagon.a
QURT_LINK_LIBS += $(QURT_LIB)/libqurtcfs.a
QURT_LINK_LIBS += $(QURT_LIB)/libtimer_island.a
QURT_LINK_LIBS += $(QURT_LIB)/libtimer_main.a
QURT_LINK_LIBS += $(QURT_LIB)/libposix.a
QURT_FINI_LIBS  = $(TARGET_DIR)/fini.o

# -----------------------------------------------------------------------------
# 4. QHL HVX BLAS library (only needed for the qhl_sdk reference target)
# Prebuilt dir naming: hexagon_toolv<TOOL_MAJOR>_<HEX_VERSION>, e.g. .._v19_v69.
# -----------------------------------------------------------------------------
QHL_TRIPLE      = hexagon_toolv$(TOOL_MAJOR)_$(HEX_VERSION)
QHL_HVX_LIB_DIR = $(HEXAGON_SDK_ROOT)/libs/qhl_hvx/prebuilt/$(QHL_TRIPLE)
QHL_LIB_DIR     = $(HEXAGON_SDK_ROOT)/libs/qhl/prebuilt/$(QHL_TRIPLE)
QHL_HVX_INC     = $(HEXAGON_SDK_ROOT)/libs/qhl_hvx/inc/qhblas_hvx
CFLAGS_QHL      = $(CFLAGS_HVX) -I$(QHL_HVX_INC)

# -----------------------------------------------------------------------------
# 5. Sources / targets
# -----------------------------------------------------------------------------
BUILD_DIR   = build
HEXAGON_DIR = $(BUILD_DIR)/hexagon

# Plain QuRT kernels (no HVX).
SCALAR_PROGS = matmul_sequential matmul_experiment matmul_dynamic
# HVX-vectorized kernels (built with CFLAGS_HVX).
HVX_PROGS    = matmul_vectorized matmul_vectorized_mt matmul_vectorized_mt_static
# Reference kernel using the SDK's QHL HVX BLAS (built + linked specially).
QHL_PROGS    = matmul_qhl_sdk

ALL_PROGS = $(SCALAR_PROGS) $(HVX_PROGS) $(QHL_PROGS)

# -----------------------------------------------------------------------------
# 6. RUN PARAMETERS (written to input.txt before each sim run)
#    N = matrix dimension (square, N x N)
#    K = grid / block size used by the tiled & multithreaded kernels
#    T = number of QuRT threads (multithreaded kernels only)
#    Single-threaded kernels read all three but only use N.
# -----------------------------------------------------------------------------
N ?= 64
K ?= 8
T ?= 4

# -----------------------------------------------------------------------------
# 7. Simulator cosimulation config
# -----------------------------------------------------------------------------
QURT_OS       = $(QURT_DIR)/sdksim_bin/runelf.pbn
QURT_MODEL    = $(QURT_DIR)/debugger/lnx64/qurt_model.so
HEXAGON_ISS_DIR = $(HEXAGON_TOOLS_ROOT)/Tools/lib/iss
OSAM_CFG      = $(HEXAGON_DIR)/osam.cfg
Q6SS_CFG      = $(HEXAGON_DIR)/q6ss.cfg

# =============================================================================
# Build rules
# =============================================================================
.PHONY: all clean help check-tools \
        $(ALL_PROGS) \
        sim-sequential sim-experiment sim-dynamic \
        sim-vectorized sim-vectorized-mt sim-vectorized-mt-static \
        sim-qhl-sdk sim-all

all: $(ALL_PROGS)
	@echo ""
	@echo "Built all kernels into $(HEXAGON_DIR)/"

$(HEXAGON_DIR):
	mkdir -p $(HEXAGON_DIR)

check-tools:
	@command -v $(CC_HEXAGON) >/dev/null 2>&1 || { \
		echo "ERROR: '$(CC_HEXAGON)' not found."; \
		echo "  Install the Hexagon SDK and set HEXAGON_SDK_ROOT, or add the"; \
		echo "  toolchain bin dir to PATH (source <SDK>/setup_sdk_env.source)."; \
		exit 1; }

# --- generic build helpers --------------------------------------------------
# $(1)=program name, called with the right CFLAGS via the rules below.
define LINK_QURT
	$(CC_HEXAGON) $(LDFLAGS_HEXAGON) -o $(HEXAGON_DIR)/$(1) \
		$(QURT_INIT_LIBS) $(HEXAGON_DIR)/$(1).o \
		$(QURT_LINK_LIBS) $(QURT_FINI_LIBS)
	@echo "Built: $(HEXAGON_DIR)/$(1)"
endef

# Scalar QuRT kernels.
$(SCALAR_PROGS): %: %.c | $(HEXAGON_DIR) check-tools
	$(CC_HEXAGON) $(CFLAGS_HEXAGON) -c -o $(HEXAGON_DIR)/$@.o $<
	$(call LINK_QURT,$@)

# HVX-vectorized kernels.
$(HVX_PROGS): %: %.c | $(HEXAGON_DIR) check-tools
	$(CC_HEXAGON) $(CFLAGS_HVX) -c -o $(HEXAGON_DIR)/$@.o $<
	$(call LINK_QURT,$@)

# QHL SDK reference kernel: compile with QHL include path, link against the
# prebuilt QHL HVX BLAS archives (outside the --whole-archive group).
matmul_qhl_sdk: matmul_qhl_sdk.c | $(HEXAGON_DIR) check-tools
	@test -f $(QHL_HVX_LIB_DIR)/libqhblas_hvx.a || { \
		echo "ERROR: QHL prebuilt not found at $(QHL_HVX_LIB_DIR)"; \
		echo "  Pick a HEX_VERSION with a matching qhl prebuilt, e.g. v68/v69/v73."; \
		exit 1; }
	$(CC_HEXAGON) $(CFLAGS_QHL) -c -o $(HEXAGON_DIR)/$@.o $<
	$(CC_HEXAGON) $(LDFLAGS_HEXAGON) -o $(HEXAGON_DIR)/$@ \
		$(QURT_INIT_LIBS) $(HEXAGON_DIR)/$@.o \
		-Wl,--no-whole-archive \
		$(QHL_HVX_LIB_DIR)/libqhblas_hvx.a \
		$(QHL_HVX_LIB_DIR)/libqhmath_hvx.a \
		$(QHL_LIB_DIR)/libqhmath.a \
		-Wl,--whole-archive \
		$(QURT_LINK_LIBS) $(QURT_FINI_LIBS)
	@echo "Built: $(HEXAGON_DIR)/$@"

# =============================================================================
# Simulator config + input generation
# =============================================================================
$(OSAM_CFG): | $(HEXAGON_DIR)
	@echo "$(QURT_MODEL)" > $(OSAM_CFG)

$(Q6SS_CFG): | $(HEXAGON_DIR)
	@echo "$(HEXAGON_ISS_DIR)/qtimer.so --csr_base=0xFC900000 --irq_p=3 --freq=19200000 --cnttid=1" > $(Q6SS_CFG)
	@echo "$(HEXAGON_ISS_DIR)/l2vic.so 32 0xFC910000" >> $(Q6SS_CFG)

# Write the run parameters into the simulator's filesystem (build/hexagon),
# which every kernel reads as input.txt.
.PHONY: input
input: | $(HEXAGON_DIR)
	@echo "$(N) $(K) $(T)" > $(HEXAGON_DIR)/input.txt
	@echo "input.txt = '$(N) $(K) $(T)'  (N=size, K=grid, T=threads)"

# $(1)=program name. Build it, refresh input/config, run on the simulator.
define RUN_SIM
	@command -v $(HEXAGON_SIM) >/dev/null 2>&1 || { echo "ERROR: $(HEXAGON_SIM) not found."; exit 1; }
	@echo "Running $(1) on Hexagon simulator (N=$(N) K=$(K) T=$(T))..."
	$(HEXAGON_SIM) -m$(HEX_VERSION) --simulated_returnval --timing \
		--usefs $(HEXAGON_DIR) \
		--pmu_statsfile $(HEXAGON_DIR)/pmu_stats.txt \
		--cosim_file $(Q6SS_CFG) \
		--l2tcm_base 0xd800 \
		--subsystem_base 0xFC90 \
		--rtos $(OSAM_CFG) $(QURT_OS) -- \
		$(HEXAGON_DIR)/$(1)
	@echo "--- output.txt ---"; cat $(HEXAGON_DIR)/output.txt 2>/dev/null || true
endef

sim-sequential:          matmul_sequential          input $(OSAM_CFG) $(Q6SS_CFG)
	$(call RUN_SIM,matmul_sequential)
sim-experiment:          matmul_experiment          input $(OSAM_CFG) $(Q6SS_CFG)
	$(call RUN_SIM,matmul_experiment)
sim-dynamic:             matmul_dynamic             input $(OSAM_CFG) $(Q6SS_CFG)
	$(call RUN_SIM,matmul_dynamic)
sim-vectorized:          matmul_vectorized          input $(OSAM_CFG) $(Q6SS_CFG)
	$(call RUN_SIM,matmul_vectorized)
sim-vectorized-mt:       matmul_vectorized_mt       input $(OSAM_CFG) $(Q6SS_CFG)
	$(call RUN_SIM,matmul_vectorized_mt)
sim-vectorized-mt-static: matmul_vectorized_mt_static input $(OSAM_CFG) $(Q6SS_CFG)
	$(call RUN_SIM,matmul_vectorized_mt_static)
sim-qhl-sdk:             matmul_qhl_sdk             input $(OSAM_CFG) $(Q6SS_CFG)
	$(call RUN_SIM,matmul_qhl_sdk)

# Build + run every kernel in sequence with the current N/K/T.
sim-all: sim-sequential sim-experiment sim-dynamic \
         sim-vectorized sim-vectorized-mt sim-vectorized-mt-static sim-qhl-sdk
	@echo ""
	@echo "Ran all kernels (N=$(N) K=$(K) T=$(T))."

# =============================================================================
clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned $(BUILD_DIR)/"

help:
	@echo "VecMatMul - Hexagon HVX matrix-multiplication kernels"
	@echo "====================================================="
	@echo ""
	@echo "Build targets (output in $(HEXAGON_DIR)/):"
	@echo "  make all                      Build every kernel"
	@echo "  make matmul_sequential        Scalar single-threaded baseline"
	@echo "  make matmul_experiment        Scalar, static row-block threads"
	@echo "  make matmul_dynamic           Scalar, atomic-counter dynamic threads"
	@echo "  make matmul_vectorized        HVX single-threaded"
	@echo "  make matmul_vectorized_mt     HVX + dynamic multithreading"
	@echo "  make matmul_vectorized_mt_static  HVX + static multithreading"
	@echo "  make matmul_qhl_sdk           SDK QHL HVX BLAS reference"
	@echo ""
	@echo "Run-on-simulator targets (build, then run):"
	@echo "  make sim-sequential | sim-experiment | sim-dynamic"
	@echo "  make sim-vectorized | sim-vectorized-mt | sim-vectorized-mt-static"
	@echo "  make sim-qhl-sdk"
	@echo "  make sim-all                  Run all kernels in sequence"
	@echo ""
	@echo "Run parameters (override on the command line):"
	@echo "  N=$(N)   matrix dimension (N x N)"
	@echo "  K=$(K)    grid/block size (tiled & MT kernels)"
	@echo "  T=$(T)    number of threads (MT kernels)"
	@echo "  e.g.  make sim-vectorized-mt N=256 K=32 T=4"
	@echo ""
	@echo "Toolchain settings:"
	@echo "  HEXAGON_SDK_ROOT = $(HEXAGON_SDK_ROOT)"
	@echo "  HEXAGON_TOOLS_ROOT = $(HEXAGON_TOOLS_ROOT)"
	@echo "  HEX_VERSION      = $(HEX_VERSION)"
	@echo ""
	@echo "  make clean                    Remove the build directory"
