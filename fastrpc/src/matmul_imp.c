/*============================================================================
  matmul_imp.c  —  DSP (skel) implementation of the FastRPC `matmul` interface.

  Port of all VecMatMul standalone QuRT kernels (matmul_*.c) to run inside a FastRPC
  user PD on the Hexagon CDSP, sharing one skel/handle. Each entry point
  below corresponds to exactly one standalone kernel; the compute bodies are
  carried over verbatim from their source files. Only the harness differs:
    * no main()/input.txt/output.txt — A,B,C and n,k,t arrive over FastRPC;
    * no QURT_DECLARE_STATIC_HEAP — the PD provides the heap;
    * printf(...) -> FARF(HIGH, ...) (DSP logging);
    * matmul_open()/matmul_close() added for the remote_handle64 interface;
    * per-kernel thread-stack scratch is shared (thread_stacks[]) since only
      one RPC call executes at a time on a given handle.

  A,B,C are caller (CPU) buffers mapped into the PD by FastRPC; they are page
  aligned (>=128B) so the HVX vector loads/stores keep their alignment as long
  as n is a multiple of 32 (see README).

  Provenance:
    matmul_seq        <- VecMatMul/matmul_sequential.c
    matmul_static_     <- VecMatMul/matmul_experiment.c
    matmul_dynamic     <- VecMatMul/matmul_dynamic.c
    matmul_hvx         <- VecMatMul/matmul_vectorized.c
    matmul_mt          <- VecMatMul/matmul_vectorized_mt.c
    matmul_hvx_mt_static <- VecMatMul/matmul_vectorized_mt_static.c
    matmul_qhl_sdk     <- VecMatMul/matmul_qhl_sdk.c
============================================================================*/

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <qurt.h>
#include <qurt_thread.h>
#include <qurt_atomic_ops.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "qhblas_hvx.h"

#define FARF_HIGH 1
#include "HAP_farf.h"
#include "matmul.h"

__attribute__((used, visibility("default"))) const char TS[] =
    "\nTIMESTAMP=" __DATE__ " " __TIME__ "\n";

/* HVX 128B mode: 128 bytes = 32 single-precision floats per vector */
#define HVX_FLOAT_LANES 32

#define MAX_THREADS 10
#define STACK_SIZE (8 * 1024)

/* Shared scratch stacks — safe because only one RPC call runs at a time. */
static char thread_stacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(8)));

static inline uint64_t read_cycles(void) {
    uint64_t cycles;
    __asm__ __volatile__(
        "%0 = c15:14"
        : "=r"(cycles)
        :
        : "memory"
    );
    return cycles;
}

/* ==================================================================== */
/* Sequential — VecMatMul/matmul_sequential.c                            */
/* ==================================================================== */

static uint64_t seq_compute(const float* A, const float* B, float* C, int n) {
    uint64_t start = read_cycles();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < n; p++) {
                sum += A[i * n + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }

    return read_cycles() - start;
}

/* ==================================================================== */
/* Static — VecMatMul/matmul_experiment.c                                */
/* ==================================================================== */

typedef struct {
    float* A;
    float* B;
    float* C;
    int n;
    int k;
    int num_threads;
    int thread_id;
    int grids_per_dim;
    uint64_t cycles;
} StaticThreadContext;

static void static_grid_worker(void* arg) {
    StaticThreadContext* ctx = (StaticThreadContext*)arg;

    float* A = ctx->A;
    float* B = ctx->B;
    float* C = ctx->C;
    int n = ctx->n;
    int k = ctx->k;
    int num_threads = ctx->num_threads;
    int thread_id = ctx->thread_id;
    int grids_per_dim = ctx->grids_per_dim;
    int total_grids = grids_per_dim * grids_per_dim;

    uint64_t start = read_cycles();

    for (int grid_num = thread_id; grid_num < total_grids; grid_num += num_threads) {
        int grid_row = grid_num / grids_per_dim;
        int grid_col = grid_num % grids_per_dim;

        int row_start = grid_row * k;
        int row_end = row_start + k;
        int col_start = grid_col * k;
        int col_end = col_start + k;

        if (row_end > n) row_end = n;
        if (col_end > n) col_end = n;

        for (int i = row_start; i < row_end; i++) {
            for (int j = col_start; j < col_end; j++) {
                float sum = 0.0f;
                for (int p = 0; p < n; p++) {
                    sum += A[i * n + p] * B[p * n + j];
                }
                C[i * n + j] = sum;
            }
        }
    }

    ctx->cycles = read_cycles() - start;
    qurt_thread_exit(0);
}

static uint64_t static_grid_parallel(float* A, float* B, float* C, int n, int k, int num_threads) {
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    if (num_threads < 1) num_threads = 1;

    int grids_per_dim = (n + k - 1) / k;
    int total_grids = grids_per_dim * grids_per_dim;

    if (num_threads > total_grids) num_threads = total_grids;

    FARF(HIGH, "Static grid matmul: n=%d, k=%d, threads=%d", n, k, num_threads);

    StaticThreadContext contexts[MAX_THREADS];
    qurt_thread_t threads[MAX_THREADS];
    qurt_thread_attr_t attrs[MAX_THREADS];

    uint64_t total_start = read_cycles();

    for (int t = 0; t < num_threads; t++) {
        contexts[t].A = A;
        contexts[t].B = B;
        contexts[t].C = C;
        contexts[t].n = n;
        contexts[t].k = k;
        contexts[t].num_threads = num_threads;
        contexts[t].thread_id = t;
        contexts[t].grids_per_dim = grids_per_dim;
        contexts[t].cycles = 0;

        qurt_thread_attr_init(&attrs[t]);
        qurt_thread_attr_set_stack_addr(&attrs[t], (void*)thread_stacks[t]);
        qurt_thread_attr_set_stack_size(&attrs[t], STACK_SIZE);
        qurt_thread_attr_set_priority(&attrs[t], 100);

        int ret = qurt_thread_create(&threads[t], &attrs[t], static_grid_worker, &contexts[t]);
        if (ret != QURT_EOK) {
            FARF(HIGH, "error: thread creation failed (ret=%d)", ret);
            for (int i = 0; i < t; i++) {
                int status;
                qurt_thread_join(threads[i], &status);
            }
            return 0;
        }
    }

    for (int t = 0; t < num_threads; t++) {
        int status;
        qurt_thread_join(threads[t], &status);
    }

    return read_cycles() - total_start;
}

/* ==================================================================== */
/* Dynamic — VecMatMul/matmul_dynamic.c                                  */
/* ==================================================================== */

typedef struct {
    float* A;
    float* B;
    float* C;
    int n;
    int k;
    int grids_per_dim;
    int total_grids;
    unsigned int* counter;
    int thread_id;
    uint64_t cycles;
    int grids_done;
} DynamicThreadContext;

static void dynamic_worker(void* arg) {
    DynamicThreadContext* ctx = (DynamicThreadContext*)arg;

    float* A = ctx->A;
    float* B = ctx->B;
    float* C = ctx->C;
    int n = ctx->n;
    int k = ctx->k;
    int grids_per_dim = ctx->grids_per_dim;
    int total_grids = ctx->total_grids;
    unsigned int* counter = ctx->counter;

    int grids_done = 0;
    uint64_t start = read_cycles();

    while (1) {
        unsigned int grid_num = qurt_atomic_inc_return(counter) - 1;
        if ((int)grid_num >= total_grids) break;

        int grid_row = (int)grid_num / grids_per_dim;
        int grid_col = (int)grid_num % grids_per_dim;

        int row_start = grid_row * k;
        int row_end = row_start + k;
        int col_start = grid_col * k;
        int col_end = col_start + k;

        if (row_end > n) row_end = n;
        if (col_end > n) col_end = n;

        for (int i = row_start; i < row_end; i++) {
            for (int j = col_start; j < col_end; j++) {
                float sum = 0.0f;
                for (int p = 0; p < n; p++) {
                    sum += A[i * n + p] * B[p * n + j];
                }
                C[i * n + j] = sum;
            }
        }

        grids_done++;
    }

    ctx->cycles = read_cycles() - start;
    ctx->grids_done = grids_done;
    qurt_thread_exit(0);
}

static uint64_t dynamic_parallel(float* A, float* B, float* C, int n, int k, int num_threads) {
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    if (num_threads < 1) num_threads = 1;

    int grids_per_dim = (n + k - 1) / k;
    int total_grids = grids_per_dim * grids_per_dim;

    if (num_threads > total_grids) num_threads = total_grids;

    FARF(HIGH, "Dynamic grid matmul: n=%d, k=%d, threads=%d", n, k, num_threads);

    unsigned int shared_counter = 0;

    DynamicThreadContext contexts[MAX_THREADS];
    qurt_thread_t threads[MAX_THREADS];
    qurt_thread_attr_t attrs[MAX_THREADS];

    uint64_t total_start = read_cycles();

    for (int t = 0; t < num_threads; t++) {
        contexts[t].A = A;
        contexts[t].B = B;
        contexts[t].C = C;
        contexts[t].n = n;
        contexts[t].k = k;
        contexts[t].grids_per_dim = grids_per_dim;
        contexts[t].total_grids = total_grids;
        contexts[t].counter = &shared_counter;
        contexts[t].thread_id = t;
        contexts[t].cycles = 0;
        contexts[t].grids_done = 0;

        qurt_thread_attr_init(&attrs[t]);
        qurt_thread_attr_set_stack_addr(&attrs[t], (void*)thread_stacks[t]);
        qurt_thread_attr_set_stack_size(&attrs[t], STACK_SIZE);
        qurt_thread_attr_set_priority(&attrs[t], 100);

        int ret = qurt_thread_create(&threads[t], &attrs[t], dynamic_worker, &contexts[t]);
        if (ret != QURT_EOK) {
            FARF(HIGH, "error: thread creation failed (ret=%d)", ret);
            for (int i = 0; i < t; i++) {
                int status;
                qurt_thread_join(threads[i], &status);
            }
            return 0;
        }
    }

    for (int t = 0; t < num_threads; t++) {
        int status;
        qurt_thread_join(threads[t], &status);
    }

    return read_cycles() - total_start;
}

/* ==================================================================== */
/* HVX single-threaded — VecMatMul/matmul_vectorized.c                   */
/* ==================================================================== */

static uint64_t hvx_single_compute(float* A, float* B, float* C, int n) {
    int vec_cols = n / HVX_FLOAT_LANES;

    int hvx_ret = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (hvx_ret != 0) {
        FARF(HIGH, "error: HVX lock failed (ret=%d)", hvx_ret);
        return 0;
    }

    uint64_t start = read_cycles();

    for (int i = 0; i < n; i++) {
        for (int jb = 0; jb < vec_cols; jb++) {
            int j = jb * HVX_FLOAT_LANES;
            HVX_Vector vqsum = Q6_V_vzero();

            for (int p = 0; p < n; p++) {
                float a_val = A[i * n + p];
                HVX_Vector va = Q6_V_vsplat_R(*(uint32_t*)&a_val);
                HVX_Vector vb = *(HVX_Vector*)&B[p * n + j];
                HVX_Vector vqprod = Q6_Vqf32_vmpy_VsfVsf(va, vb);
                vqsum = Q6_Vqf32_vadd_Vqf32Vqf32(vqsum, vqprod);
            }

            *(HVX_Vector*)&C[i * n + j] = Q6_Vsf_equals_Vqf32(vqsum);
        }

        for (int j = vec_cols * HVX_FLOAT_LANES; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < n; p++) {
                sum += A[i * n + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }

    uint64_t elapsed = read_cycles() - start;
    qurt_hvx_unlock();
    return elapsed;
}

/* ==================================================================== */
/* HVX + dynamic MT — VecMatMul/matmul_vectorized_mt.c                   */
/* ==================================================================== */

typedef struct {
    float* A;
    float* B;
    float* C;
    int n;
    int k;
    int grids_per_dim;
    int total_grids;
    unsigned int* counter;
    int thread_id;
    uint64_t cycles;
    int grids_done;
} MtThreadContext;

/* Compute one grid block [row_start..row_end) x [col_start..col_end) using HVX */
static inline void compute_grid_hvx(float* A, float* B, float* C,
                                    int n, int row_start, int row_end,
                                    int col_start, int col_end) {
    int block_width = col_end - col_start;
    int vec_cols = block_width / HVX_FLOAT_LANES;
    int tail_start = col_start + vec_cols * HVX_FLOAT_LANES;

    for (int i = row_start; i < row_end; i++) {
        for (int jb = 0; jb < vec_cols; jb++) {
            int j = col_start + jb * HVX_FLOAT_LANES;
            HVX_Vector vqsum = Q6_V_vzero();

            for (int p = 0; p < n; p++) {
                float a_val = A[i * n + p];
                HVX_Vector va = Q6_V_vsplat_R(*(uint32_t*)&a_val);
                HVX_Vector vb = *(HVX_Vector*)&B[p * n + j];
                HVX_Vector vqprod = Q6_Vqf32_vmpy_VsfVsf(va, vb);
                vqsum = Q6_Vqf32_vadd_Vqf32Vqf32(vqsum, vqprod);
            }

            *(HVX_Vector*)&C[i * n + j] = Q6_Vsf_equals_Vqf32(vqsum);
        }

        for (int j = tail_start; j < col_end; j++) {
            float sum = 0.0f;
            for (int p = 0; p < n; p++) {
                sum += A[i * n + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

static void mt_scalar_worker(void* arg) {
    MtThreadContext* ctx = (MtThreadContext*)arg;

    float* A = ctx->A;
    float* B = ctx->B;
    float* C = ctx->C;
    int n = ctx->n;
    int k = ctx->k;
    int grids_per_dim = ctx->grids_per_dim;
    int total_grids = ctx->total_grids;
    unsigned int* counter = ctx->counter;

    int grids_done = 0;
    uint64_t start = read_cycles();

    while (1) {
        unsigned int grid_num = qurt_atomic_inc_return(counter) - 1;
        if ((int)grid_num >= total_grids) break;

        int grid_row = (int)grid_num / grids_per_dim;
        int grid_col = (int)grid_num % grids_per_dim;

        int row_start = grid_row * k;
        int row_end = row_start + k;
        int col_start = grid_col * k;
        int col_end = col_start + k;

        if (row_end > n) row_end = n;
        if (col_end > n) col_end = n;

        for (int i = row_start; i < row_end; i++) {
            for (int j = col_start; j < col_end; j++) {
                float sum = 0.0f;
                for (int p = 0; p < n; p++) {
                    sum += A[i * n + p] * B[p * n + j];
                }
                C[i * n + j] = sum;
            }
        }

        grids_done++;
    }

    ctx->cycles = read_cycles() - start;
    ctx->grids_done = grids_done;
    qurt_thread_exit(0);
}

static void mt_hvx_worker(void* arg) {
    MtThreadContext* ctx = (MtThreadContext*)arg;

    int hvx_ret = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (hvx_ret != 0) {
        FARF(HIGH, "Thread %d: HVX lock failed (ret=%d), falling back to scalar",
             ctx->thread_id, hvx_ret);
        mt_scalar_worker(arg);
        return;
    }

    float* A = ctx->A;
    float* B = ctx->B;
    float* C = ctx->C;
    int n = ctx->n;
    int k = ctx->k;
    int grids_per_dim = ctx->grids_per_dim;
    int total_grids = ctx->total_grids;
    unsigned int* counter = ctx->counter;

    int grids_done = 0;
    uint64_t start = read_cycles();

    while (1) {
        unsigned int grid_num = qurt_atomic_inc_return(counter) - 1;
        if ((int)grid_num >= total_grids) {
            break;
        }

        int grid_row = (int)grid_num / grids_per_dim;
        int grid_col = (int)grid_num % grids_per_dim;

        int row_start = grid_row * k;
        int row_end = row_start + k;
        int col_start = grid_col * k;
        int col_end = col_start + k;

        if (row_end > n) row_end = n;
        if (col_end > n) col_end = n;

        compute_grid_hvx(A, B, C, n, row_start, row_end, col_start, col_end);

        grids_done++;
    }

    ctx->cycles = read_cycles() - start;
    ctx->grids_done = grids_done;

    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

static uint64_t hvx_mt_parallel(float* A, float* B, float* C,
                                 int n, int k, int num_threads) {
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    if (num_threads < 1) num_threads = 1;

    int grids_per_dim = (n + k - 1) / k;
    int total_grids = grids_per_dim * grids_per_dim;

    if (num_threads > total_grids) num_threads = total_grids;

    FARF(HIGH, "HVX + dynamic multithreaded matmul: n=%d, k=%d, threads=%d", n, k, num_threads);

    unsigned int shared_counter = 0;

    MtThreadContext contexts[MAX_THREADS];
    qurt_thread_t threads[MAX_THREADS];
    qurt_thread_attr_t attrs[MAX_THREADS];

    uint64_t total_start = read_cycles();

    for (int t = 0; t < num_threads; t++) {
        contexts[t].A = A;
        contexts[t].B = B;
        contexts[t].C = C;
        contexts[t].n = n;
        contexts[t].k = k;
        contexts[t].grids_per_dim = grids_per_dim;
        contexts[t].total_grids = total_grids;
        contexts[t].counter = &shared_counter;
        contexts[t].thread_id = t;
        contexts[t].cycles = 0;
        contexts[t].grids_done = 0;

        qurt_thread_attr_init(&attrs[t]);
        qurt_thread_attr_set_stack_addr(&attrs[t], (void*)thread_stacks[t]);
        qurt_thread_attr_set_stack_size(&attrs[t], STACK_SIZE);
        qurt_thread_attr_set_priority(&attrs[t], 100);

        int ret = qurt_thread_create(&threads[t], &attrs[t], mt_hvx_worker, &contexts[t]);
        if (ret != QURT_EOK) {
            FARF(HIGH, "error: thread creation failed (ret=%d)", ret);
            for (int i = 0; i < t; i++) {
                int status;
                qurt_thread_join(threads[i], &status);
            }
            return 0;
        }
    }

    for (int t = 0; t < num_threads; t++) {
        int status;
        qurt_thread_join(threads[t], &status);
    }

    return read_cycles() - total_start;
}

/* ==================================================================== */
/* HVX + static MT — VecMatMul/matmul_vectorized_mt_static.c             */
/* ==================================================================== */

typedef struct {
    float* A;
    float* B;
    float* C;
    int n;
    int k;
    int grids_per_dim;
    int total_grids;
    int num_threads;
    int thread_id;
    uint64_t cycles;
    int grids_done;
} MtStaticThreadContext;

static void mt_static_scalar_worker(void* arg) {
    MtStaticThreadContext* ctx = (MtStaticThreadContext*)arg;

    float* A = ctx->A;
    float* B = ctx->B;
    float* C = ctx->C;
    int n = ctx->n;
    int k = ctx->k;
    int grids_per_dim = ctx->grids_per_dim;
    int total_grids = ctx->total_grids;
    int num_threads = ctx->num_threads;
    int thread_id = ctx->thread_id;

    int grids_done = 0;
    uint64_t start = read_cycles();

    for (int grid_num = thread_id; grid_num < total_grids; grid_num += num_threads) {
        int grid_row = grid_num / grids_per_dim;
        int grid_col = grid_num % grids_per_dim;

        int row_start = grid_row * k;
        int row_end = row_start + k;
        int col_start = grid_col * k;
        int col_end = col_start + k;

        if (row_end > n) row_end = n;
        if (col_end > n) col_end = n;

        for (int i = row_start; i < row_end; i++) {
            for (int j = col_start; j < col_end; j++) {
                float sum = 0.0f;
                for (int p = 0; p < n; p++) {
                    sum += A[i * n + p] * B[p * n + j];
                }
                C[i * n + j] = sum;
            }
        }

        grids_done++;
    }

    ctx->cycles = read_cycles() - start;
    ctx->grids_done = grids_done;
    qurt_thread_exit(0);
}

static void mt_static_hvx_worker(void* arg) {
    MtStaticThreadContext* ctx = (MtStaticThreadContext*)arg;

    int hvx_ret = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (hvx_ret != 0) {
        FARF(HIGH, "Thread %d: HVX lock failed (ret=%d), falling back to scalar",
             ctx->thread_id, hvx_ret);
        mt_static_scalar_worker(arg);
        return;
    }

    float* A = ctx->A;
    float* B = ctx->B;
    float* C = ctx->C;
    int n = ctx->n;
    int k = ctx->k;
    int grids_per_dim = ctx->grids_per_dim;
    int total_grids = ctx->total_grids;
    int num_threads = ctx->num_threads;
    int thread_id = ctx->thread_id;

    int grids_done = 0;
    uint64_t start = read_cycles();

    for (int grid_num = thread_id; grid_num < total_grids; grid_num += num_threads) {
        int grid_row = grid_num / grids_per_dim;
        int grid_col = grid_num % grids_per_dim;

        int row_start = grid_row * k;
        int row_end = row_start + k;
        int col_start = grid_col * k;
        int col_end = col_start + k;

        if (row_end > n) row_end = n;
        if (col_end > n) col_end = n;

        compute_grid_hvx(A, B, C, n, row_start, row_end, col_start, col_end);

        grids_done++;
    }

    ctx->cycles = read_cycles() - start;
    ctx->grids_done = grids_done;

    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

static uint64_t hvx_mt_static_parallel(float* A, float* B, float* C, int n, int k, int num_threads) {
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    if (num_threads < 1) num_threads = 1;

    int grids_per_dim = (n + k - 1) / k;
    int total_grids = grids_per_dim * grids_per_dim;

    if (num_threads > total_grids) num_threads = total_grids;

    FARF(HIGH, "HVX + static multithreaded matmul: n=%d, k=%d, threads=%d", n, k, num_threads);

    MtStaticThreadContext contexts[MAX_THREADS];
    qurt_thread_t threads[MAX_THREADS];
    qurt_thread_attr_t attrs[MAX_THREADS];

    uint64_t total_start = read_cycles();

    for (int t = 0; t < num_threads; t++) {
        contexts[t].A = A;
        contexts[t].B = B;
        contexts[t].C = C;
        contexts[t].n = n;
        contexts[t].k = k;
        contexts[t].grids_per_dim = grids_per_dim;
        contexts[t].total_grids = total_grids;
        contexts[t].num_threads = num_threads;
        contexts[t].thread_id = t;
        contexts[t].cycles = 0;
        contexts[t].grids_done = 0;

        qurt_thread_attr_init(&attrs[t]);
        qurt_thread_attr_set_stack_addr(&attrs[t], (void*)thread_stacks[t]);
        qurt_thread_attr_set_stack_size(&attrs[t], STACK_SIZE);
        qurt_thread_attr_set_priority(&attrs[t], 100);

        int ret = qurt_thread_create(&threads[t], &attrs[t], mt_static_hvx_worker, &contexts[t]);
        if (ret != QURT_EOK) {
            FARF(HIGH, "error: thread creation failed (ret=%d)", ret);
            for (int i = 0; i < t; i++) {
                int status;
                qurt_thread_join(threads[i], &status);
            }
            return 0;
        }
    }

    for (int t = 0; t < num_threads; t++) {
        int status;
        qurt_thread_join(threads[t], &status);
    }

    return read_cycles() - total_start;
}

/* ------------------------------------------------------------------ */
/* FastRPC remote_handle64 entry points                               */
/* ------------------------------------------------------------------ */

int matmul_open(const char* uri, remote_handle64* handle) {
    void* tptr = malloc(1);
    *handle = (remote_handle64)tptr;
    return (*handle) ? 0 : AEE_ENOMEMORY;
}

int matmul_close(remote_handle64 handle) {
    if (handle)
        free((void*)handle);
    return 0;
}

int matmul_seq(remote_handle64 _h,
               const float* A, int ALen,
               const float* B, int BLen,
               float* C, int CLen,
               unsigned int n,
               uint64* cycles) {
    if (A == NULL || B == NULL || C == NULL || n == 0) return AEE_EBADPARM;
    if ((unsigned)ALen < n * n || (unsigned)BLen < n * n || (unsigned)CLen < n * n)
        return AEE_EBADSIZE;

    *cycles = seq_compute(A, B, C, (int)n);
    return AEE_SUCCESS;
}

int matmul_static_(remote_handle64 _h,
                    const float* A, int ALen,
                    const float* B, int BLen,
                    float* C, int CLen,
                    unsigned int n, unsigned int k, unsigned int t,
                    uint64* cycles) {
    if (A == NULL || B == NULL || C == NULL || n == 0 || k == 0 || t == 0)
        return AEE_EBADPARM;
    if ((unsigned)ALen < n * n || (unsigned)BLen < n * n || (unsigned)CLen < n * n)
        return AEE_EBADSIZE;

    *cycles = static_grid_parallel((float*)A, (float*)B, C, (int)n, (int)k, (int)t);
    return AEE_SUCCESS;
}

int matmul_dynamic(remote_handle64 _h,
                    const float* A, int ALen,
                    const float* B, int BLen,
                    float* C, int CLen,
                    unsigned int n, unsigned int k, unsigned int t,
                    uint64* cycles) {
    if (A == NULL || B == NULL || C == NULL || n == 0 || k == 0 || t == 0)
        return AEE_EBADPARM;
    if ((unsigned)ALen < n * n || (unsigned)BLen < n * n || (unsigned)CLen < n * n)
        return AEE_EBADSIZE;

    *cycles = dynamic_parallel((float*)A, (float*)B, C, (int)n, (int)k, (int)t);
    return AEE_SUCCESS;
}

int matmul_hvx(remote_handle64 _h,
               const float* A, int ALen,
               const float* B, int BLen,
               float* C, int CLen,
               unsigned int n,
               uint64* cycles) {
    if (A == NULL || B == NULL || C == NULL || n == 0) return AEE_EBADPARM;
    if ((unsigned)ALen < n * n || (unsigned)BLen < n * n || (unsigned)CLen < n * n)
        return AEE_EBADSIZE;

    *cycles = hvx_single_compute((float*)A, (float*)B, C, (int)n);
    return AEE_SUCCESS;
}

/*
 * A,B are `in`  sequences (const float*, len = n*n)
 * C   is  `rout` sequence (float*,       len = n*n)
 * cycles returns the DSP wall pcycles of the parallel matmul region.
 */
int matmul_mt(remote_handle64 _h,
              const float* A, int ALen,
              const float* B, int BLen,
              float* C, int CLen,
              unsigned int n, unsigned int k, unsigned int t,
              uint64* cycles) {
    if (A == NULL || B == NULL || C == NULL || n == 0 || k == 0 || t == 0)
        return AEE_EBADPARM;
    if ((unsigned)ALen < n * n || (unsigned)BLen < n * n || (unsigned)CLen < n * n)
        return AEE_EBADSIZE;

    *cycles = hvx_mt_parallel((float*)A, (float*)B, C, (int)n, (int)k, (int)t);
    return AEE_SUCCESS;
}

int matmul_hvx_mt_static(remote_handle64 _h,
                          const float* A, int ALen,
                          const float* B, int BLen,
                          float* C, int CLen,
                          unsigned int n, unsigned int k, unsigned int t,
                          uint64* cycles) {
    if (A == NULL || B == NULL || C == NULL || n == 0 || k == 0 || t == 0)
        return AEE_EBADPARM;
    if ((unsigned)ALen < n * n || (unsigned)BLen < n * n || (unsigned)CLen < n * n)
        return AEE_EBADSIZE;

    *cycles = hvx_mt_static_parallel((float*)A, (float*)B, C, (int)n, (int)k, (int)t);
    return AEE_SUCCESS;
}

int matmul_qhl_sdk(remote_handle64 _h,
                    const float* A, int ALen,
                    const float* B, int BLen,
                    float* C, int CLen,
                    unsigned int n,
                    uint64* cycles) {
    if (A == NULL || B == NULL || C == NULL || n == 0) return AEE_EBADPARM;
    if ((unsigned)ALen < n * n || (unsigned)BLen < n * n || (unsigned)CLen < n * n)
        return AEE_EBADSIZE;

    uint64_t start = read_cycles();
    int ret = qhblas_hvx_matrix_matrix_mpy_af((float*)A, (float*)B, C,
                                               (uint32_t)n, (uint32_t)n, (uint32_t)n);
    *cycles = read_cycles() - start;

    if (ret != 0) {
        FARF(HIGH, "error: qhblas_hvx_matrix_matrix_mpy_af returned %d", ret);
        return AEE_EFAILED;
    }
    return AEE_SUCCESS;
}
