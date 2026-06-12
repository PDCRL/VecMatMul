

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <qurt.h>
#include <qurt_thread.h>
#include <qurt_atomic_ops.h>
#include <qurt_alloc.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

QURT_DECLARE_STATIC_HEAP(16777216);  // 16 MB heap

unsigned int heapSize = 16777216;

// HVX 128B mode: 128 bytes = 32 single-precision floats per vector
#define HVX_FLOAT_LANES 32

#define MAX_THREADS 8
#define STACK_SIZE (8 * 1024)

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


typedef struct {
    float* A;               // Input matrix A (n x n)
    float* B;               // Input matrix B (n x n)
    float* C;               // Output matrix C (n x n)
    int n;                  // Matrix dimension
    int k;                  // Grid size (k x k blocks)
    int grids_per_dim;      // Number of grids per dimension (n/k)
    int total_grids;        // Total number of grid blocks
    int num_threads;        // Total thread count (for static striping)
    int thread_id;          // This thread's ID
    uint64_t cycles;        // Per-thread cycle count
    int grids_done;         // Number of grids this thread processed
} ThreadContext;


// Compute one grid block [row_start..row_end) x [col_start..col_end) using HVX
static inline void compute_grid_hvx(float* A, float* B, float* C,
                                     int n, int row_start, int row_end,
                                     int col_start, int col_end) {
    int block_width = col_end - col_start;
    int vec_cols = block_width / HVX_FLOAT_LANES;
    int tail_start = col_start + vec_cols * HVX_FLOAT_LANES;

    for (int i = row_start; i < row_end; i++) {
        // Vectorized: process 32 columns at a time within this grid block
        for (int jb = 0; jb < vec_cols; jb++) {
            int j = col_start + jb * HVX_FLOAT_LANES;
            HVX_Vector vqsum = Q6_V_vzero();  // qf32 accumulator

            for (int p = 0; p < n; p++) {
                float a_val = A[i * n + p];
                HVX_Vector va = Q6_V_vsplat_R(*(uint32_t*)&a_val);
                HVX_Vector vb = *(HVX_Vector*)&B[p * n + j];
                // Multiply in qfloat: sf * sf -> qf32
                HVX_Vector vqprod = Q6_Vqf32_vmpy_VsfVsf(va, vb);
                // Accumulate in qfloat: qf32 + qf32 -> qf32
                vqsum = Q6_Vqf32_vadd_Vqf32Vqf32(vqsum, vqprod);
            }

            // Convert qf32 back to sf and store
            *(HVX_Vector*)&C[i * n + j] = Q6_Vsf_equals_Vqf32(vqsum);
        }

        // Scalar tail: remaining columns that don't fill a full vector
        for (int j = tail_start; j < col_end; j++) {
            float sum = 0.0f;
            for (int p = 0; p < n; p++) {
                sum += A[i * n + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}


// Scalar fallback worker — used when HVX lock fails
void matmul_scalar_mt_static_worker(void* arg) {
    ThreadContext* ctx = (ThreadContext*)arg;

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

    // Static assignment: thread owns all grid_num where grid_num % num_threads == thread_id
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


void matmul_hvx_mt_static_worker(void* arg) {
    ThreadContext* ctx = (ThreadContext*)arg;

    // Try to lock an HVX context for this thread
    int hvx_ret = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (hvx_ret != 0) {
        printf("Thread %d: HVX lock failed (ret=%d), falling back to scalar\n",
               ctx->thread_id, hvx_ret);
        matmul_scalar_mt_static_worker(arg);
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

    // Static assignment: thread owns all grid_num where grid_num % num_threads == thread_id
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


uint64_t matmul_hvx_mt_static(float* A, float* B, float* C, int n, int k, int num_threads) {
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    if (num_threads < 1) num_threads = 1;

    int grids_per_dim = (n + k - 1) / k;
    int total_grids = grids_per_dim * grids_per_dim;

    if (num_threads > total_grids) num_threads = total_grids;

    printf("HVX + static multithreaded matmul: n=%d, k=%d, threads=%d\n", n, k, num_threads);
    printf("Grids per dimension: %d, Total grids: %d\n", grids_per_dim, total_grids);
    printf("Task allocation: static (grid_id %% num_threads == thread_id)\n");
    printf("Vectorization: HVX 128B (%d floats/vector)\n", HVX_FLOAT_LANES);

    ThreadContext contexts[MAX_THREADS];
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

        int ret = qurt_thread_create(&threads[t], &attrs[t], matmul_hvx_mt_static_worker, &contexts[t]);
        if (ret != QURT_EOK) {
            printf("error: thread creation failed (ret=%d)\n", ret);
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
        printf("Thread %d completed: %llu cycles, %d grids processed\n",
            t, (unsigned long long)contexts[t].cycles, contexts[t].grids_done);
    }

    return read_cycles() - total_start;
}


int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    FILE* fp = fopen("input.txt", "r");
    if (!fp) {
        printf("error: cannot open input.txt\n");
        return 1;
    }

    int n, k, num_threads;
    if (fscanf(fp, "%d %d %d", &n, &k, &num_threads) != 3) {
        printf("error: failed to read parameters (expected: n k t)\n");
        fclose(fp);
        return 1;
    }

    printf("Matrix size (n): %d\n", n);
    printf("Grid size (k):   %d\n", k);
    printf("Num threads (t): %d\n", num_threads);

    fclose(fp);

    // Allocate with 128-byte alignment for HVX vector loads/stores
    float* A = (float*)memalign(128, (size_t)n * n * sizeof(float));
    float* B = (float*)memalign(128, (size_t)n * n * sizeof(float));
    float* C = (float*)memalign(128, (size_t)n * n * sizeof(float));

    if (!A || !B || !C) {
        printf("error: memory allocation failed\n");
        if (A) free(A);
        if (B) free(B);
        if (C) free(C);
        return 1;
    }

    // Generate matrices in-memory instead of reading from file
    for (int i = 0; i < n * n; i++) {
        A[i] = (float)(i % 100) * 0.1f;
        B[i] = (float)((i * 7 + 13) % 100) * 0.1f;
    }

    memset(C, 0, (size_t)n * n * sizeof(float));

    printf("Starting HVX + static multithreaded matmul...\n\n");
    uint64_t cycles = matmul_hvx_mt_static(A, B, C, n, k, num_threads);

    // Compute checksum to prevent compiler from eliminating the matmul
    double checksum = 0.0;
    for (int i = 0; i < n * n; i++) {
        checksum += C[i];
    }

    printf("\nTotal wall cycles: %llu\n", (unsigned long long)cycles);
    printf("Checksum: %.6f\n", checksum);

    FILE* out = fopen("output.txt", "w");
    if (out) {
        fprintf(out, "%d\n", n);
        fclose(out);
        printf("Result written to output.txt\n");
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
