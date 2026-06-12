


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <qurt.h>
#include <qurt_thread.h>
#include <qurt_atomic_ops.h>
#include <qurt_alloc.h>

QURT_DECLARE_STATIC_HEAP(16777216);  // 16 MB heap

unsigned int heapSize = 16777216;

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
    unsigned int* counter;  // Shared atomic counter for dynamic dispatch
    int thread_id;          // This thread's ID (for logging)
    uint64_t cycles;        // Per-thread cycle count
    int grids_done;         // Number of grids this thread processed
} ThreadContext;



void matmul_dynamic_worker(void* arg) {
    ThreadContext* ctx = (ThreadContext*)arg;

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



uint64_t matmul_dynamic_parallel(float* A, float* B, float* C, int n, int k, int num_threads) {
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    if (num_threads < 1) num_threads = 1;

    int grids_per_dim = (n + k - 1) / k;
    int total_grids = grids_per_dim * grids_per_dim;

    if (num_threads > total_grids) num_threads = total_grids;

    printf("Dynamic grid-based matmul: n=%d, k=%d, threads=%d\n", n, k, num_threads);
    printf("Grids per dimension: %d, Total grids: %d\n", grids_per_dim, total_grids);
    printf("Task allocation: dynamic (atomic counter)\n");

    // Shared atomic counter — all threads grab work from this
    unsigned int shared_counter = 0;

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
        contexts[t].counter = &shared_counter;
        contexts[t].thread_id = t;
        contexts[t].cycles = 0;
        contexts[t].grids_done = 0;

        qurt_thread_attr_init(&attrs[t]);
        qurt_thread_attr_set_stack_addr(&attrs[t], (void*)thread_stacks[t]);
        qurt_thread_attr_set_stack_size(&attrs[t], STACK_SIZE);
        qurt_thread_attr_set_priority(&attrs[t], 100);

        int ret = qurt_thread_create(&threads[t], &attrs[t], matmul_dynamic_worker, &contexts[t]);
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

    float* A = (float*)malloc((size_t)n * n * sizeof(float));
    float* B = (float*)malloc((size_t)n * n * sizeof(float));
    float* C = (float*)malloc((size_t)n * n * sizeof(float));

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

    printf("Starting dynamic parallel matmul...\n\n");
    uint64_t cycles = matmul_dynamic_parallel(A, B, C, n, k, num_threads);

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
        // for (int i = 0; i < n; i++) {
        //     for (int j = 0; j < n; j++) {
        //         fprintf(out, "%.6f ", C[i * n + j]);
        //     }
        //     fprintf(out, "\n");
        // }
        fclose(out);
        printf("Result written to output.txt\n");
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
