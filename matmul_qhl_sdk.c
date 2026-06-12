
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <qurt.h>
#include <qurt_alloc.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "qhblas_hvx.h"

QURT_DECLARE_STATIC_HEAP(16777216);
unsigned int heapSize = 16777216;

static inline uint64_t read_cycles(void) {
    uint64_t cycles;
    __asm__ __volatile__("%0 = c15:14" : "=r"(cycles) :: "memory");
    return cycles;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

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
    fclose(fp);

    printf("Matrix size (n): %d\n", n);
    printf("Grid size (k):   %d  (unused — SDK GEMM is row-based)\n", k);
    printf("Num threads (t): %d  (unused — SDK GEMM is single-threaded)\n", num_threads);

    /* Allocate 128-byte aligned matrices for HVX */
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

    for (int i = 0; i < n * n; i++) {
        A[i] = (float)(i % 100) * 0.1f;
        B[i] = (float)((i * 7 + 13) % 100) * 0.1f;
    }
    memset(C, 0, (size_t)n * n * sizeof(float));

    printf("Starting SDK qhblas_hvx_matrix_matrix_mpy_af (single-threaded HVX GEMM)...\n\n");

    uint64_t start = read_cycles();
    int ret = qhblas_hvx_matrix_matrix_mpy_af(A, B, C, (uint32_t)n, (uint32_t)n, (uint32_t)n);
    uint64_t cycles = read_cycles() - start;

    if (ret != 0) {
        printf("error: qhblas_hvx_matrix_matrix_mpy_af returned %d\n", ret);
    }

    double checksum = 0.0;
    for (int i = 0; i < n * n; i++) checksum += C[i];

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
