

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <qurt.h>
#include <qurt_alloc.h>

QURT_DECLARE_STATIC_HEAP(16777216);  // 16 MB heap

unsigned int heapSize = 16777216;  // override weak heapSize symbol

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
    printf("Sequential matmul (single-threaded, no grids)\n");

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

    printf("Starting sequential matmul...\n\n");

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

    uint64_t cycles = read_cycles() - start;

    // Compute checksum to prevent compiler from eliminating the matmul
    double checksum = 0.0;
    for (int i = 0; i < n * n; i++) {
        checksum += C[i];
    }

    printf("Sequential matmul completed: %llu cycles\n", (unsigned long long)cycles);
    printf("Total wall cycles: %llu\n", (unsigned long long)cycles);
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
