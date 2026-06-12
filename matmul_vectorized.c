

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <qurt.h>
#include <qurt_alloc.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

QURT_DECLARE_STATIC_HEAP(16777216); // 16 MB heap

unsigned int heapSize = 16777216;

// HVX 128B mode: 128 bytes = 32 single-precision floats per vector
#define HVX_FLOAT_LANES 32

static inline uint64_t read_cycles(void)
{
    uint64_t cycles;
    __asm__ __volatile__(
        "%0 = c15:14"
        : "=r"(cycles)
        :
        : "memory");
    return cycles;
}

// Vectorized matmul using HVX: C[i][j..j+31] += A[i][p] * B[p][j..j+31]
// Processes 32 output columns per iteration using HVX vector ops
uint64_t matmul_hvx(float *A, float *B, float *C, int n)
{
    int vec_cols = n / HVX_FLOAT_LANES; // number of full vector-width column blocks
    int tail = n % HVX_FLOAT_LANES;     // remaining columns

    printf("HVX vectorized matmul: n=%d\n", n);
    printf("Vector lanes: %d floats, Full vector blocks per row: %d, Tail: %d\n",
           HVX_FLOAT_LANES, vec_cols, tail);

    // Lock HVX context before using vector instructions
    int hvx_ret = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (hvx_ret != 0)
    {
        printf("error: HVX lock failed (ret=%d)\n", hvx_ret);
        return 0;
    }

    uint64_t start = read_cycles();

    for (int i = 0; i < n; i++)
    {
        // Process full vector-width blocks of columns
        for (int jb = 0; jb < vec_cols; jb++)
        {
            int j = jb * HVX_FLOAT_LANES;
            HVX_Vector vqsum = Q6_V_vzero(); // qf32 accumulator

            for (int p = 0; p < n; p++)
            {
                // Broadcast A[i][p] to all 32 lanes
                float a_val = A[i * n + p];
                HVX_Vector va = Q6_V_vsplat_R(*(uint32_t *)&a_val);

                // Load 32 contiguous floats from B[p][j..j+31]
                HVX_Vector vb = *(HVX_Vector *)&B[p * n + j];

                // Multiply in qfloat: sf * sf -> qf32
                HVX_Vector vqprod = Q6_Vqf32_vmpy_VsfVsf(va, vb);
                // Accumulate in qfloat: qf32 + qf32 -> qf32
                vqsum = Q6_Vqf32_vadd_Vqf32Vqf32(vqsum, vqprod);
            }

            // Convert qf32 back to sf and store
            *(HVX_Vector *)&C[i * n + j] = Q6_Vsf_equals_Vqf32(vqsum);
        }

        // Handle remaining columns with scalar code
        for (int j = vec_cols * HVX_FLOAT_LANES; j < n; j++)
        {
            float sum = 0.0f;
            for (int p = 0; p < n; p++)
            {
                sum += A[i * n + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }

    uint64_t elapsed = read_cycles() - start;

    qurt_hvx_unlock();
    return elapsed;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    FILE *fp = fopen("input.txt", "r");
    if (!fp)
    {
        printf("error: cannot open input.txt\n");
        return 1;
    }

    int n, k, num_threads;
    if (fscanf(fp, "%d %d %d", &n, &k, &num_threads) != 3)
    {
        printf("error: failed to read parameters (expected: n k t)\n");
        fclose(fp);
        return 1;
    }

    printf("Matrix size (n): %d\n", n);
    printf("HVX vectorized matmul (single-threaded)\n");

    fclose(fp);

    // Allocate with 128-byte alignment for HVX vector loads/stores
    float *A = (float *)memalign(128, (size_t)n * n * sizeof(float));
    float *B = (float *)memalign(128, (size_t)n * n * sizeof(float));
    float *C = (float *)memalign(128, (size_t)n * n * sizeof(float));

    if (!A || !B || !C)
    {
        printf("error: memory allocation failed\n");
        if (A)
            free(A);
        if (B)
            free(B);
        if (C)
            free(C);
        return 1;
    }

    // Generate matrices in-memory instead of reading from file
    for (int i = 0; i < n * n; i++)
    {
        A[i] = (float)(i % 100) * 0.1f;
        B[i] = (float)((i * 7 + 13) % 100) * 0.1f;
        //  A[i] = (float)(1<<20);
        // B[i] = (float)2.1;
    }

    memset(C, 0, (size_t)n * n * sizeof(float));

    printf("Starting HVX vectorized matmul...\n\n");
    uint64_t cycles = matmul_hvx(A, B, C, n);

    // Compute checksum to prevent compiler from eliminating the matmul
    double checksum = 0.0;
    for (int i = 0; i < n * n; i++)
    {
        checksum += C[i];
    }

    printf("\nTotal wall cycles: %llu\n", (unsigned long long)cycles);
    printf("Checksum: %.6f\n", checksum);

    FILE *out = fopen("output.txt", "w");
    if (out)
    {
        fprintf(out, "%d\n", n);
        // for (int i = 0; i < n; i++) {
        //     for (int j = 0; j < n; j++) {
        //         fprintf(out, "%.6f", C[i * n + j]);
        //         if (j < n - 1) fprintf(out, " ");
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
