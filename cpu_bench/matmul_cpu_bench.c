/*============================================================================
  matmul_cpu_bench.c — pure ARM CPU matmul benchmark, no DSP/FastRPC at all.

  Baseline for "is offloading to the Hexagon CDSP worth it": same matrix
  generation formula and same three access patterns (Sequential / Static /
  Dynamic) as the DSP scalar kernels in VecMatMul/matmul_{sequential,
  experiment,dynamic}.c, but running natively on the RUBIK Pi 3's ARM cores
  (4x Cortex-A55 + 4x Cortex-A78) via pthreads. Timed with wall-clock
  (CLOCK_MONOTONIC), matching the "-DSP call wall ms-" timing added to
  fastrpc/src/matmul_main.c, so the two sides are directly comparable.

  Usage: matmul_cpu_bench -n size [-t threads] -m seq|static|dynamic
============================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static void gen_matrices(float* A, float* B, int n) {
    for (int i = 0; i < n * n; i++) {
        A[i] = (float)(i % 100) * 0.1f;
        B[i] = (float)((i * 7 + 13) % 100) * 0.1f;
    }
}

static double checksum(const float* C, int n) {
    double s = 0.0;
    for (int i = 0; i < n * n; i++) s += C[i];
    return s;
}

static void ref_matmul(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int p = 0; p < n; p++)
                s += A[i * n + p] * B[p * n + j];
            C[i * n + j] = s;
        }
}

typedef struct {
    const float* A;
    const float* B;
    float* C;
    int n, tid, num_threads;
    _Atomic int* counter; /* used by dynamic only */
} Ctx;

/* Static: thread tid owns rows [tid*n/T, (tid+1)*n/T). */
static void* worker_static(void* arg) {
    Ctx* c = (Ctx*)arg;
    int n = c->n;
    int row_start = (int)((int64_t)n * c->tid / c->num_threads);
    int row_end = (int)((int64_t)n * (c->tid + 1) / c->num_threads);
    for (int i = row_start; i < row_end; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int p = 0; p < n; p++)
                s += c->A[i * n + p] * c->B[p * n + j];
            c->C[i * n + j] = s;
        }
    return NULL;
}

/* Dynamic: threads pull rows from a shared atomic counter. */
static void* worker_dynamic(void* arg) {
    Ctx* c = (Ctx*)arg;
    int n = c->n;
    int i;
    while ((i = atomic_fetch_add(c->counter, 1)) < n) {
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int p = 0; p < n; p++)
                s += c->A[i * n + p] * c->B[p * n + j];
            c->C[i * n + j] = s;
        }
    }
    return NULL;
}

static double run_parallel(const float* A, const float* B, float* C, int n,
                            int num_threads, int dynamic) {
    pthread_t threads[64];
    Ctx ctxs[64];
    _Atomic int counter = 0;
    if (num_threads > 64) num_threads = 64;

    double t0 = now_ms();
    for (int t = 0; t < num_threads; t++) {
        ctxs[t] = (Ctx){ .A = A, .B = B, .C = C, .n = n, .tid = t,
                          .num_threads = num_threads, .counter = &counter };
        pthread_create(&threads[t], NULL, dynamic ? worker_dynamic : worker_static, &ctxs[t]);
    }
    for (int t = 0; t < num_threads; t++) pthread_join(threads[t], NULL);
    return now_ms() - t0;
}

static double run_sequential(const float* A, const float* B, float* C, int n) {
    double t0 = now_ms();
    ref_matmul(A, B, C, n);
    return now_ms() - t0;
}

int main(int argc, char** argv) {
    int n = 64, t = 4;
    char variant[16] = "seq";
    int opt;

    while ((opt = getopt(argc, argv, "n:t:m:h")) != -1) {
        switch (opt) {
            case 'n': n = atoi(optarg); break;
            case 't': t = atoi(optarg); break;
            case 'm': strncpy(variant, optarg, sizeof(variant) - 1); break;
            default:
                printf("usage: %s -n size [-t threads] -m seq|static|dynamic\n", argv[0]);
                return 1;
        }
    }
    if (n <= 0 || t <= 0) { printf("error: n,t must be > 0\n"); return 1; }
    if (strcmp(variant, "seq") && strcmp(variant, "static") && strcmp(variant, "dynamic")) {
        printf("error: unknown -m variant '%s'\n", variant);
        return 1;
    }

    printf("matmul CPU bench (no DSP): n=%d t=%d variant=%s\n", n, t, variant);

    size_t bytes = (size_t)n * n * sizeof(float);
    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);
    if (!A || !B || !C) { printf("error: malloc failed\n"); return 1; }

    gen_matrices(A, B, n);
    memset(C, 0, bytes);

    double ms;
    if (!strcmp(variant, "seq")) {
        ms = run_sequential(A, B, C, n);
    } else if (!strcmp(variant, "static")) {
        ms = run_parallel(A, B, C, n, t, 0);
    } else {
        ms = run_parallel(A, B, C, n, t, 1);
    }

    double cs = checksum(C, n);
    printf("\nCPU wall ms:    %.3f\n", ms);
    printf("Checksum (CPU): %.6f\n", cs);

    /* seq *is* ref_matmul -- skip recomputing it (doubles cost at large n). */
    if (!strcmp(variant, "seq")) {
        printf("Result: PASS\n");
    } else {
        float* ref = (float*)malloc(bytes);
        if (ref) {
            ref_matmul(A, B, ref, n);
            double cref = checksum(ref, n);
            double diff = cs - cref; if (diff < 0) diff = -diff;
            double tol = 1e-3 * (cref < 0 ? -cref : cref) + 1e-3;
            printf("Result: %s\n", (diff <= tol) ? "PASS" : "FAIL");
            free(ref);
        }
    }

    free(A); free(B); free(C);
    return 0;
}
