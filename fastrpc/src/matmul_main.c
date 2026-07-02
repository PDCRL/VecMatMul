/*============================================================================
  matmul_main.c  —  CPU (HLOS) driver for the FastRPC `matmul` interface.

  Replaces main()/input.txt/output.txt from the standalone VecMatMul/*.c
  kernels:
    * allocates A,B,C as ION/rpcmem buffers (zero-copy to the DSP),
    * generates A,B with the same deterministic formula as the kernels,
    * requests an (un)signed PD, opens the skel on the CDSP, calls the
      selected kernel's RPC method (-m),
    * prints the DSP wall pcycles + checksum, and verifies against a CPU
      reference matmul.

  Usage: matmul [-n size] [-k grid] [-t threads] [-d domain] [-U 0|1] [-m variant]
     -n  matrix dimension N (N x N), default 64 (use a multiple of 32)
     -k  grid/block size, default 8 (ignored by seq/hvx/qhl_sdk)
     -t  DSP worker threads (1..10), default 4 (ignored by seq/hvx/qhl_sdk)
     -d  FastRPC domain id, default 3 (CDSP)
     -U  1 = unsigned PD (default), 0 = signed PD
     -m  kernel variant, default hvx_mt. One of:
           seq | static | dynamic | hvx | hvx_mt | hvx_mt_static | qhl_sdk
============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>

#include "rpcmem.h"
#include "remote.h"
#include "matmul.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif

/* Human-readable labels matching the simulator CSV 'method' column. */
static const char* variant_label(const char* m) {
    if (!strcmp(m, "seq")) return "Sequential";
    if (!strcmp(m, "static")) return "Static";
    if (!strcmp(m, "dynamic")) return "Dynamic";
    if (!strcmp(m, "hvx")) return "HVX";
    if (!strcmp(m, "hvx_mt")) return "HVX+MT";
    if (!strcmp(m, "hvx_mt_static")) return "HVX+MT+Static";
    if (!strcmp(m, "qhl_sdk")) return "QHL SDK";
    return "?";
}

static void gen_matrices(float* A, float* B, int n) {
    for (int i = 0; i < n * n; i++) {
        A[i] = (float)(i % 100) * 0.1f;
        B[i] = (float)((i * 7 + 13) % 100) * 0.1f;
    }
}

/* CPU reference matmul (row-major) for correctness cross-check. */
static void ref_matmul(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int p = 0; p < n; p++)
                s += A[i * n + p] * B[p * n + j];
            C[i * n + j] = s;
        }
}

static double checksum(const float* C, int n) {
    double s = 0.0;
    for (int i = 0; i < n * n; i++) s += C[i];
    return s;
}

int main(int argc, char** argv) {
    int n = 64, k = 8, t = 4, domain = CDSP_DOMAIN_ID, unsigned_pd = 1;
    char variant[32] = "hvx_mt";
    int opt, nErr = 0;
    remote_handle64 handle = -1;
    float *A = NULL, *B = NULL, *C = NULL, *ref = NULL;

    while ((opt = getopt(argc, argv, "n:k:t:d:U:m:h")) != -1) {
        switch (opt) {
            case 'n': n = atoi(optarg); break;
            case 'k': k = atoi(optarg); break;
            case 't': t = atoi(optarg); break;
            case 'd': domain = atoi(optarg); break;
            case 'U': unsigned_pd = atoi(optarg); break;
            case 'm': strncpy(variant, optarg, sizeof(variant) - 1); break;
            default:
                printf("usage: %s [-n size] [-k grid] [-t threads] [-d domain] [-U 0|1] [-m variant]\n", argv[0]);
                return 1;
        }
    }
    if (n <= 0 || k <= 0 || t <= 0) { printf("error: n,k,t must be > 0\n"); return 1; }
    if (strcmp(variant, "seq") && strcmp(variant, "static") && strcmp(variant, "dynamic") &&
        strcmp(variant, "hvx") && strcmp(variant, "hvx_mt") && strcmp(variant, "hvx_mt_static") &&
        strcmp(variant, "qhl_sdk")) {
        printf("error: unknown -m variant '%s'\n", variant);
        return 1;
    }

    size_t bytes = (size_t)n * n * sizeof(float);
    printf("matmul FastRPC: n=%d k=%d t=%d domain=%d pd=%s variant=%s (%s)\n",
           n, k, t, domain, unsigned_pd ? "unsigned" : "signed", variant, variant_label(variant));

    rpcmem_init();
    int heapid = RPCMEM_HEAP_ID_SYSTEM;
    A = (float*)rpcmem_alloc(heapid, RPCMEM_DEFAULT_FLAGS, bytes);
    B = (float*)rpcmem_alloc(heapid, RPCMEM_DEFAULT_FLAGS, bytes);
    C = (float*)rpcmem_alloc(heapid, RPCMEM_DEFAULT_FLAGS, bytes);
    if (!A || !B || !C) { nErr = -1; printf("ERROR: rpcmem_alloc failed\n"); goto bail; }

    gen_matrices(A, B, n);
    memset(C, 0, bytes);

    /* Request (un)signed PD before opening the handle. */
    if (remote_session_control) {
        struct remote_rpc_control_unsigned_module data;
        data.domain = domain;
        data.enable = unsigned_pd ? 1 : 0;
        if (AEE_SUCCESS != (nErr = remote_session_control(
                DSPRPC_CONTROL_UNSIGNED_MODULE, (void*)&data, sizeof(data)))) {
            printf("ERROR 0x%x: remote_session_control(UNSIGNED_MODULE) failed\n", nErr);
            goto bail;
        }
    }

    /* Open the skel on the requested domain. */
    char uri[256];
    const char* dom = (domain == 3) ? "cdsp" : (domain == 0) ? "adsp" : "cdsp";
    snprintf(uri, sizeof(uri), "%s&_dom=%s", matmul_URI, dom);
    if (AEE_SUCCESS != (nErr = matmul_open(uri, &handle))) {
        printf("ERROR 0x%x: matmul_open failed for %s\n", nErr, uri);
        goto bail;
    }

    /* Invoke the selected DSP matmul kernel. */
    uint64 cycles = 0;
    printf("Invoking matmul_%s on domain %d ...\n", variant, domain);
    double t0 = now_ms();
    if (!strcmp(variant, "seq")) {
        nErr = matmul_seq(handle, A, n * n, B, n * n, C, n * n, (unsigned)n, &cycles);
    } else if (!strcmp(variant, "static")) {
        nErr = matmul_static_(handle, A, n * n, B, n * n, C, n * n,
                               (unsigned)n, (unsigned)k, (unsigned)t, &cycles);
    } else if (!strcmp(variant, "dynamic")) {
        nErr = matmul_dynamic(handle, A, n * n, B, n * n, C, n * n,
                               (unsigned)n, (unsigned)k, (unsigned)t, &cycles);
    } else if (!strcmp(variant, "hvx")) {
        nErr = matmul_hvx(handle, A, n * n, B, n * n, C, n * n, (unsigned)n, &cycles);
    } else if (!strcmp(variant, "hvx_mt")) {
        nErr = matmul_mt(handle, A, n * n, B, n * n, C, n * n,
                          (unsigned)n, (unsigned)k, (unsigned)t, &cycles);
    } else if (!strcmp(variant, "hvx_mt_static")) {
        nErr = matmul_hvx_mt_static(handle, A, n * n, B, n * n, C, n * n,
                                     (unsigned)n, (unsigned)k, (unsigned)t, &cycles);
    } else { /* qhl_sdk */
        nErr = matmul_qhl_sdk(handle, A, n * n, B, n * n, C, n * n, (unsigned)n, &cycles);
    }
    double t1 = now_ms();
    if (AEE_SUCCESS != nErr) {
        printf("ERROR 0x%x: matmul_%s failed on domain %d\n", nErr, variant, domain);
        goto bail;
    }

    double cs = checksum(C, n);
    printf("\nDSP wall pcycles: %llu\n", (unsigned long long)cycles);
    printf("DSP call wall ms: %.3f\n", t1 - t0);
    printf("Checksum (DSP):   %.6f\n", cs);

    /* CPU reference cross-check. */
    ref = (float*)malloc(bytes);
    if (ref) {
        ref_matmul(A, B, ref, n);
        double cref = checksum(ref, n);
        printf("Checksum (CPU):   %.6f\n", cref);
        double diff = cs - cref; if (diff < 0) diff = -diff;
        double tol = 1e-3 * (cref < 0 ? -cref : cref) + 1e-3;
        printf("Result: %s\n", (diff <= tol) ? "PASS" : "FAIL");
    }

bail:
    if (handle != -1) matmul_close(handle);
    if (ref) free(ref);
    if (A) rpcmem_free(A);
    if (B) rpcmem_free(B);
    if (C) rpcmem_free(C);
    rpcmem_deinit();
    if (nErr) printf("\nmatmul FastRPC example FAILED (0x%x)\n", nErr);
    else      printf("\nmatmul FastRPC example done\n");
    return nErr;
}
