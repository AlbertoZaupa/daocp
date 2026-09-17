/*
 * Copyright (c) 2026 Alberto Zaupa
 * SPDX-License-Identifier: MIT
 * See LICENSE in the project root for license information.
 */

#include <internal.h>
#include <blasfeo.h>
#define DAOCP_GE_ZERO_TOL 1e-7

void daocp_trsv(f64* x, f64* L, u32 n, u32 stride) {
    if (n==0) return;

    x[0] *= L[0];
    for (u32 i=1; i<n; ++i) {
        x[i] -= daocp_dot(L + i*stride, x, i);
        x[i] *= L[i*stride + i];
    }
}

void daocp_trsv_t(f64* x, f64* L, u32 n, u32 stride) {
    for (i32 i=n-1; i>=0; --i) {
        x[i] *= L[i*stride + i];
        daocp_daxpy(L + i*stride, x, -x[i], i);
    }
} 

void daocp_fma_mv(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride) {
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1; v1.pa = x;
    for (u32 i=0; i<ny; ++i) {
        v0.pa = A + i*stride;
        y[i] += blasfeo_ddot(nx, &v0, 0, &v1, 0);
    }
}

void daocp_fms_mv(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride) {
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1; v1.pa = x;
    for (u32 i=0; i<ny; ++i) {
        v0.pa = A + i*stride;
        y[i] -= blasfeo_ddot(nx, &v0, 0, &v1, 0);
    }
}

void daocp_fma_mm_nt(f64* C, f64* A, f64* B, u32 nr, u32 nc, u32 k, u32 ostride) {
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1;
    for (u32 i=0; i<nr; ++i)
        for (u32 j=0; j<nc; ++j) {
            v0.pa = A + i*k; v1.pa = B + j*k;
            C[i*ostride + j] += blasfeo_ddot(k, &v0, 0, &v1, 0);
        }
}

void daocp_fms_mm_nt(f64* C, f64* A, f64* B, u32 nr, u32 nc, u32 k, u32 ostride) {
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1;
    for (u32 i=0; i<nr; ++i)
        for (u32 j=0; j<nc; ++j) {
            v0.pa = A + i*k; v1.pa = B + j*k;
            C[i*ostride + j] -= blasfeo_ddot(k, &v0, 0, &v1, 0);
        }
}

u32 daocp_gaussian_elimination(f64* A, f64* tmp, u32 nr, u32 nc, u32 nctot, u32 R) {
    u32 rho = 0;
    for (u32 i=0; i<nc; ++i) {
        if (rho == R) break;
        // Find pivot
        u32 pi = rho-1;
        f64 p = DAOCP_GE_ZERO_TOL;
        for (u32 j=rho; j<nr; ++j)
            if (DAOCP_ABS(A[j*nctot+i]) > DAOCP_ABS(p)) {
                pi = j;
                p = A[j*nctot+i];
            }
        if (pi==rho-1) continue;

        // Swap rows pi and rho
        if (rho != pi) {
            memcpy(tmp, A+rho*nctot, nctot*sizeof(f64));
            memcpy(A+rho*nctot, A+pi*nctot, nctot*sizeof(f64));
            memcpy(A+pi*nctot, tmp, nctot*sizeof(f64));
        }

        // Perform elimination step
        for (u32 j=rho+1; j<nr; ++j) {
            f64 alpha = A[j*nctot + i] / p;
            for (u32 k=i; k<nctot; ++k) A[j*nctot + k] -= alpha * A[rho*nctot + k];
        }
        
        // Increase rank
        rho += 1;
    }
    return rho;
}

void daocp_daxpy(const f64* __restrict__ x, f64* __restrict__ y, f64 a, u32 n) {
    if (n==0) return;

    // Vectorization width = 4. This allows the cpu to use maximum-width simd,
    // thereby reducing load pressure => higher bandwidth.
    // Increasing width to 8 may reduce frontend pressure, but less clearly advantageous.
    if (n>=4) {
        for (i32 i=0; i<=n-4; i+=4) {
            y[0] += x[0] * a;
            y[1] += x[1] * a;
            y[2] += x[2] * a;
            y[3] += x[3] * a;
            x += 4; y += 4;
        }
    }
    if (n - ((n>>2)<<2) >= 2) {
        y[0] += x[0] * a;
        y[1] += x[1] * a;
        x += 2; y += 2;
    }
    if (n - ((n>>1)<<1) >= 1) y[0] += x[0] * a;
}

void daocp_negate(f64* v, u32 n) {
    if (n==0) return;

    // Vectorization width = 4. This allows the cpu to use maximum-width simd,
    // thereby reducing load pressure => higher bandwidth.
    // Increasing width to 8 may reduce frontend pressure, but less clearly advantageous.
    if (n >= 4) {
        for (i32 i=0; i<=n-4; i+=4) {
            v[0] *= -1.0;
            v[1] *= -1.0;
            v[2] *= -1.0;
            v[3] *= -1.0;
            v += 4;
        }
    }
    if (n - ((n>>2)<<2) >= 2) {
        v[0] *= -1.0;
        v[1] *= -1.0;
        v += 2;
    }
    if (n - ((n>>1)<<1) >= 1) {
        v[0] *= -1.0;
    }
}

f64 daocp_dot(const f64* __restrict__ v, const f64* __restrict__ w, u32 n) {
    if (n==0) return 0.0;

    // 8 accumulators to reduce length of dependency chain from n to n/8 + 3.
    f64 acc0 = 0;
    f64 acc1 = 0;
    f64 acc2 = 0;
    f64 acc3 = 0;
    f64 acc4 = 0;
    f64 acc5 = 0;
    f64 acc6 = 0;
    f64 acc7 = 0;

    if (n>=8) {
        for (i32 i=0; i<=n-8; i+=8) {
            acc0 += v[0] * w[0];
            acc1 += v[1] * w[1];
            acc2 += v[2] * w[2];
            acc3 += v[3] * w[3];
            acc4 += v[4] * w[4];
            acc5 += v[5] * w[5];
            acc6 += v[6] * w[6];
            acc7 += v[7] * w[7];
            v += 8; w += 8;
        }
    }
    if (n - ((n>>3)<<3) >= 4) {
        acc0 += v[0] * w[0];
        acc1 += v[1] * w[1];
        acc3 += v[2] * w[2];
        acc4 += v[3] * w[3];
        v += 4; w += 4;
    }
    if (n - ((n>>2)<<2) >= 2) {
        acc0 += v[0]*w[0];
        acc1 += v[1]*w[1];
        v += 2; w += 2;
    }
    if (n - ((n>>1)<<1)) acc0 += v[0]*w[0]; 

    acc0 += acc2;
    acc1 += acc3;
    acc4 += acc6;
    acc5 += acc7;
    acc0 += acc1;
    acc4 += acc5;
    return acc0 + acc4;
}