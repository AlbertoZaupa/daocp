/*
 * Copyright (c) 2026 Alberto Zaupa
 * SPDX-License-Identifier: MIT
 * See LICENSE in the project root for license information.
 */

#include <internal.h>
#include <blasfeo.h>
#define DAOCP_GE_ZERO_TOL 1e-7

void daocp_trsv(f64* x, f64* L, u32 n, u32 stride) {
    for (u32 i=0; i<n; ++i) {
        x[i] /= L[i*stride + i];
        for (u32 j=i+1; j<n; ++j) x[j] -= L[j*stride + i] * x[i];
    }
}

void daocp_trsv_t(f64* x, f64* L, u32 n, u32 stride) {
    for (i32 i=n-1; i>=0; --i) {
        x[i] /= L[i*stride + i];
        for (i32 j=i-1; j>=0; --j) x[j] -= L[i*stride + j] * x[i];
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

void daocp_negate(f64* v, u32 n) {
    for (u32 i=0; i<n; ++i) v[i] *= -1.0;
}

f64 daocp_dot(f64* v, f64* w, u32 n) {
    f64 acc = 0.0;
    for (u32 i=0; i<n; ++i) acc += v[i] * w[i];
    return acc;
}