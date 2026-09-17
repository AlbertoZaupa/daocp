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
    if (ny == 0 || nx == 0) return;

    // We maintain a grid of 16 (scalar) accumulators. This gives a decent amount
    // of parallelism.
    f64 x0, x1, x2, x3;

    if (ny >= 4) {
        for (u32 i=0; i<=ny-4; i+=4) {
            f64 a00 = 0; f64 a01 = 0; f64 a02 = 0; f64 a03 = 0;
            f64 a10 = 0; f64 a11 = 0; f64 a12 = 0; f64 a13 = 0;
            f64 a20 = 0; f64 a21 = 0; f64 a22 = 0; f64 a23 = 0;
            f64 a30 = 0; f64 a31 = 0; f64 a32 = 0; f64 a33 = 0;
            // Accumulate dot products
            if (nx >= 4) {
                for (u32 j=0; j<=nx-4; j+=4) {
                    x0 = x[j]; x1 = x[j+1]; x2 = x[j+2]; x3 = x[j+3];
                    a00 += A[j]*x0; a01 += A[j+1]*x1; a02 += A[j+2]*x2; a03 += A[j+3]*x3;
                    a10 += A[stride + j]*x0; a11 += A[stride+j+1]*x1;
                    a12 += A[stride+j+2]*x2; a13 += A[stride+j+3]*x3;
                    a20 += A[2*stride + j]*x0; a21 += A[2*stride+j+1]*x1;
                    a22 += A[2*stride+j+2]*x2; a23 += A[2*stride+j+3]*x3;
                    a30 += A[3*stride + j]*x0; a31 += A[3*stride+j+1]*x1;
                    a32 += A[3*stride+j+2]*x2; a33 += A[3*stride+j+3]*x3;
                }
            }
            u32 nx_mod = (nx>>2)<<2;
            if (nx - nx_mod >= 2) {
                x0 = x[nx_mod]; x1 = x[nx_mod+1];
                a00 += A[nx_mod]*x0; a01 += A[nx_mod+1]*x1;
                a10 += A[stride + nx_mod]*x0; a11 += A[stride+nx_mod+1]*x1;
                a20 += A[2*stride + nx_mod]*x0; a21 += A[2*stride+nx_mod+1]*x1;
                a30 += A[3*stride + nx_mod]*x0; a31 += A[3*stride+nx_mod+1]*x1;
            }
            nx_mod = (nx>>1)<<1;
            if (nx > nx_mod) {
                x0 = x[nx_mod];
                a00 += A[nx_mod]*x0;
                a10 += A[stride + nx_mod]*x0;
                a20 += A[2*stride + nx_mod]*x0;
                a30 += A[3*stride + nx_mod]*x0;
            }

            // Store
            a00 += a02; a01 += a03;
            a10 += a12; a11 += a13;
            a20 += a22; a21 += a23;
            a30 += a32; a31 += a33;
            y[0] += a00 + a01;
            y[1] += a10 + a11;
            y[2] += a20 + a21;
            y[3] += a30 + a31;
            A += 4*stride;
            y += 4;
        }
    }

    // Handle two of the remaining rows, then the final odd row.
    if (ny - ((ny>>2)<<2) >= 2) {
        f64 a00 = 0; f64 a01 = 0; f64 a02 = 0; f64 a03 = 0;
        f64 a10 = 0; f64 a11 = 0; f64 a12 = 0; f64 a13 = 0;
        if (nx >= 4) {
            for (u32 j=0; j<=nx-4; j+=4) {
                x0 = x[j]; x1 = x[j+1]; x2 = x[j+2]; x3 = x[j+3];
                a00 += A[j]*x0; a01 += A[j+1]*x1;
                a02 += A[j+2]*x2; a03 += A[j+3]*x3;
                a10 += A[stride+j]*x0; a11 += A[stride+j+1]*x1;
                a12 += A[stride+j+2]*x2; a13 += A[stride+j+3]*x3;
            }
        }
        u32 nx_mod = (nx>>2)<<2;
        if (nx - nx_mod >= 2) {
            x0 = x[nx_mod]; x1 = x[nx_mod+1];
            a00 += A[nx_mod]*x0; a01 += A[nx_mod+1]*x1;
            a10 += A[stride+nx_mod]*x0; a11 += A[stride+nx_mod+1]*x1;
        }
        nx_mod = (nx>>1)<<1;
        if (nx > nx_mod) {
            x0 = x[nx_mod];
            a00 += A[nx_mod]*x0;
            a10 += A[stride+nx_mod]*x0;
        }
        a00 += a02; a01 += a03;
        a10 += a12; a11 += a13;
        y[0] += a00 + a01;
        y[1] += a10 + a11;
        A += 2*stride;
        y += 2;
    }
    if (ny - ((ny>>1)<<1) >= 1) {
        f64 a00 = 0; f64 a01 = 0; f64 a02 = 0; f64 a03 = 0;
        if (nx >= 4) {
            for (u32 j=0; j<=nx-4; j+=4) {
                a00 += A[j]*x[j]; a01 += A[j+1]*x[j+1];
                a02 += A[j+2]*x[j+2]; a03 += A[j+3]*x[j+3];
            }
        }
        u32 nx_mod = (nx>>2)<<2;
        if (nx - nx_mod >= 2) {
            a00 += A[nx_mod]*x[nx_mod];
            a01 += A[nx_mod+1]*x[nx_mod+1];
        }
        nx_mod = (nx>>1)<<1;
        if (nx > nx_mod) a00 += A[nx_mod]*x[nx_mod];
        a00 += a02; a01 += a03;
        y[0] += a00 + a01;
    }
}

void daocp_fms_mv(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride) {
    if (ny == 0 || nx == 0) return;

    // We maintain a grid of 16 (scalar) accumulators. This gives a decent amount
    // of parallelism.
    f64 x0, x1, x2, x3;

    if (ny >= 4) {
        for (u32 i=0; i<=ny-4; i+=4) {
            f64 a00 = 0; f64 a01 = 0; f64 a02 = 0; f64 a03 = 0;
            f64 a10 = 0; f64 a11 = 0; f64 a12 = 0; f64 a13 = 0;
            f64 a20 = 0; f64 a21 = 0; f64 a22 = 0; f64 a23 = 0;
            f64 a30 = 0; f64 a31 = 0; f64 a32 = 0; f64 a33 = 0;
            // Accumulate dot products
            if (nx >= 4) {
                for (u32 j=0; j<=nx-4; j+=4) {
                    x0 = x[j]; x1 = x[j+1]; x2 = x[j+2]; x3 = x[j+3];
                    a00 += A[j]*x0; a01 += A[j+1]*x1; a02 += A[j+2]*x2; a03 += A[j+3]*x3;
                    a10 += A[stride + j]*x0; a11 += A[stride+j+1]*x1;
                    a12 += A[stride+j+2]*x2; a13 += A[stride+j+3]*x3;
                    a20 += A[2*stride + j]*x0; a21 += A[2*stride+j+1]*x1;
                    a22 += A[2*stride+j+2]*x2; a23 += A[2*stride+j+3]*x3;
                    a30 += A[3*stride + j]*x0; a31 += A[3*stride+j+1]*x1;
                    a32 += A[3*stride+j+2]*x2; a33 += A[3*stride+j+3]*x3;
                }
            }
            u32 nx_mod = (nx>>2)<<2;
            if (nx - nx_mod >= 2) {
                x0 = x[nx_mod]; x1 = x[nx_mod+1];
                a00 += A[nx_mod]*x0; a01 += A[nx_mod+1]*x1;
                a10 += A[stride + nx_mod]*x0; a11 += A[stride+nx_mod+1]*x1;
                a20 += A[2*stride + nx_mod]*x0; a21 += A[2*stride+nx_mod+1]*x1;
                a30 += A[3*stride + nx_mod]*x0; a31 += A[3*stride+nx_mod+1]*x1;
            }
            nx_mod = (nx>>1)<<1;
            if (nx > nx_mod) {
                x0 = x[nx_mod];
                a00 += A[nx_mod]*x0;
                a10 += A[stride + nx_mod]*x0;
                a20 += A[2*stride + nx_mod]*x0;
                a30 += A[3*stride + nx_mod]*x0;
            }

            // Store
            a00 += a02; a01 += a03;
            a10 += a12; a11 += a13;
            a20 += a22; a21 += a23;
            a30 += a32; a31 += a33;
            y[0] -= a00 + a01;
            y[1] -= a10 + a11;
            y[2] -= a20 + a21;
            y[3] -= a30 + a31;
            A += 4*stride;
            y += 4;
        }
    }

    // Handle two of the remaining rows, then the final odd row.
    if (ny - ((ny>>2)<<2) >= 2) {
        f64 a00 = 0; f64 a01 = 0; f64 a02 = 0; f64 a03 = 0;
        f64 a10 = 0; f64 a11 = 0; f64 a12 = 0; f64 a13 = 0;
        if (nx >= 4) {
            for (u32 j=0; j<=nx-4; j+=4) {
                x0 = x[j]; x1 = x[j+1]; x2 = x[j+2]; x3 = x[j+3];
                a00 += A[j]*x0; a01 += A[j+1]*x1;
                a02 += A[j+2]*x2; a03 += A[j+3]*x3;
                a10 += A[stride+j]*x0; a11 += A[stride+j+1]*x1;
                a12 += A[stride+j+2]*x2; a13 += A[stride+j+3]*x3;
            }
        }
        u32 nx_mod = (nx>>2)<<2;
        if (nx - nx_mod >= 2) {
            x0 = x[nx_mod]; x1 = x[nx_mod+1];
            a00 += A[nx_mod]*x0; a01 += A[nx_mod+1]*x1;
            a10 += A[stride+nx_mod]*x0; a11 += A[stride+nx_mod+1]*x1;
        }
        nx_mod = (nx>>1)<<1;
        if (nx > nx_mod) {
            x0 = x[nx_mod];
            a00 += A[nx_mod]*x0;
            a10 += A[stride+nx_mod]*x0;
        }
        a00 += a02; a01 += a03;
        a10 += a12; a11 += a13;
        y[0] -= a00 + a01;
        y[1] -= a10 + a11;
        A += 2*stride;
        y += 2;
    }
    if (ny - ((ny>>1)<<1) >= 1) {
        f64 a00 = 0; f64 a01 = 0; f64 a02 = 0; f64 a03 = 0;
        if (nx >= 4) {
            for (u32 j=0; j<=nx-4; j+=4) {
                a00 += A[j]*x[j]; a01 += A[j+1]*x[j+1];
                a02 += A[j+2]*x[j+2]; a03 += A[j+3]*x[j+3];
            }
        }
        u32 nx_mod = (nx>>2)<<2;
        if (nx - nx_mod >= 2) {
            a00 += A[nx_mod]*x[nx_mod];
            a01 += A[nx_mod+1]*x[nx_mod+1];
        }
        nx_mod = (nx>>1)<<1;
        if (nx > nx_mod) a00 += A[nx_mod]*x[nx_mod];
        a00 += a02; a01 += a03;
        y[0] -= a00 + a01;
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
