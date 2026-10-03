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

void daocp_fma_mv(
    f64* __restrict__ y, const f64* __restrict__ A, 
    const f64* __restrict__ x, f64 alpha, u32 ny, u32 nx, u32 stride) {
    if (ny == 0 || nx == 0) return;

    // We maintain a grid of 16 (scalar) accumulators. This gives a decent amount
    // of parallelism.
    f64 a00, a01, a02, a03;
    f64 a10, a11, a12, a13;
    f64 a20, a21, a22, a23;
    f64 a30, a31, a32, a33;
    f64 x0, x1, x2, x3;

    if (ny >= 4) {
        for (u32 i=0; i<=ny-4; i+=4) {
            a00=0; a01=0; a02=0; a03=0;
            a10=0; a11=0; a12=0; a13=0;
            a20=0; a21=0; a22=0; a23=0;
            a30=0; a31=0; a32=0; a33=0;
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
            y[0] += alpha * (a00 + a01);
            y[1] += alpha * (a10 + a11);
            y[2] += alpha * (a20 + a21);
            y[3] += alpha * (a30 + a31);
            A += 4*stride;
            y += 4;
        }
    }

    // Handle two of the remaining rows, then the final odd row.
    if (ny - ((ny>>2)<<2) >= 2) {
        a00=0; a01=0; a02=0; a03=0;
        a10=0; a11=0; a12=0; a13=0;
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
        y[0] += alpha * (a00 + a01);
        y[1] += alpha * (a10 + a11);
        A += 2*stride;
        y += 2;
    }
    if (ny - ((ny>>1)<<1) >= 1) {
        a00=0; a01=0; a02=0; a03=0;
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
        y[0] += alpha * (a00 + a01);
    }
}

void daocp_fma_mv_temporal_support(
    f64* __restrict__ y, const f64* __restrict__ A,
    const f64* __restrict__ x, f64 alpha, u32 ny, u32 nx, u32 stride,
    const daocp_workspace* wrk, u32 equality) {
    if (ny == 0 || nx == 0) return;

    f64 a00, a01, a02, a03;
    f64 a10, a11, a12, a13;
    f64 a20, a21, a22, a23;
    f64 a30, a31, a32, a33;
    f64 x0, x1, x2, x3;
    u32 var_tot = equality ? wrk->neta : wrk->nu_tot;
    u32* dim = equality ? wrk->rho : wrk->dims->nu;
    u32* cdim = equality ? wrk->crho : wrk->cnu;

    if (ny >= 4) {
        for (u32 i=0; i<=ny-4; i+=4) {
            a00=0; a01=0; a02=0; a03=0;
            a10=0; a11=0; a12=0; a13=0;
            a20=0; a21=0; a22=0; a23=0;
            a30=0; a31=0; a32=0; a33=0;
            // Compute temporal support for the 4 rows support 
            u32 csupport0 = DAOCP_CONSTRAINT_SUPPORT(wrk->as.xi2con+i, wrk->dims->N, var_tot, dim, cdim);
            u32 csupport1 = DAOCP_CONSTRAINT_SUPPORT(wrk->as.xi2con+i+1, wrk->dims->N, var_tot, dim, cdim);
            u32 csupport2 = DAOCP_CONSTRAINT_SUPPORT(wrk->as.xi2con+i+2, wrk->dims->N, var_tot, dim, cdim);
            u32 csupport3 = DAOCP_CONSTRAINT_SUPPORT(wrk->as.xi2con+i+3, wrk->dims->N, var_tot, dim, cdim);
            u32 ncols = DAOCP_MAX(csupport0, csupport1);
            u32 ncols1 = DAOCP_MAX(csupport2, csupport3);
            ncols = DAOCP_MAX(ncols, ncols1);
            ncols = DAOCP_MIN(ncols, nx);

            // Accumulate dot products
            if (ncols >= 4) {
                for (u32 j=0; j<=ncols-4; j+=4) {
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
            u32 ncols_mod = (ncols>>2)<<2;
            if (ncols - ncols_mod >= 2) {
                x0 = x[ncols_mod]; x1 = x[ncols_mod+1];
                a00 += A[ncols_mod]*x0; a01 += A[ncols_mod+1]*x1;
                a10 += A[stride + ncols_mod]*x0; a11 += A[stride+ncols_mod+1]*x1;
                a20 += A[2*stride + ncols_mod]*x0; a21 += A[2*stride+ncols_mod+1]*x1;
                a30 += A[3*stride + ncols_mod]*x0; a31 += A[3*stride+ncols_mod+1]*x1;
            }
            ncols_mod = (ncols>>1)<<1;
            if (ncols > ncols_mod) {
                x0 = x[ncols_mod];
                a00 += A[ncols_mod]*x0;
                a10 += A[stride + ncols_mod]*x0;
                a20 += A[2*stride + ncols_mod]*x0;
                a30 += A[3*stride + ncols_mod]*x0;
            }

            // Store
            a00 += a02; a01 += a03;
            a10 += a12; a11 += a13;
            a20 += a22; a21 += a23;
            a30 += a32; a31 += a33;
            y[0] += alpha * (a00 + a01);
            y[1] += alpha * (a10 + a11);
            y[2] += alpha * (a20 + a21);
            y[3] += alpha * (a30 + a31);
            A += 4*stride;
            y += 4;
        }
    }

    // Handle two of the remaining rows, then the final odd row.
    if (ny - ((ny>>2)<<2) >= 2) {
        a00=0; a01=0; a02=0; a03=0;
        a10=0; a11=0; a12=0; a13=0;
        u32 csupport0 = DAOCP_CONSTRAINT_SUPPORT(wrk->as.xi2con+((ny>>2)<<2), wrk->dims->N, var_tot, dim, cdim);
        u32 csupport1 = DAOCP_CONSTRAINT_SUPPORT(wrk->as.xi2con+((ny>>2)<<2)+1, wrk->dims->N, var_tot, dim, cdim);
        u32 ncols = DAOCP_MAX(csupport0, csupport1);
        ncols = DAOCP_MIN(ncols, nx);

        if (ncols >= 4) {
            for (u32 j=0; j<=ncols-4; j+=4) {
                x0 = x[j]; x1 = x[j+1]; x2 = x[j+2]; x3 = x[j+3];
                a00 += A[j]*x0; a01 += A[j+1]*x1;
                a02 += A[j+2]*x2; a03 += A[j+3]*x3;
                a10 += A[stride+j]*x0; a11 += A[stride+j+1]*x1;
                a12 += A[stride+j+2]*x2; a13 += A[stride+j+3]*x3;
            }
        }
        u32 ncols_mod = (ncols>>2)<<2;
        if (ncols - ncols_mod >= 2) {
            x0 = x[ncols_mod]; x1 = x[ncols_mod+1];
            a00 += A[ncols_mod]*x0; a01 += A[ncols_mod+1]*x1;
            a10 += A[stride+ncols_mod]*x0; a11 += A[stride+ncols_mod+1]*x1;
        }
        ncols_mod = (ncols>>1)<<1;
        if (ncols > ncols_mod) {
            x0 = x[ncols_mod];
            a00 += A[ncols_mod]*x0;
            a10 += A[stride+ncols_mod]*x0;
        }
        a00 += a02; a01 += a03;
        a10 += a12; a11 += a13;
        y[0] += alpha * (a00 + a01);
        y[1] += alpha * (a10 + a11);
        A += 2*stride;
        y += 2;
    }
    if (ny - ((ny>>1)<<1) >= 1) {
        a00=0; a01=0; a02=0; a03=0;
        u32 ncols = DAOCP_MIN(nx, DAOCP_CONSTRAINT_SUPPORT(wrk->as.xi2con+((ny>>1)<<1), wrk->dims->N, var_tot, dim, cdim));

        if (ncols >= 4) {
            for (u32 j=0; j<=ncols-4; j+=4) {
                a00 += A[j]*x[j]; a01 += A[j+1]*x[j+1];
                a02 += A[j+2]*x[j+2]; a03 += A[j+3]*x[j+3];
            }
        }
        u32 ncols_mod = (ncols>>2)<<2;
        if (ncols - ncols_mod >= 2) {
            a00 += A[ncols_mod]*x[ncols_mod];
            a01 += A[ncols_mod+1]*x[ncols_mod+1];
        }
        ncols_mod = (ncols>>1)<<1;
        if (ncols > ncols_mod) a00 += A[ncols_mod]*x[ncols_mod];
        a00 += a02; a01 += a03;
        y[0] += alpha * (a00 + a01);
    }
}

void daocp_fma_mv_t(
    f64* __restrict__ y, const f64* __restrict__ A, 
    const f64* __restrict__ x, u32 ny, u32 nx, u32 stride) {
    if (ny == 0 || nx == 0) return;

    // 8x2 grid of accumulators.
    f64 a00; f64 a01; f64 a10; f64 a11;
    f64 a20; f64 a21; f64 a30; f64 a31;
    f64 a40; f64 a41; f64 a50; f64 a51;
    f64 a60; f64 a61; f64 a70; f64 a71;
    f64 x0, x1;

    if (ny >= 8) {
        for (i32 i=0; i<=ny-8; i+=8) {
            // Load / Initialize
            a00=y[0]; a01=0; a10=y[1]; a11=0;
            a20=y[2]; a21=0; a30=y[3]; a31=0;
            a40=y[4]; a41=0; a50=y[5]; a51=0;
            a60=y[6]; a61=0; a70=y[7]; a71=0;
            // Accumulate
            if (nx >= 2) {
                for (i32 j=0; j<=nx-2; j+=2) {
                    x0 = x[j]; x1 = x[j+1];
                    a00 += A[j*stride]*x0; a01 += A[(j+1)*stride]*x1;
                    a10 += A[j*stride+1]*x0; a11 += A[(j+1)*stride+1]*x1;
                    a20 += A[j*stride+2]*x0; a21 += A[(j+1)*stride+2]*x1;
                    a30 += A[j*stride+3]*x0; a31 += A[(j+1)*stride+3]*x1;
                    a40 += A[j*stride+4]*x0; a41 += A[(j+1)*stride+4]*x1;
                    a50 += A[j*stride+5]*x0; a51 += A[(j+1)*stride+5]*x1;
                    a60 += A[j*stride+6]*x0; a61 += A[(j+1)*stride+6]*x1;
                    a70 += A[j*stride+7]*x0; a71 += A[(j+1)*stride+7]*x1;
                }
            }   
            u32 nx_mod = (nx>>1)<<1;
            if (nx - nx_mod >= 1) {
                x0 = x[nx_mod];
                a00 += A[nx_mod*stride]*x0;
                a10 += A[nx_mod*stride+1]*x0;
                a20 += A[nx_mod*stride+2]*x0;
                a30 += A[nx_mod*stride+3]*x0;
                a40 += A[nx_mod*stride+4]*x0;
                a50 += A[nx_mod*stride+5]*x0;
                a60 += A[nx_mod*stride+6]*x0;
                a70 += A[nx_mod*stride+7]*x0;
            }
            // Store
            a00 += a01; y[0] = a00;
            a10 += a11; y[1] = a10;
            a20 += a21; y[2] = a20;
            a30 += a31; y[3] = a30; 
            a40 += a41; y[4] = a40;
            a50 += a51; y[5] = a50;
            a60 += a61; y[6] = a60;
            a70 += a71; y[7] = a70;
            
            A += 8; y += 8;
        }
    }

    if (ny - ((ny>>3)<<3) >= 4) {
        // Load / Initialize
        a00=y[0]; a01=0; a10=y[1]; a11=0;
        a20=y[2]; a21=0; a30=y[3]; a31=0;
        // Accumulate
        if (nx >= 2) {
            for (i32 j=0; j<=nx-2; j+=2) {
                x0 = x[j]; x1 = x[j+1];
                a00 += A[j*stride]*x0; a01 += A[(j+1)*stride]*x1;
                a10 += A[j*stride+1]*x0; a11 += A[(j+1)*stride+1]*x1;
                a20 += A[j*stride+2]*x0; a21 += A[(j+1)*stride+2]*x1;
                a30 += A[j*stride+3]*x0; a31 += A[(j+1)*stride+3]*x1;
            }
        }   
        u32 nx_mod = (nx>>1)<<1;
        if (nx - nx_mod >= 1) {
            x0 = x[nx_mod];
            a00 += A[nx_mod*stride]*x0;
            a10 += A[nx_mod*stride+1]*x0;
            a20 += A[nx_mod*stride+2]*x0;
            a30 += A[nx_mod*stride+3]*x0;
        }
        // Store
        a00 += a01; y[0] = a00;
        a10 += a11; y[1] = a10;
        a20 += a21; y[2] = a20;
        a30 += a31; y[3] = a30;
        A += 4; y += 4;
    }

    if (ny - ((ny>>2)<<2) >= 2) {
        // Load / Initialize
        a00=y[0]; a01=0; a10=y[1]; a11=0;
        // Accumulate
        if (nx >= 2) {
            for (i32 j=0; j<=nx-2; j+=2) {
                x0 = x[j]; x1 = x[j+1];
                a00 += A[j*stride]*x0; a01 += A[(j+1)*stride]*x1;
                a10 += A[j*stride+1]*x0; a11 += A[(j+1)*stride+1]*x1;
            }
        }   
        u32 nx_mod = (nx>>1)<<1;
        if (nx - nx_mod >= 1) {
            x0 = x[nx_mod];
            a00 += A[nx_mod*stride]*x0;
            a10 += A[nx_mod*stride+1]*x0;
        }
        // Store
        a00 += a01; y[0] = a00;
        a10 += a11; y[1] = a10;
        A += 2; y += 2;
    }

    if (ny - ((ny>>1)<<1) >= 1) {
        a00=y[0]; a01=0;
        // Accumulate
        if (nx >= 2) {
            for (i32 j=0; j<=nx-2; j+=2) {
                x0 = x[j]; x1 = x[j+1];
                a00 += A[j*stride]*x0; a01 += A[(j+1)*stride]*x1;
            }
        }   
        u32 nx_mod = (nx>>1)<<1;
        if (nx - nx_mod >= 1) {
            x0 = x[nx_mod];
            a00 += A[nx_mod*stride]*x0;
        }
        // Store
        a00 += a01; y[0] = a00;
    }
}

void daocp_fma_mm_nt(f64* C, f64* A, f64* B, u32 nr, u32 nc, u32 k, u32 ostride) {
    for (u32 i=0; i<nr; ++i)
        for (u32 j=0; j<nc; ++j) {
            C[i*ostride + j] += daocp_dot(A+i*k, B+j*k, k);
        }
}

void daocp_fms_mm_nt(f64* C, f64* A, f64* B, u32 nr, u32 nc, u32 k, u32 ostride) {
    for (u32 i=0; i<nr; ++i)
        for (u32 j=0; j<nc; ++j) {
            C[i*ostride + j] -= daocp_dot(A+i*k, B+j*k, k);
        }
}

u32 daocp_gaussian_elimination(f64* A, f64* J, f64* tmp, u32 nr, u32 nc, u32 nctot, u32 R) {
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
        
        // Store swap index
        *((u32*) J) = pi;

        // Swap rows pi and rho
        if (rho != pi) {
            memcpy(tmp, A+rho*nctot, nctot*sizeof(f64));
            memcpy(A+rho*nctot, A+pi*nctot, nctot*sizeof(f64));
            memcpy(A+pi*nctot, tmp, nctot*sizeof(f64));
        }

        // Perform elimination step, while saving elimination coefficients
        for (u32 j=rho+1; j<nr; ++j) {
            f64 alpha = A[j*nctot + i] / p;
            J[j-rho] = alpha;
            for (u32 k=i; k<nctot; ++k) A[j*nctot + k] -= alpha * A[rho*nctot + k];
        }
        
        // Advance J pointer and increase rank
        J += nr - rho;
        rho += 1;
    }
    return rho;
}

void daocp_GE_transpose(f64* J, f64* mu, u32 m, u32 rho) {
    // We start from the end of J (pivots are visited in reversed order)
    J += (rho*(2*m + 1 - rho)) >> 1;

    for (i32 i=rho-1; i>=0; --i) {
        // Position at the beginning of the row.
        J -= m - i;
        // Transpose elimination
        mu[i] -= daocp_dot(J+1, mu+i+1, m-i-1);

        // Retrieve swap index from diagonal elements of J
        u32 k = *((u32*)J); 
        // Swap i-k components of mu
        f64 tmp = mu[k];
        mu[k] = mu[i];
        mu[i] = tmp;
    }
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
