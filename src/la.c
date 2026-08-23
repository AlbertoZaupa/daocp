#include <defs.h>
#include <blasfeo.h>

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

void daocp_negate(f64* v, u32 n) {
    for (u32 i=0; i<n; ++i) v[i] *= -1.0;
}

f64 daocp_dot(f64* v, f64* w, u32 n) {
    f64 acc = 0.0;
    for (u32 i=0; i<n; ++i) acc += v[i] * w[i];
    return acc;
}