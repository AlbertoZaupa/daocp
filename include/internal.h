/*
 * Copyright (c) 2026 Alberto Zaupa
 * SPDX-License-Identifier: MIT
 * See LICENSE in the project root for license information.
 */

#ifndef INTERNAL_H
#define INTERNAL_H

#include <daocp.h>
#include <string.h>

#define DAOCP_ABS(x) ((x) > 0 ? (x) : -(x))
#define DAOCP_MAX(x, y) ((x) > (y) ? (x) : (y))
#define DAOCP_MIN(x, y) ((x) < (y) ? (x) : (y))
#define DAOCP_PW2(x) (x)*(x)
#define DAOCP_ZERO_TOL 1e-12

typedef enum {
    DAOCP_BOUND_X = 0,
    DAOCP_BOUND_U = 1,
    DAOCP_ONLY_X = 2,
    DAOCP_ONLY_U = 3,
    DAOCP_MIXED = 4
} daocp_constraint_type;

typedef struct {
    u32 t;
    u32 idx;
    daocp_constraint_type type;
    u32 is_upper;
    u32 is_soft;
} daocp_constraint;

typedef struct {
    daocp_constraint* xi2con;
    u32** constraint_status;
    u32 n_active;
    u32 max_t;
    u32 n_valid_intermediate;
} daocp_active_set;

typedef struct {
    daocp_dims* dims;
    daocp_status status;
    daocp_active_set as;
    u32 iters;

    struct blasfeo_dmat* P;
    struct blasfeo_dmat* Luu;
    struct blasfeo_dmat* Lue;
    struct blasfeo_dmat* Lee;
    struct blasfeo_dmat* Ku;
    struct blasfeo_dmat* Ke;
    struct blasfeo_dvec* ux_lqr;
    struct blasfeo_dvec* eta_lqr;
    struct blasfeo_dvec* b;
    f64** u;
    f64** x;
    f64** eta;

    f64** lbx_wrk;
    f64** ubx_wrk;
    f64** lbu_wrk;
    f64** ubu_wrk;
    f64** lg_wrk;
    f64** ug_wrk;
    
    daocp_constraint_type** contypes; 
    u32* cnu;
    u32* rho;
    u32* crho;
    u32 nx_tot;
    u32 nu_tot;
    u32 nb_tot;
    u32 ng_tot;
    u32 neta;

    f64* xi;
    f64* xis;
    f64* p;
    f64* ps;
    f64* dual_linear;
    f64* dual_intermediate;
    f64* Ld;
    f64* Mu;
    f64* Me;
    u32* xi_sign;
    u32 W_stride;
    u32 nH0;
    u32 singular;
    u32 singular_idx;

    f64* H;
    f64* h;
    f64* tmp1;
    f64* GEtmp;
    f64* ABtmp;
    struct blasfeo_dmat tmp2;
    struct blasfeo_dvec costate0;
    struct blasfeo_dvec costate1;
} daocp_workspace;

// Solver logic
u32 daocp_check_x0_feasibility(daocp_workspace* wrk, daocp_qp* qp, daocp_args* args);
void daocp_solve_dual_eqcon_qp(daocp_workspace* wrk, daocp_qp* qp);
u32 daocp_is_step_dual_feasible(daocp_workspace* wrk, daocp_args* args);
u32 daocp_take_step(daocp_workspace* wrk);
u32 daocp_selection_greedy(daocp_workspace* wrk, daocp_qp* qp, daocp_args* args, daocp_constraint* violated);
u32 daocp_selection_most_violated(daocp_workspace* wrk, daocp_qp* qp, daocp_args* args, daocp_constraint* violated);
void daocp_add_to_working_set(daocp_workspace* wrk, daocp_qp* qp, daocp_constraint* violated, u32 is_slack);
void daocp_remove_from_working_set(daocp_workspace* wrk, daocp_qp* qp, u32 xi_idx);
u32 daocp_get_descent_dir(daocp_workspace* wrk, daocp_qp* qp);
u32 daocp_check_infeasibility_from_descent_dir(daocp_workspace* wrk);
void daocp_retrieve_sol(daocp_workspace* wrk, daocp_sol* sol);
u32 daocp_constraint_idx(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type);
u32 daocp_get_xi_idx(daocp_workspace* wrk, daocp_constraint* constr);
u32 daocp_is_active(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type);
u32 daocp_is_soft(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type);
u32 daocp_is_softened(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type);
void daocp_change_status(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type, u32 status);
void daocp_change_softening(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type, u32 soft_status);

// Solver update logic
u32 daocp_compute_chol_from_scratch(daocp_workspace* wrk, daocp_qp* qp);
void daocp_compute_dual_linear_term(daocp_workspace* wrk, daocp_qp* qp);
void daocp_reset_working_set(daocp_workspace* wrk);
void daocp_update(daocp_workspace* wrk, daocp_qp* qp, 
    f64* x0, struct blasfeo_dvec* rq, f64** lbx, 
    f64** ubx, f64** lbu, f64** ubu, f64** cl, f64** cu);

// Riccati and lqr routines
void daocp_solve_riccati(daocp_workspace* wrk, daocp_qp* qp);
void daocp_solve_lqr(daocp_workspace* wrk, daocp_qp* qp);

// Linear algebra and various utils
#define DAOCP_TRSVLQR(vu, ve, Luu, Lue, Lee, nu, rho) { \
    blasfeo_dtrsv_lnn(nu, Luu, 0, 0, &vu, 0, &vu, 0); \
    if ((rho) > 0) { \
        blasfeo_dgemv_n(rho, nu, -1.0, Lue, 0, 0, &vu, 0, 1.0, &ve, 0, &ve, 0); \
        blasfeo_dtrsv_lnn(rho, Lee, 0, 0, &ve, 0, &ve, 0); \
    } \
}
#define DAOCP_TRSVLQR_T(vu, ve, Luu, Lue, Lee, nu, rho) { \
    if ((rho) > 0) { \
        blasfeo_dtrsv_ltn(rho, Lee, 0, 0, &ve, 0, &ve, 0); \
        blasfeo_dgemv_t(rho, nu, -1.0, Lue, 0, 0, &ve, 0, 1.0, &vu, 0, &vu, 0); \
    } \
    blasfeo_dtrsv_ltn(nu, Luu, 0, 0, &vu, 0, &vu, 0); \
}
#define DAOCP_CONSTRAINT_SUPPORT(c, N, tot, dim, cdim) \
    ((c)->t == N ? tot : cdim[(c)->t] + (((c)->type != DAOCP_BOUND_X && (c)->type != DAOCP_ONLY_X) ? dim[(c)->t] : 0))
void daocp_trsv(f64* x, f64* L, u32 n, u32 stride);
void daocp_trsv_t(f64* x, f64* L, u32 n, u32 stride);
void daocp_fma_mv(f64* y, const f64* A, const f64* x, f64 alpha, u32 ny, u32 nx, u32 stride);
void daocp_fma_mv_temporal_support(f64* y, const f64* A, const f64* x, f64 alpha, u32 ny, u32 nx, u32 stride, const daocp_workspace* wrk, u32 equality);
void daocp_fma_mv_t(f64* y, const f64* A, const f64* x, u32 ny, u32 nx, u32 stride);
void daocp_fma_mm_nt(f64* C, f64* A, f64* B, u32 nr, u32 nc, u32 k, u32 ostride);
void daocp_fms_mm_nt(f64* C, f64* A, f64* B, u32 nr, u32 nc, u32 k, u32 ostride);
u32 daocp_gaussian_elimination(f64* A, f64* tmp, u32 nr, u32 nc, u32 nctot, u32 R);
void daocp_daxpy(const f64* x, f64* y, f64 a, u32 n);
void daocp_negate(f64* v, u32 n);
f64 daocp_dot(const f64* v, const f64* w, u32 n);
void daocp_pointer_swap(unsigned char** p1, unsigned char** p2);
void daocp_compute_prefix_sum(u32* ca, u32* a, u32 n);

#endif
