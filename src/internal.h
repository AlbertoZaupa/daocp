#ifndef INTERNAL_H
#define INTERNAL_H

#include <daocp.h>

#define ABS(x) (x > 0 ? x : -(x))
#define MAX(x, y) (x > y ? x : y)
#define MIN(x, y) (x < y ? x : y)
#define ZERO_TOL 1e-12

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
} daocp_constraint;

typedef struct {
    daocp_constraint* xi2con;
    u32* constraint_status;
    u32 n_active;
    u32 max_t;
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

    f64* lbx_wrk;
    f64* ubx_wrk;
    f64* lbu_wrk;
    f64* ubu_wrk;
    f64* lg_wrk;
    f64* ug_wrk;

    u32* cnx;
    u32* cnu;
    u32* cbx;
    u32* cbu;
    u32* cg;
    u32* rho;
    u32* crho;
    u32 nx_tot;
    u32 nu_tot;
    u32 nb_tot;
    u32 ng_tot;
    u32 neta;

    f64* xi;
    f64* p;
    f64* dual_linear;
    f64* Ld;
    f64* Mu;
    f64* Me;
    u32* xi_sign;
    u32 W_stride;
    u32 nH0;
    u32 singular;

    struct blasfeo_dmat H;
    struct blasfeo_dvec h;
    struct blasfeo_dmat tmp1;
    struct blasfeo_dmat tmp2;
    struct blasfeo_dmat tmp3;
    struct blasfeo_dmat tmp4;
    struct blasfeo_dvec tmp5;
    struct blasfeo_dvec costate0;
    struct blasfeo_dvec costate1;
} daocp_workspace;


// Solver logic
u32 daocp_check_x0_feasibility(daocp_workspace* wrk, daocp_qp* qp);
void daocp_solve_dual_eqcon_qp(daocp_workspace* wrk);
u32 daocp_is_dual_feasible(f64* p, u32* sign, u32 n);
u32 daocp_take_step(f64* xi, u32* xi_sign, f64*p, u32 n);
void daocp_selection_greedy(daocp_workspace* wrk, daocp_qp* qp, daocp_constraint* violated);
void daocp_selection_most_violated(daocp_workspace* wrk, daocp_qp* qp, daocp_constraint* violated);
void daocp_add_to_working_set(daocp_workspace* wrk, daocp_qp* qp, daocp_constraint* violated);
void daocp_remove_from_working_set(daocp_workspace* wrk, u32 xi_idx);
u32 daocp_get_descent_dir(daocp_workspace* wrk);
u32 daocp_check_infeasibility_from_descent_dir(f64* p, u32* sign, u32 n);
void daocp_retrieve_sol(daocp_workspace* wrk, daocp_sol* sol);
u32 daocp_global_constraint_idx(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type);
u32 daocp_is_active(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type);
void daocp_change_status(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type, u32 status);

// Linear algebra and various utils
#define TRSVLQR(vu, ve, Luu, Lue, Lee, nu, rho) { \
    blasfeo_dtrsv_lnn(nu, Luu, 0, 0, &vu, 0, &vu, 0); \
    blasfeo_dgemv_n(rho, nu, -1.0, Lue, 0, 0, &vu, 0, 1.0, &ve, 0, &ve, 0); \
    blasfeo_dtrsv_lnn(rho, Lee, 0, 0, &ve, 0, &ve, 0); \
}
#define TRSVLQR_T(vu, ve, Luu, Lue, Lee, nu, rho) { \
    blasfeo_dtrsv_ltn(rho, Lee, 0, 0, &ve, 0, &ve, 0); \
    blasfeo_dgemv_t(rho, nu, -1.0, Lue, 0, 0, &ve, 0, 1.0, &vu, 0, &vu, 0); \
    blasfeo_dtrsv_ltn(nu, Luu, 0, 0, &vu, 0, &vu, 0); \
}
void daocp_trsv(f64* x, f64* L, u32 n, u32 stride);
void daocp_trsv_t(f64* x, f64* L, u32 n, u32 stride);
void daocp_fma_mv(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride);
void daocp_fms_mv(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride);
void daocp_pointer_swap(unsigned char** p1, unsigned char** p2);

#endif