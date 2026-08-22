#ifndef INTERNAL_H
#define INTERNAL_H

#include <daocp.h>

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
    u32 W_stride;
    u32 nH0;

    struct blasfeo_dmat* H;
    struct blasfeo_dvec* h;
    struct blasfeo_dmat* tmp1;
    struct blasfeo_dmat* tmp2;
    struct blasfeo_dmat* tmp3;
    struct blasfeo_dmat* tmp4;
} daocp_ws_internal;


u32 daocp_global_constraint_idx(daocp_ws_internal* wrk, u32 t, u32 idx, daocp_constraint_type type);
u32 daocp_is_active(daocp_ws_internal* wrk, u32 t, u32 idx, daocp_constraint_type type);
void daocp_change_status(daocp_ws_internal* wrk, u32 t, u32 idx, daocp_constraint_type type, u32 status);

#endif