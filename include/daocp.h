#ifndef DAOCP_H
#define DAOCP_H

#include <defs.h>
#include <blasfeo.h>

// Dimensions of a OCP-structured QP.
// State and control dims can be time-varying.
// Same for constraints dimensions.
typedef struct {
    u32 N;
    u32* nx;
    u32* nu;
    u32* nb;
    u32* ng;
    u32* ne;
} daocp_dims;

// QP data in the format accepted by the solver.
typedef struct {
    daocp_dims *dim;
    f64* x0;
    struct blasfeo_dmat* Bt;
    struct blasfeo_dmat* At;
    struct blasfeo_dvec* w;
    struct blasfeo_dmat* R;
    struct blasfeo_dmat* Q;
    struct blasfeo_dmat* S;
    struct blasfeo_dvec* r;
    struct blasfeo_dvec* q;
    struct blasfeo_dmat* Cx;
    struct blasfeo_dmat* Cu;
    struct blasfeo_dvec* cl;
    struct blasfeo_dvec* cu;
    struct blasfeo_dvec* lbu;
    struct blasfeo_dvec* ubu;
    struct blasfeo_dvec* lbx;
    struct blasfeo_dvec* ubx;
    u32* is_eq;
    u32* idxbu;
    u32* idxbx;
} daocp_qp;

// Solution data
typedef struct {
    struct blasfeo_dvec* ux;
    struct blasfeo_dvec* lam;
} daocp_sol;

// DAOCP workspace memory
typedef struct {
    void* mem;
    u32 memsize;
} daocp_workspace;

// Constraint selection heuristic
typedef enum {
    DAOCP_SELECT_GREEDY = 0,
    DAOCP_SELECT_MOST_VIOLATED = 1
} daocp_selection;

// DAOCP arguments
typedef struct {
    u32 max_iter;
    daocp_selection selection;
} daocp_args;

void daocp_arg_set_default(daocp_args* args);
void daocp_workspace_memsize(daocp_dims* dims);
void daocp_workspace_create(daocp_dims* dims, daocp_args* args,
                            daocp_workspace* ws, void* memory);
void daocp_workspace_init(daocp_dims* dims, daocp_qp* qp, daocp_workspace* ws);
void daocp_solve(daocp_dims* dims, daocp_qp* qp, daocp_workspace* ws, daocp_sol* sol);


#endif