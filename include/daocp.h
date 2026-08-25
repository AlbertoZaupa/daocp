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
    u32* nbu;
    u32* nbx;
    u32* ng;
    u32* ne;
} daocp_dims;

// QP data in the format accepted by the solver.
typedef struct {
    daocp_dims dims;
    f64* x0;
    struct blasfeo_dmat* BAwt;
    struct blasfeo_dmat* RSQrq;
    f64** Cx;
    f64** Cu;
    f64** Dx;
    f64** Du;
    f64** cl;
    f64** cu;
    f64** d;
    f64** lbu;
    f64** ubu;
    f64** lbx;
    f64** ubx;
    u32** idxbu;
    u32** idxbx;
} daocp_qp;

// Solution data
typedef struct {
    struct blasfeo_dvec* ux;
} daocp_sol;

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

// DAOCP EXIT STATUS
typedef enum {
    DAOCP_SOLVED = 0,
    DAOCP_INFEASIBLE = 1,
    DAOCP_MAX_ITER = 2,
    DAOCP_ILL_CONDITIONED = 3
} daocp_status;

void daocp_args_set_default(daocp_args* args);
u32 daocp_qp_memsize(daocp_dims* dims);
void daocp_qp_memory_assign(daocp_dims* dims, daocp_qp* qp, void* memory);
u32 daocp_workspace_memsize(daocp_dims* dims);
void daocp_workspace_memory_assign(daocp_dims* dims, daocp_qp* qp, void* memory);
u32 daocp_sol_memsize(daocp_dims* dims);
void daocp_sol_memory_assign(daocp_dims* dims, daocp_sol* sol, void* memory);

void daocp_solve(daocp_args* args, daocp_qp* qp, void* ws, daocp_sol* sol);


#endif
