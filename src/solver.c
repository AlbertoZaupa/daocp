/*
 * Copyright (c) 2026 Alberto Zaupa
 * SPDX-License-Identifier: MIT
 * See LICENSE in the project root for license information.
 */

#include <internal.h>

void daocp_solve(
    daocp_args* args, daocp_qp* qp, void* ws,
    daocp_sol* sol
) {
    daocp_workspace* wrk = (daocp_workspace*) ws;
    u32 (*selection_handle)(daocp_workspace*, daocp_qp*, daocp_args* args, daocp_constraint* ) =
        args->selection == DAOCP_SELECT_GREEDY ? daocp_selection_greedy 
        : daocp_selection_most_violated;
    u32 status_set = 0;

    // Check feasibility on x0
    if (daocp_check_x0_feasibility(wrk, qp, args)) {
        wrk->status = DAOCP_INFEASIBLE;
        wrk->iters = 0;
        // Set ux to the lqr solution and return
        for (u32 t=0; t<qp->dims.N; ++t) 
            blasfeo_dveccp(qp->dims.nu[t]+qp->dims.nx[t], wrk->ux_lqr+t, 0, sol->ux+t, 0);
        blasfeo_dveccp(qp->dims.nx[qp->dims.N], wrk->ux_lqr+qp->dims.N, 0, sol->ux+qp->dims.N, 0);
        return;
    }

    for (u32 k=0; k<args->max_iter; ++k) {
        if (!wrk->singular) {
            // Solve H_W p = -g_W
            daocp_solve_dual_eqcon_qp(wrk, qp);
            if (daocp_is_step_dual_feasible(wrk, args)) {
                // If p is dual feasible, xi = p
                memcpy(wrk->xi, wrk->p, wrk->as.n_active*sizeof(f64));
                memcpy(wrk->xis, wrk->ps, wrk->as.n_active*sizeof(f64));
                // Add a primal-violated constraint, if it exists
                daocp_constraint violated;
                u32 is_slack = selection_handle(wrk, qp, args, &violated);
                if (violated.t > qp->dims.N) {
                    wrk->status = DAOCP_SOLVED;
                    wrk->iters = k;
                    status_set = 1;
                    break;
                }
                daocp_add_to_working_set(wrk, qp, &violated, is_slack); 
            } else {
                // Form descent direction
                for (u32 i=0; i<wrk->as.n_active; ++i) wrk->p[i] -= wrk->xi[i];
                for (u32 i=0; i<wrk->as.n_active; ++i) wrk->ps[i] -= wrk->xis[i];
                u32 idx_remove = daocp_take_step(wrk);
                daocp_remove_from_working_set(wrk, qp, idx_remove);
            }
        } else {
            // Retrieve a descent direction by exploiting infeasibility
            // of the dual equality constrained problem.
            if (daocp_get_descent_dir(wrk, qp)) {
                wrk->status = DAOCP_ILL_CONDITIONED;
                wrk->iters = k;
                status_set = 1;
                break;
            }
            // Check for an infeasibility certificate
            if (daocp_check_infeasibility_from_descent_dir(wrk)) {
                wrk->status = DAOCP_INFEASIBLE;
                wrk->iters = k;
                status_set = 1;
                break;
            }
            u32 idx_remove = daocp_take_step(wrk);
            daocp_remove_from_working_set(wrk, qp, idx_remove);
        }
    }
    if (status_set==0) {
        wrk->status = DAOCP_MAX_ITER;
        wrk->iters = args->max_iter;
    }

    daocp_retrieve_sol(wrk, sol);
}
