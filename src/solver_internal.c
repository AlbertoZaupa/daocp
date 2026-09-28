/*
 * Copyright (c) 2026 Alberto Zaupa
 * SPDX-License-Identifier: MIT
 * See LICENSE in the project root for license information.
 */

#include <internal.h>
#include <math.h>

u32 daocp_check_x0_feasibility(daocp_workspace* wrk, daocp_qp* qp, daocp_args* args) {
    // Check H x0 == h
    memcpy(wrk->tmp1, wrk->h, wrk->nH0*sizeof(f64));
    daocp_fma_mv(wrk->tmp1, wrk->H, qp->x0, -1.0, wrk->nH0, qp->dims.nx[0], qp->dims.nx[0]);
    for (u32 i=0; i<wrk->nH0; ++i)
        if (DAOCP_ABS(wrk->tmp1[i]) > args->primal_tol) return 1;
    return 0;
}

void daocp_solve_dual_eqcon_qp(daocp_workspace* wrk) {
    /*
        Solve linear system H p = - h
    */
    u32 n_active = wrk->as.n_active;
    
    // Reuse the cached solution of L y = -d.
    for (u32 i=wrk->as.n_valid_intermediate; i<n_active; ++i) {
        wrk->dual_intermediate[i] = -wrk->dual_linear[i]
            - daocp_dot(wrk->Ld + i*wrk->W_stride, wrk->dual_intermediate, i);
        wrk->dual_intermediate[i] *= wrk->Ld[i*wrk->W_stride + i];
    }
    wrk->as.n_valid_intermediate = n_active;

    // Solve L'p = y
    memcpy(wrk->p, wrk->dual_intermediate, n_active*sizeof(f64));
    daocp_trsv_t(wrk->p, wrk->Ld, n_active, wrk->W_stride);
}

u32 daocp_is_dual_feasible(daocp_args* args, f64* p, u32* sign, u32 n) {
    /*
        Check that all components have the right sign.   
    */
    for (u32 i=0; i<n; ++i) {
        if ((sign[i] == 0 && p[i] > args->dual_tol) || (sign[i] == 1 && p[i] < -args->dual_tol))
            return 0;
    }
    return 1;
}

u32 daocp_take_step(f64* xi, u32* xi_sign, f64*p, u32 n) {
    /*
        Line search along p[i] < 0 for i upper bounds, and p[i] > 0
        fo i lower bounds.
            xi + t*p = 0 \iff
            t = -xi / p
    */
    f64 t = INFINITY;
    u32 argmin = 0;
    for (u32 i=0; i<n; ++i) {
        if ((xi_sign[i] == 1 && p[i] > -DAOCP_ZERO_TOL) || 
            (xi_sign[i] == 0 && p[i] < DAOCP_ZERO_TOL)) continue;
        
        f64 tau = - xi[i] / p[i];
        if (tau < t) {
            t = tau;
            argmin = i;
        }
    }
    for (u32 i=0; i<n; ++i) xi[i] += t*p[i];
    return argmin;
}

static void compute_feedforwards(daocp_workspace* wrk, daocp_qp* qp) {
    f64* u = *wrk->u;
    f64* eta = *wrk->eta;
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1;
    memset(u, 0, wrk->nu_tot*sizeof(f64));
    memset(eta, 0, wrk->neta*sizeof(f64));

    // Get feedforwards (du, deta).
    u32 u_cols = wrk->cnu[wrk->as.max_t] + qp->dims.nu[wrk->as.max_t];
    u32 eta_cols = wrk->crho[wrk->as.max_t] + wrk->rho[wrk->as.max_t];
    daocp_fma_mv_t(u, wrk->Mu, wrk->xi, u_cols, wrk->as.n_active, wrk->nu_tot);
    daocp_fma_mv_t(eta, wrk->Me, wrk->xi, eta_cols, wrk->as.n_active, wrk->neta);
}

static inline void populate_constraint_struct(daocp_constraint* constr, u32 t, u32 idx, daocp_constraint_type type, u32 is_upper) {
    constr->t = t;
    constr->idx = idx;
    constr->is_upper = is_upper;
    constr->type = type;
}

static inline u32 check_bounds_at_t(
    daocp_workspace* wrk, daocp_args* args, 
    u32 t, u32 nb, u32* idxb, f64* v, f64* lb, 
    f64* ub, u32 is_state, daocp_constraint* constr
) {
    daocp_constraint_type type = is_state ? DAOCP_BOUND_X : DAOCP_BOUND_U;
    for (u32 i=0; i<nb; ++i) {
        if (daocp_is_active(wrk, t, i, type)) continue;
        u32 idx = idxb[i];
        f64 uval = v[idx];
        f64 tmp1 = uval - ub[i];
        f64 tmp2 = lb[i] - uval;
        if (tmp1 > args->primal_tol || tmp2 > args->primal_tol) {
            populate_constraint_struct(constr, t, i, type, tmp1 > args->primal_tol ? 1 : 0);
            return 1;
        }
    }
    return 0;
}

static inline u32 check_constraints_at_t(
    daocp_workspace* wrk, daocp_args* args,
    u32 t, u32 nc, u32 nx, u32 nu, 
    daocp_constraint_type* types, f64* Cx, f64* Cu, 
    f64* x, f64* u, f64* lb, f64* ub, daocp_constraint* constr
) {
    for (u32 i=0; i<nc; ++i) {
        if (daocp_is_active(wrk, t, i, types[i])) continue;
        f64 val = 0;
        if (types[i] == DAOCP_ONLY_U || types[i] == DAOCP_MIXED) {
            val += daocp_dot(Cu+i*nu, u, nu); 
        }
        if (types[i] == DAOCP_ONLY_X || types[i] == DAOCP_MIXED) {
            val += daocp_dot(Cx+i*nx, x, nx); 
        }
        f64 tmp1 = val - ub[i];
        f64 tmp2 = lb[i] - val;
        if (tmp1 > args->primal_tol || tmp2 > args->primal_tol) {
            populate_constraint_struct(constr, t, i, types[i], tmp1 > args->primal_tol ? 1 : 0);
            return 1;
        }
    }
    return 0;
}

void daocp_selection_greedy(
    daocp_workspace* wrk, daocp_qp* qp, 
    daocp_args* args, daocp_constraint* violated) {
    u32 N = qp->dims.N;
    daocp_constraint_type** contypes = wrk->contypes;
    u32* nx = qp->dims.nx; u32* nu = qp->dims.nu; 
    u32* rho = wrk->rho;
    u32* nbx = qp->dims.nbx; u32* nbu = qp->dims.nbu;
    u32* ng = qp->dims.ng;
    f64** u = wrk->u; f64** x = wrk->x; f64** eta = wrk->eta;
    f64** lbu = wrk->lbu_wrk; f64** ubu = wrk->ubu_wrk;
    f64** lbx = wrk->lbx_wrk; f64** ubx = wrk->ubx_wrk;
    u32** idxbx = qp->idxbx; u32** idxbu = qp->idxbu;
    f64** Cu = qp->Cu; f64** Cx = qp->Cx;
    f64** lg = wrk->lg_wrk; f64** ug = wrk->ug_wrk;

    // Compute feedforward terms
    compute_feedforwards(wrk, qp);

    // Run forward recursion and stop as soon as a violated constraint
    // is detected.
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1;
    violated->t = qp->dims.N+1;

    // First iteration (x0 = 0)
    v0.pa = u[0]; v1.pa = eta[0];
    blasfeo_dvecsc(nu[0], -1.0, &v0, 0);
    DAOCP_TRSVLQR_T(v0, v1, wrk->Luu, wrk->Lue, wrk->Lee, nu[0], rho[0]);
    if (check_bounds_at_t(wrk, args, 0, nbu[0], idxbu[0], u[0], lbu[0], ubu[0], 0, violated))
        return;
    if (check_constraints_at_t(wrk, args, 0, ng[0], nx[0], nu[0], contypes[0], Cx[0], Cu[0], x[0], u[0], lg[0], ug[0], violated)) return;
    // State evolution
    v1.pa = x[1];
    blasfeo_dgemv_t(nu[0], nx[1], 1.0, &qp->BAwt[0], 0, 0, &v0, 0, 0.0, &v1, 0, &v1, 0);
    if (check_bounds_at_t(wrk, args, 1, qp->dims.nbx[1], idxbx[1], x[1], lbx[1], ubx[1], 1, violated))
        return;

    for (u32 t=1; t<N; ++t) {
        // Compute control
        v0.pa = u[t]; v1.pa = x[t];
        blasfeo_dgemv_t(nx[t], nu[t], -1.0, wrk->Ku+t, 0, 0, &v1, 0, -1.0, &v0, 0, &v0, 0);
        v0.pa = eta[t];
        if (rho[t]) blasfeo_dgemv_t(nx[t], rho[t], 1.0, wrk->Ke+t, 0, 0, &v1, 0, 1.0, &v0, 0, &v0, 0);
        v0.pa = u[t]; v1.pa = eta[t];
        DAOCP_TRSVLQR_T(v0, v1, wrk->Luu+t, wrk->Lue+t, wrk->Lee+t, nu[t], rho[t]);

        // Check control bounds
        if (check_bounds_at_t(wrk, args, t, nbu[t], idxbu[t], u[t], lbu[t], ubu[t], 0, violated))
            return;
        // Check constraints at t
        if (check_constraints_at_t(wrk, args, t, ng[t], nx[t], nu[t], contypes[t], Cx[t], Cu[t], x[t], u[t], lg[t], ug[t], violated))
            return; 

        // Propagate state 
        memcpy(wrk->GEtmp, wrk->u[t], nu[t]*sizeof(f64));
        memcpy(wrk->GEtmp+nu[t], wrk->x[t], nx[t]*sizeof(f64));
        v0.pa = wrk->GEtmp;
        v1.pa = wrk->x[t+1];
        blasfeo_dgemv_t(nu[t]+nx[t], nx[t+1], 1.0, qp->BAwt+t, 0, 0,
                    &v0, 0, 0.0, &v1, 0, &v1, 0);

        // Check state bounds
        if (check_bounds_at_t(wrk, args, t+1, nbx[t+1], idxbx[t+1], x[t+1], lbx[t+1], ubx[t+1], 1, violated))
            return;
    }

    // Check constraints on terminal state
    check_constraints_at_t(wrk, args, N, ng[N], nx[N], 0, contypes[N], Cx[N], 0, x[N], 0, lg[N], ug[N], violated);
}

void daocp_selection_most_violated(
    daocp_workspace* wrk, daocp_qp* qp, 
    daocp_args* args, daocp_constraint* violated) {
    u32 N = qp->dims.N;
    daocp_constraint_type** contypes = wrk->contypes;
    u32* nx = qp->dims.nx; u32* nu = qp->dims.nu;
    u32* rho = wrk->rho;
    u32* nbx = qp->dims.nbx; u32* nbu = qp->dims.nbu;
    u32* ng = qp->dims.ng;
    f64** u = wrk->u; f64** x = wrk->x; f64** eta = wrk->eta;
    f64** lbu = wrk->lbu_wrk; f64** ubu = wrk->ubu_wrk;
    f64** lbx = wrk->lbx_wrk; f64** ubx = wrk->ubx_wrk;
    u32** idxbx = qp->idxbx; u32** idxbu = qp->idxbu;
    f64** Cu = qp->Cu; f64** Cx = qp->Cx;
    f64** lg = wrk->lg_wrk; f64** ug = wrk->ug_wrk;

    // Compute feedforward terms
    compute_feedforwards(wrk, qp);

    // Run forward recursion to compute the primal minimizer.
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1;
    v0.pa = u[0]; v1.pa = eta[0];
    blasfeo_dvecsc(nu[0], -1.0, &v0, 0);
    DAOCP_TRSVLQR_T(v0, v1, wrk->Luu, wrk->Lue, wrk->Lee, nu[0], rho[0]);
    v1.pa = x[1];
    blasfeo_dgemv_t(nu[0], nx[1], 1.0, &qp->BAwt[0], 0, 0,
                    &v0, 0, 0.0, &v1, 0, &v1, 0);

    for (u32 t=1; t<N; ++t) {
        // Compute control
        v0.pa = u[t]; v1.pa = x[t];
        blasfeo_dgemv_t(nx[t], nu[t], -1.0, wrk->Ku+t, 0, 0,
                        &v1, 0, -1.0, &v0, 0, &v0, 0);
        v0.pa = eta[t];
        if (rho[t]) blasfeo_dgemv_t(nx[t], rho[t], 1.0, wrk->Ke+t, 0, 0,
                        &v1, 0, 1.0, &v0, 0, &v0, 0);
        v0.pa = u[t]; v1.pa = eta[t];
        DAOCP_TRSVLQR_T(v0, v1, wrk->Luu+t, wrk->Lue+t, wrk->Lee+t, nu[t], rho[t]);

        // Propagate state
        memcpy(wrk->GEtmp, wrk->u[t], nu[t]*sizeof(f64));
        memcpy(wrk->GEtmp+nu[t], wrk->x[t], nx[t]*sizeof(f64));
        v0.pa = wrk->GEtmp;
        v1.pa = wrk->x[t+1];
        blasfeo_dgemv_t(nu[t]+nx[t], nx[t+1], 1.0, qp->BAwt+t, 0, 0,
                    &v0, 0, 0.0, &v1, 0, &v1, 0);
    }

    // Find the inactive constraint with the largest violation.
    f64 max_violation = args->primal_tol;
    violated->t = N+1;
    for (u32 t=0; t<=N; ++t) {
        if (t < N) {
            // Check control bounds
            for (u32 i=0; i<nbu[t]; ++i) {
                u32 idx = idxbu[t][i];
                if (daocp_is_active(wrk, t, i, DAOCP_BOUND_U)) continue;

                f64 tmp = u[t][idx] - ubu[t][i];
                if (tmp > max_violation) {
                    max_violation = tmp;
                    populate_constraint_struct(violated, t, i, DAOCP_BOUND_U, 1);
                }
                tmp = lbu[t][i] - u[t][idx];
                if (tmp > max_violation) {
                    max_violation = tmp;
                    populate_constraint_struct(violated, t, i, DAOCP_BOUND_U, 0);
                }
            }
        }

        // Check state bounds (x0 is fixed and handled separately)
        if (t > 0) {
            for (u32 i=0; i<nbx[t]; ++i) {
                u32 idx = idxbx[t][i];
                if (daocp_is_active(wrk, t, i, DAOCP_BOUND_X)) continue;

                f64 tmp = x[t][idx] - ubx[t][i];
                if (tmp > max_violation) {
                    max_violation = tmp;
                    populate_constraint_struct(violated, t, i, DAOCP_BOUND_X, 1);
                }
                tmp = lbx[t][i] - x[t][idx];
                if (tmp > max_violation) {
                    max_violation = tmp;
                    populate_constraint_struct(violated, t, i, DAOCP_BOUND_X, 0);
                }
            }
        }

        // Check general constraints
        for (u32 i=0; i<ng[t]; ++i) {
            daocp_constraint_type type = contypes[t][i];
            if (daocp_is_active(wrk, t, i, type)) continue;

            f64 val = 0.0;
            if (t < N && (type == DAOCP_ONLY_U || type == DAOCP_MIXED)) {
                val = daocp_dot(Cu[t]+i*nu[t], u[t], nu[t]); 
            }
            if (type == DAOCP_ONLY_X || type == DAOCP_MIXED) {
                val += daocp_dot(Cx[t]+i*nx[t], x[t], nx[t]); 
            }

            f64 tmp = val - ug[t][i];
            if (tmp > max_violation) {
                max_violation = tmp;
                populate_constraint_struct(violated, t, i, type, 1);
            }
            tmp = lg[t][i] - val;
            if (tmp > max_violation) {
                max_violation = tmp;
                populate_constraint_struct(violated, t, i, type, 0);
            }
        }
    }

}

static void compute_M_row(
    daocp_workspace* wrk, daocp_qp* qp,
    daocp_constraint* constr, u32 M_idx
) {
    /*
    Mu and Meta contain the positive and negative signature components
    of the constraint square-root response. 
    Input-only rows are given by the du, deta recursion initialized with 
    r[t] = C_{t,i}, r[tau!=t]=0, q=0, w=0.
    State-only rows are given by the same recursion initialized with
    q[t] = D_{t,i}, q[tau!=t]=0, r=0, w=0.
    Mixed tows are given by the du, deta recursion initialized with
    r[t] = C_t{t,i}, r[tau!=t]=0, q[t]=D_{t,i}, q[tau!=t]=0, w=0.
    */
    u32 t = constr->t;
    daocp_constraint_type type = constr->type;
    u32 idx = constr->idx;
    struct blasfeo_dvec* p = &wrk->costate0;
    struct blasfeo_dvec* ptmp = &wrk->costate1;
    u32* nx = qp->dims.nx;
    u32* nu = qp->dims.nu;
    u32* rho = wrk->rho;
    u32* cnu = wrk->cnu;
    u32* crho = wrk->crho;
    u32 nu_tot = wrk->nu_tot;
    u32 neta = wrk->neta;
    f64* Mu = wrk->Mu + nu_tot*M_idx;
    f64* Me = wrk->Me + neta*M_idx;
    memset(Mu, 0, nu_tot*sizeof(f64));
    memset(Me, 0, neta*sizeof(f64));
    struct blasfeo_dvec vu, ve;

    // FIRST STEP OF THE RECURSION

    // Initialize costate
    if (type == DAOCP_ONLY_X || type == DAOCP_MIXED) {
        memcpy(p->pa, qp->Cx[t]+idx*nx[t], nx[t]*sizeof(f64));
    } else if (type == DAOCP_BOUND_X) {
        memset(p->pa, 0, nx[t]*sizeof(f64));
        p->pa[qp->idxbx[t][idx]] = 1.0;
    } else {
        memset(p->pa, 0, nx[t]*sizeof(f64));
    }
    // Compute costate-independent feedforwards
    if (type == DAOCP_BOUND_U || type == DAOCP_ONLY_U || type == DAOCP_MIXED) {
        if (type != DAOCP_BOUND_U)
            memcpy(Mu+cnu[t], qp->Cu[t]+idx*nu[t], nu[t]*sizeof(f64));
        else *(Mu+cnu[t]+qp->idxbu[t][idx]) = 1.0;
        vu.pa = Mu+cnu[t]; ve.pa = Me+crho[t];
        DAOCP_TRSVLQR(vu, ve, &wrk->Luu[t], &wrk->Lue[t], &wrk->Lee[t], nu[t], rho[t]);
        // Add costate contribution
        if (t > 0) {
            blasfeo_dgemv_n(nx[t], nu[t], -1.0, &wrk->Ku[t], 0, 0, &vu, 0, 1.0, p, 0, p, 0);
            if (rho[t]) blasfeo_dgemv_n(nx[t], rho[t], 1.0, &wrk->Ke[t], 0, 0, &ve, 0, 1.0, p, 0, p, 0);
        }
    }

    for (i32 tau=t-1; tau>=0; --tau) {
        // Compute B' p
        vu.pa = Mu + cnu[tau]; 
        blasfeo_dgemv_n(nu[tau], nx[tau+1], 1.0, &qp->BAwt[tau], 0, 0, p, 0, 0.0, &vu, 0, &vu, 0);
        // Solve Lu [du; deta] = [B' p; 0]
        ve.pa = Me + crho[tau];
        DAOCP_TRSVLQR(vu, ve, wrk->Luu+tau, wrk->Lue+tau, wrk->Lee+tau, nu[tau], rho[tau]);
        if (tau == 0) break;
        // p = A[tau]' p - Ku du + Keta deta
        blasfeo_dgemv_n(nx[tau], nx[tau+1], 1.0, qp->BAwt+tau, nu[tau], 0, p, 0, 0.0, ptmp, 0, ptmp, 0);
        blasfeo_dgemv_n(nx[tau], nu[tau], -1.0, wrk->Ku+tau, 0, 0, &vu, 0, 1.0, ptmp, 0, ptmp, 0);
        if (rho[tau]) blasfeo_dgemv_n(nx[tau], rho[tau], 1.0, wrk->Ke+tau, 0, 0, &ve, 0, 1.0, ptmp, 0, ptmp, 0);
        daocp_pointer_swap((unsigned char**) &p, (unsigned char**) &ptmp);
    }
}

static void compute_hessian_row(daocp_workspace* wrk, daocp_qp* qp, daocp_constraint* constr) {
    /*
        dH = Mu Mu' - Meta Meta'.
        1) First compute the new rows of Mu and Meta.
        2) Compute their signed products with existing rows.
    */
    u32 n_active = wrk->as.n_active;
    compute_M_row(wrk, qp, constr, n_active);
    
    // Write the new row of H into the new row of L.
    memset(wrk->Ld + n_active*wrk->W_stride, 0, (n_active+1)*sizeof(f64));
    f64* mu_ptr = wrk->Mu + n_active*wrk->nu_tot;
    f64* me_ptr = wrk->Me + n_active*wrk->neta;
    u32 nu_cols = DAOCP_CONSTRAINT_SUPPORT(constr, wrk->dims->N, wrk->nu_tot, wrk->dims->nu, wrk->cnu);
    u32 eta_cols = DAOCP_CONSTRAINT_SUPPORT(constr, wrk->dims->N, wrk->neta, wrk->rho, wrk->crho);
    daocp_fma_mv_temporal_support(wrk->Ld + n_active*wrk->W_stride,
        wrk->Mu, mu_ptr, 1.0, n_active, nu_cols, wrk->nu_tot, wrk, 0);
    daocp_fma_mv_temporal_support(wrk->Ld + n_active*wrk->W_stride,
        wrk->Me, me_ptr, -1.0, n_active, eta_cols, wrk->neta, wrk, 1);
    wrk->Ld[n_active*wrk->W_stride + n_active] = daocp_dot(mu_ptr, mu_ptr, nu_cols)
                                                - daocp_dot(me_ptr, me_ptr, eta_cols);
}

static void update_cholesky_add(daocp_workspace* wrk) {
    u32 n_active = wrk->as.n_active;
    // Solve
    daocp_trsv(wrk->Ld + n_active*wrk->W_stride, wrk->Ld, n_active, wrk->W_stride);
    // Diagonal
    wrk->Ld[n_active*wrk->W_stride + n_active] -= 
            daocp_dot(wrk->Ld+n_active*wrk->W_stride, wrk->Ld+n_active*wrk->W_stride, n_active);
    if (wrk->Ld[n_active*wrk->W_stride + n_active] < DAOCP_ZERO_TOL) {
        wrk->singular = 1;
        wrk->Ld[n_active*wrk->W_stride + n_active] = 0.0;
    } else wrk->Ld[n_active*wrk->W_stride + n_active] = 1 / sqrt(wrk->Ld[n_active*wrk->W_stride + n_active]);
}

static void daocp_update_working_set__add(daocp_workspace* wrk, daocp_constraint* constr) {
    u32 t = constr->t;
    u32 idx = constr->idx;
    daocp_constraint_type type = constr->type;
    u32 is_upper = constr->is_upper;

    // Update ( xi_idx -> constraint ) map.
    daocp_constraint new_constraint = {t, idx, type, is_upper};
    wrk->as.xi2con[wrk->as.n_active] = new_constraint;

    wrk->xi[wrk->as.n_active] = 0.0;
    wrk->xi_sign[wrk->as.n_active] = is_upper ? 1 : 0;
    daocp_change_status(wrk, t, idx, type, 1);
    wrk->as.n_active += 1;
    
    // Update max_t, depending on the constraint type
    if (type == DAOCP_BOUND_U || type == DAOCP_ONLY_U || type == DAOCP_MIXED)
        wrk->as.max_t = DAOCP_MAX(wrk->as.max_t, t);
    else wrk->as.max_t = DAOCP_MAX(wrk->as.max_t, t-1);
}

void daocp_add_to_working_set(daocp_workspace* wrk, daocp_qp* qp, daocp_constraint* violated) {
    u32 t = violated->t;
    u32 idx = violated->idx;
    daocp_constraint_type type = violated->type;
    u32 is_upper = violated->is_upper;
    
    // Update cholesky factorization
    compute_hessian_row(wrk, qp, violated);
    update_cholesky_add(wrk);

    // Update dual linear term
    f64 tmp;
    if (type == DAOCP_BOUND_X)
        tmp = (is_upper ? wrk->ubx_wrk : wrk->lbx_wrk)[t][idx];
    else if (type == DAOCP_BOUND_U)
        tmp = (is_upper ? wrk->ubu_wrk : wrk->lbu_wrk)[t][idx];
    else
        tmp = (is_upper ? wrk->ug_wrk : wrk->lg_wrk)[t][idx];
    wrk->dual_linear[wrk->as.n_active] = tmp;

    // Update working set data structures
    daocp_update_working_set__add(wrk, violated);
}

static void update_cholesky_remove(daocp_workspace* wrk, u32 idx) {
    u32 n_active = wrk->as.n_active;
    u32 W_stride = wrk->W_stride;
    f64* l = wrk->Ld + n_active*W_stride;

    // Remove row at idx.
    for (u32 i=idx+1; i<n_active; ++i)
        memcpy(wrk->Ld+(i-1)*W_stride, wrk->Ld+i*W_stride, (i+1)*sizeof(f64));

    // Extract column[idx] at l. Fix bottom-right lower triangle
    for (u32 i=idx; i<n_active-1; ++i) {
        l[i-idx] = wrk->Ld[i*W_stride + idx];
        for (u32 j=idx+1; j<=i+1; ++j)
            wrk->Ld[i*W_stride + j-1] = wrk->Ld[i*W_stride + j];
    }

    // Perform rank1 update of bottom-right lower triangle
    f64 lii, lii_new, a, b;
    for (u32 i=idx; i<n_active-1; ++i) {
        // A singular last pivot is stored as zero, not its reciprocal.
        // Removing an older constraint can restore a positive pivot.
        lii = wrk->Ld[i*W_stride+i];
        if (lii != 0.0) lii = 1.0 / lii;
        lii_new = 1 / sqrt(DAOCP_PW2(lii) + DAOCP_PW2(l[i-idx])); 
        wrk->Ld[i*W_stride+i] = lii_new;
        a = l[i-idx] * lii_new;
        b = lii * lii_new;
        for (u32 j=i+1; j<n_active-1; ++j) {
            lii = l[j-idx];
            lii_new = wrk->Ld[j*W_stride + i];
            l[j-idx] = lii * b - lii_new * a;
            wrk->Ld[j*W_stride + i] = b * lii_new + a * lii;
        }
    }

    // Clear singularity flag
    wrk->singular = 0;
}

static void daocp_update_working_set__remove(daocp_workspace* wrk, u32 xi_idx) {
    // Retrieve constraint info.
    u32 t = wrk->as.xi2con[xi_idx].t;
    u32 idx = wrk->as.xi2con[xi_idx].idx;
    daocp_constraint_type type = wrk->as.xi2con[xi_idx].type;

    // Update (xi_idx -> constraint info) map.
    for (u32 i=xi_idx+1; i<wrk->as.n_active; ++i) 
        wrk->as.xi2con[(i-1)] = wrk->as.xi2con[i]; 

    // Compact xi and xi_sign
    for (u32 i=xi_idx+1; i < wrk->as.n_active; ++i)
        wrk->xi[i-1] = wrk->xi[i];
    for (u32 i=xi_idx+1; i < wrk->as.n_active; ++i)
        wrk->xi_sign[i-1] = wrk->xi_sign[i];
    
    daocp_change_status(wrk, t, idx, type, 0);
    wrk->as.n_active -= 1;
    wrk->as.n_valid_intermediate = DAOCP_MIN(wrk->as.n_valid_intermediate, xi_idx);
    
    // Recompute max_t
    wrk->as.max_t = 0;
    for (u32 i=0; i<wrk->as.n_active; ++i) {
        u32 mask = wrk->as.xi2con[i].type == DAOCP_BOUND_X || wrk->as.xi2con[i].type == DAOCP_ONLY_X;
        wrk->as.max_t = DAOCP_MAX(wrk->as.max_t, wrk->as.xi2con[i].t - mask*1);
    }
}

void daocp_remove_from_working_set(daocp_workspace* wrk, u32 xi_idx) {
    // Update cholesky of dH
    update_cholesky_remove(wrk, xi_idx);

    // Update linear term
    for (u32 i=xi_idx+1; i<wrk->as.n_active; ++i)
        wrk->dual_linear[i-1] = wrk->dual_linear[i];

    // Update Mu and Meta
    for (u32 i=xi_idx+1; i<wrk->as.n_active; ++i) {
        memcpy(wrk->Mu + (i-1)*wrk->nu_tot,
               wrk->Mu + i*wrk->nu_tot,
               wrk->nu_tot*sizeof(f64));
        memcpy(wrk->Me + (i-1)*wrk->neta,
               wrk->Me + i*wrk->neta,
               wrk->neta*sizeof(f64));
    }

    // Update active set
    daocp_update_working_set__remove(wrk, xi_idx);
}

u32 daocp_get_descent_dir(daocp_workspace* wrk) {
    u32 n_active = wrk->as.n_active;
    u32 W_stride = wrk->W_stride;

    // Solve LL'p = 0, p != 0. Assume L_{n_active, n_active} = 0.
    memset(wrk->p, 0, n_active*sizeof(f64));
    wrk->p[n_active-1] = 1.0;
    for (u32 i=0; i<n_active-1; ++i)
        wrk->p[i] = -wrk->Ld[(n_active-1)*W_stride + i];
    daocp_trsv_t(wrk->p, wrk->Ld, n_active-1, W_stride);

    // Enforce p' b < 0.
    f64 dotv = daocp_dot(wrk->p, wrk->dual_linear, n_active);
    if (dotv >= -DAOCP_ZERO_TOL && dotv <= DAOCP_ZERO_TOL) return 1;
    if (dotv > DAOCP_ZERO_TOL)
        daocp_negate(wrk->p, n_active);
    return 0;
}

u32 daocp_check_infeasibility_from_descent_dir(f64* p, u32* sign, u32 n) {
    for (u32 i=0; i<n; ++i)
        if ((sign[i] == 1 && p[i] < -DAOCP_ZERO_TOL) || (sign[i] == 0 && p[i] > DAOCP_ZERO_TOL)) return 0;

    return 1; 
}

u32 daocp_constraint_idx(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type) {
    u32 base = idx;
    u32 flag = type != DAOCP_BOUND_U;
    base += flag * wrk->dims->nbu[t];
    flag = flag && (type != DAOCP_BOUND_X);
    base += flag * wrk->dims->nbx[t];
    return base;
}

u32 daocp_is_active(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type) {
    return wrk->as.constraint_status[t][2*daocp_constraint_idx(wrk, t, idx, type)];
}

u32 daocp_is_soft(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type) {
    return 1;
}

u32 daocp_is_softened(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type) {
    return wrk->as.constraint_status[t][2*daocp_constraint_idx(wrk, t, idx, type) + 1];
}

void daocp_change_status(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type, u32 status) {
    wrk->as.constraint_status[t][2*daocp_constraint_idx(wrk, t, idx, type)] = status;
}

void daocp_change_softening(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type, u32 soft_status) {
    wrk->as.constraint_status[t][2*daocp_constraint_idx(wrk, t, idx, type) + 1] = soft_status;
}

void daocp_retrieve_sol(daocp_workspace* wrk, daocp_sol* sol) {
    struct blasfeo_dvec v;
    v.pa = wrk->u[0];
    blasfeo_daxpy(wrk->dims->nu[0], 1.0, &v, 0, wrk->ux_lqr, 0, sol->ux, 0);
    blasfeo_dveccp(wrk->dims->nx[0], wrk->ux_lqr, wrk->dims->nu[0], sol->ux, wrk->dims->nu[0]);
    
    for (u32 t=1; t<wrk->dims->N; ++t) {
        v.pa = wrk->u[t];
        blasfeo_daxpy(wrk->dims->nu[t], 1.0, &v, 0, wrk->ux_lqr+t, 0, sol->ux+t, 0);
        v.pa = wrk->x[t];
        blasfeo_daxpy(wrk->dims->nx[t], 1.0, &v, 0, wrk->ux_lqr+t, wrk->dims->nu[t], sol->ux+t, wrk->dims->nu[t]);
    }
    v.pa = wrk->x[wrk->dims->N];
    blasfeo_daxpy(wrk->dims->nx[wrk->dims->N], 1.0, &v, 0, wrk->ux_lqr+wrk->dims->N, 0, sol->ux+wrk->dims->N, 0);
}

u32 daocp_compute_chol_from_scratch(daocp_workspace* wrk, daocp_qp* qp) {
    u32 n_active = wrk->as.n_active;
    u32 W_stride = wrk->W_stride;
    wrk->as.n_valid_intermediate = 0;

    // Compute Mu and Meta from scratch
    for (u32 ci=0; ci<n_active; ++ci) {
        daocp_constraint* constr = wrk->as.xi2con + ci;
        compute_M_row(wrk, qp, constr, ci);
    }

    // Compute dH = Mu Mu' - Meta Meta'
    memset(wrk->Ld, 0, n_active*W_stride*sizeof(f64));
    // Specialized syrk algorithm.
    u32 nu_cols = wrk->cnu[wrk->as.max_t] + wrk->dims->nu[wrk->as.max_t];
    u32 eta_cols = wrk->crho[wrk->as.max_t] + wrk->rho[wrk->as.max_t];
    for (u32 i=0; i<n_active; ++i)
        for (u32 j=0; j<n_active; ++j) {
            wrk->Ld[i*W_stride+j] += daocp_dot(wrk->Mu+i*wrk->nu_tot, wrk->Mu+j*wrk->nu_tot, nu_cols);
        }
    for (u32 i=0; i<n_active; ++i)
        for (u32 j=0; j<n_active; ++j) {
            wrk->Ld[i*W_stride+j] -= daocp_dot(wrk->Me+i*wrk->neta, wrk->Me+j*wrk->neta, eta_cols);
        }

    // Compute chol(dH), checking for singularity.
    for (u32 i=0; i<n_active; ++i) {
        f64* lii = wrk->Ld + i*W_stride + i;
        if (*lii < DAOCP_ZERO_TOL) return 1; // Detected singularity.
        *lii = 1 / sqrt(*lii);
        for (u32 j=i+1; j<n_active; ++j) wrk->Ld[j*W_stride + i] *= *lii;
        for (u32 j=i+1; j<n_active; ++j)
            for (u32 k=i+1; k<n_active; ++k)
                wrk->Ld[j*W_stride + k] -= wrk->Ld[j*W_stride + i] * wrk->Ld[k*W_stride + i];
    }
    return 0;
}

void daocp_reset_working_set(daocp_workspace* wrk) {
    for (u32 t=0; t<=wrk->dims->N; ++t) {
        u32 nbu = wrk->dims->nbu[t];
        u32 nbx = wrk->dims->nbx[t];
        u32 ng = wrk->dims->ng[t];
        for (u32 i=0; i<2*nbu; ++i) wrk->as.constraint_status[t][i] = 0; 
        for (u32 i=0; i<2*nbx; ++i) wrk->as.constraint_status[t][nbu+i] = 0; 
        for (u32 i=0; i<2*ng; ++i) wrk->as.constraint_status[t][nbu+nbx+i] = 0; 
    }
    wrk->as.n_active = 0;
    wrk->as.n_valid_intermediate = 0;
    wrk->singular = 0;
    wrk->as.max_t = 0;
}

void daocp_update(daocp_workspace* wrk, daocp_qp* qp, 
    f64* x0, struct blasfeo_dvec* rq, f64** lbx, 
    f64** ubx, f64** lbu, f64** ubu, f64** cl, f64** cu)
{
    u32 N = qp->dims.N;
    u32* nx = qp->dims.nx;
    u32* nu = qp->dims.nu;
    // Update x0
    memcpy(qp->x0, x0, nx[0]*sizeof(f64));
    // Update cost
    for (u32 t=0; t<=N; ++t)
        blasfeo_drowin(nx[t]+nu[t], 1.0, rq+t, 0, qp->RSQrq+t, nx[t]+nu[t], 0);
    // Update bounds
    u32* nbu = qp->dims.nbu;
    u32* nbx = qp->dims.nbx;
    u32* ng = qp->dims.ng;
    for (u32 t=0; t<=N; ++t) {
        if (t>0 && lbx[t] != qp->lbx[t])
            memcpy(qp->lbx[t], lbx[t], nbx[t]*sizeof(f64));
        if (t>0 && ubx[t] != qp->ubx[t])
            memcpy(qp->ubx[t], ubx[t], nbx[t]*sizeof(f64));
        if (t<N && lbu[t] != qp->lbu[t])
            memcpy(qp->lbu[t], lbu[t], nbu[t]*sizeof(f64));
        if (t<N && ubu[t] != qp->ubu[t])
            memcpy(qp->ubu[t], ubu[t], nbu[t]*sizeof(f64));
        if (cl[t] != qp->cl[t]) memcpy(qp->cl[t], cl[t], ng[t]*sizeof(f64));
        if (cu[t] != qp->cu[t]) memcpy(qp->cu[t], cu[t], ng[t]*sizeof(f64));
    }

    if (wrk->singular) daocp_reset_working_set(wrk);

    // Solve LQR
    daocp_solve_lqr(wrk, qp);
    for (u32 i=0; i<wrk->as.n_active; ++i) {
        daocp_constraint* c = wrk->as.xi2con + i;
        f64* p;
        switch (c->type) {
            case DAOCP_BOUND_U:
                p = (c->is_upper ? wrk->ubu_wrk : wrk->lbu_wrk)[c->t];
                break;
            case DAOCP_BOUND_X:
                p = (c->is_upper ? wrk->ubx_wrk : wrk->lbx_wrk)[c->t];
                break;
            default:
                p = (c->is_upper ? wrk->ug_wrk : wrk->lg_wrk)[c->t];
                break;
        }
        wrk->dual_linear[i] = p[c->idx];
    }
    wrk->as.n_valid_intermediate = 0;
}

void daocp_pointer_swap(unsigned char** p1, unsigned char** p2) {
    unsigned char* p3 = *p1;
    *p1 = *p2;
    *p2 = p3;
}

void daocp_compute_prefix_sum(u32* ca, u32* a, u32 n) {
    u32 sum = 0;
    for (u32 i=0; i<n; ++i) {
        ca[i] = sum;
        sum += a[i];
    }
}
