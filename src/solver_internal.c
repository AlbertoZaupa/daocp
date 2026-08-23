#include <internal.h>
#include <math.h>

u32 daocp_check_x0_feasibility(daocp_workspace* wrk, daocp_qp* qp) {
    // Check H x0 == h
    blasfeo_dgemv_n(wrk->nH0, qp->dims->nx[0], -1.0, &wrk->H, 0, 0,
                qp->x0, 0, 1.0, &wrk->h, 0, &wrk->tmp5, 0);
    for (u32 i=0; i<wrk->nH0; ++i)
        if (ABS(wrk->tmp5.pa[i] - wrk->h.pa[i]) > ZERO_TOL) return 1;
    return 0;
}

void daocp_solve_dual_eqcon_qp(daocp_workspace* wrk) {
    /*
        Solve linear system H p = - h
    */
    u32 n_active = wrk->as.n_active;
    
    // copy d into p
    for (u32 i=0; i<n_active; ++i) wrk->p[i] = -wrk->dual_linear[i];

    // Solve L y = -h
    daocp_trsv(wrk->p, wrk->Ld, n_active, wrk->W_stride);
    // Solve L'p = y
    daocp_trsv_t(wrk->p, wrk->Ld, n_active, wrk->W_stride);
}

u32 daocp_is_dual_feasible(f64* p, u32* sign, u32 n) {
    /*
        Check that all components have the right sign.   
    */
    for (u32 i=0; i<n; ++i) {
        if ((sign[i] == 0 && p[i] > ZERO_TOL) || (sign[i] == 1 && p[i] < -ZERO_TOL))
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
        if ((xi_sign[i] == 1 && p[i] > -ZERO_TOL) || 
            (xi_sign[i] == 0 && p[i] < ZERO_TOL)) continue;
        
        f64 tau = - xi[i] / p[i];
        if (tau < t) {
            t = tau;
            argmin = i;
        }
    }
    for (u32 i=0; i<n; ++i) xi[i] += t*p[i];
    return argmin;
}

void daocp_selection_greedy(daocp_workspace* wrk, daocp_qp* qp, daocp_constraint* violated) {

}

void daocp_selection_most_violated(daocp_workspace* wrk, daocp_qp* qp, daocp_constraint* violated) {

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
    u32 N = qp->dims->N;
    u32* nx = qp->dims->nx;
    u32* nu = qp->dims->nu;
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
    if (constr->type == DAOCP_ONLY_X || constr->type == DAOCP_MIXED) {
        memcpy(p->pa, qp->Cx[t]+idx*nx[t], nx[t]*sizeof(f64));
    } else if (constr->type == DAOCP_BOUND_X) {
        memset(p->pa, 0, nx[t]*sizeof(f64));
        p->pa[idx] = 1.0;
    }
    // Compute costate-independent feedforwards
    if (constr->type == DAOCP_ONLY_U || constr->type == DAOCP_MIXED) {
        memcpy(Mu+cnu[t], qp->Cu[t]+idx*nu[t], nu[t]*sizeof(f64));
        vu.pa = Mu+cnu[t]; ve.pa = Me+crho[t];
        TRSVLQR(vu, ve, &wrk->Luu[t], &wrk->Lue[t], &wrk->Lee[t], nu[t], rho[t]);
        // Add costate contribution
        if (t > 0) {
            blasfeo_dgemv_n(nx[t], nu[t], -1.0, &wrk->Ku[t], 0, 0, &vu, 0, 1.0, p, 0, p, 0);
            blasfeo_dgemv_n(nx[t], rho[t], 1.0, &wrk->Ke[t], 0, 0, &ve, 0, 1.0, p, 0, p, 0);
        }
    }

    for (i32 tau=t-1; tau>=0; --tau) {
        // Compute B' p
        vu.pa = Mu + cnu[tau]; 
        blasfeo_dgemv_n(nu[tau], nx[tau], 1.0, &qp->Bt[tau], 0, 0, p, 0, 0.0, &vu, 0, &vu, 0);
        // Solve Lu [du; deta] = [B' p; 0]
        ve.pa = Me + crho[tau];
        TRSVLQR(vu, ve, wrk->Luu+tau, wrk->Lue+tau, wrk->Lee+tau, nu[tau], rho[tau]);
        // p = A[tau]' p - Ku du + Keta deta
        blasfeo_dgemv_n(nx[tau], nx[tau], 1.0, qp->At+tau, 0, 0, p, 0, 0.0, ptmp, 0, ptmp, 0);
        blasfeo_dgemv_n(nx[tau], nu[tau], -1.0, wrk->Ku+tau, 0, 0, &vu, 0, 1.0, ptmp, 0, ptmp, 0);
        blasfeo_dgemv_n(nx[tau], rho[tau], 1.0, wrk->Ke+tau, 0, 0, &ve, 0, 1.0, &ptmp, 0, &ptmp, 0);
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
    u32 N = qp->dims->N;
    u32* nu = qp->dims->nu;

    compute_M_row(wrk, qp, constr, n_active);
    
    // Write the new row of H into the new row of L.
    memset(wrk->Ld + n_active*wrk->W_stride, 0, (n_active+1)*sizeof(f64));
    f64* mu_ptr = wrk->Mu + n_active*wrk->nu_tot;
    f64* me_ptr = wrk->Me + n_active*wrk->neta;
    u32 nu_cols = wrk->cnu[wrk->as.max_t] + nu[wrk->as.max_t];
    u32 eta_cols = wrk->crho[wrk->as.max_t] + wrk->rho[wrk->as.max_t];
    daocp_fma_mv(wrk->Ld + n_active*wrk->W_stride,
           wrk->Mu, mu_ptr, n_active, nu_cols, wrk->nu_tot);
    daocp_fms_mv(wrk->Ld + n_active*wrk->W_stride,
           wrk->Me, me_ptr, n_active, eta_cols, wrk->neta);
    struct blasfeo_dvec vu, ve;
    vu.pa = mu_ptr; ve.pa = me_ptr;
    wrk->Ld[n_active*wrk->W_stride + n_active] = blasfeo_ddot(wrk->nu_tot, &vu, 0, &vu, 0)
                    - blasfeo_ddot(wrk->neta, &ve, 0, &ve, 0);
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
        wrk->as.max_t = MAX(wrk->as.max_t, t);
    else wrk->as.max_t = MAX(wrk->as.max_t, t-1);
}

void daocp_add_to_working_set(daocp_workspace* wrk, daocp_qp* qp, daocp_constraint* violated) {
    u32 t = violated->t;
    u32 idx = violated->idx;
    daocp_constraint_type type = violated->type;
    u32 is_upper = violated->is_upper;
    
    // Update cholesky factorization
    compute_hessian_row(wrk, qp, violated);
    update_dH_chol_add(wrk);

    // Update dual linear term
    f64 tmp;
    if (type == DAOCP_BOUND_X)
        tmp = (is_upper ? wrk->ubx_wrk : wrk->lbx_wrk)[wrk->cbx[t] + idx];
    else if (type == DAOCP_BOUND_U)
        tmp = (is_upper ? wrk->ubu_wrk : wrk->lbu_wrk)[wrk->cbu[t] + idx];
    else
        tmp = (is_upper ? wrk->ug_wrk : wrk->lg_wrk)[wrk->cg[t] + idx];
    wrk->dual_linear[wrk->as.n_active] = tmp;

    // Update working set data structures
    daocp_update_working_set__add(wrk, violated);
}

static update_cholesky_remove(daocp_workspace* wrk, u32 idx) {
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
        lii = wrk->Ld[i*W_stride+i];
        lii_new = sqrt(PW2(lii) + PW2(l[i-idx])); 
        wrk->Ld[i*W_stride+i] = lii_new;
        a = l[i-idx] / lii_new;
        b = lii / lii_new;
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

static daocp_update_working_set__remove(daocp_workspace* wrk, u32 xi_idx) {
    // Retrieve constraint info.
    u32 t = wrk->as.xi2con[xi_idx].t;
    u32 idx = wrk->as.xi2con[xi_idx].idx;
    daocp_constraint_type type = wrk->as.xi2con[xi_idx].type;
    u32 is_upper = wrk->as.xi2con[xi_idx].is_upper;

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
    
    // Recompute max_t
    wrk->as.max_t = 0;
    for (u32 i=0; i<wrk->as.n_active; ++i) {
        u32 mask = wrk->as.xi2con[i].type == DAOCP_BOUND_X || wrk->as.xi2con[i].type == DAOCP_ONLY_X;
        wrk->as.max_t = MAX(wrk->as.max_t, wrk->as.xi2con[i].t - mask*1);
    }
}

void daocp_remove_from_working_set(daocp_workspace* wrk, u32 xi_idx) {
    // Update cholesky of dH
    update_dH_chol_remove(wrk, xi_idx);

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
    if (dotv >= -ZERO_TOL && dotv <= ZERO_TOL) return 1;
    if (dotv > ZERO_TOL)
        daocp_negate(wrk->p, n_active);
}

u32 daocp_check_infeasibility_from_descent_dir(f64* p, u32* sign, u32 n) {
    for (u32 i=0; i<n; ++i)
        if ((sign[i] == 1 && p[i] < -ZERO_TOL) || (sign[i] == 0 && p[i] > ZERO_TOL)) return 0;

    return 1; 
}

u32 daocp_global_constraint_idx(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type) {
    u32 base = wrk->cbx[t]+wrk->cbu[t]+wrk->cg[t] + idx;
    u32 flag = type != DAOCP_BOUND_U;
    base += flag * wrk->dims->nbu[t];
    flag = flag && (type != DAOCP_BOUND_X);
    base += flag * wrk->dims->nbx[t];
    return base;
}

u32 daocp_is_active(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type) {
    return wrk->as.constraint_status[daocp_global_constraint_idx(wrk, t, idx, type)];
}

void daocp_change_status(daocp_workspace* wrk, u32 t, u32 idx, daocp_constraint_type type, u32 status) {
    wrk->as.constraint_status[daocp_global_constraint_idx(wrk, t, idx, type)];
}

void daocp_pointer_swap(unsigned char** p1, unsigned char** p2) {
    unsigned char* p3 = *p1;
    *p1 = *p2;
    *p2 = p3;
}