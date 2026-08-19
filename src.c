#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <blasfeo.h>
typedef uint32_t u32;
typedef double f64;
typedef int32_t i32;

#define ZERO_TOL 1e-12
#define SOLVED 1
#define INFEASIBLE 2
#define MAX_ITER 3
#define ILL_CONDITIONED 4
#define PW2(x) x*x
#define MAX(x, y) (x > y ? x : y)
#define MIN(x, y) (x < y ? x : y)
#define ABS(x) (x > 0 ? x : -(x))
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

typedef struct {
    i32 t;
    i32 idx;
    u32 is_state;
} constraint_t;

typedef struct {
    constraint_t* xi2con; // For each dual variable, the corresponding constraint.
    u32* as_members;
    u32 n_active;
    u32 max_t;
    struct blasfeo_dvec a;
} active_set;

typedef struct {
    active_set as;
    void (*constraint_selection_rule)(workspace*, constraint_t*);
    void* smemory;
    void* ineq_memory;
    void* eq_memory;
    f64* xi;
    f64* x;
    f64* u;
    f64* eta;
    f64* Dx;
    f64* Cu;
    f64* p;
    f64* dl;
    f64* r_wrk;
    f64* q_wrk;
    f64* Mu;
    f64* Meta;
    f64* L;
    f64* r;
    f64* q;
    f64* C;
    f64* D;
    f64* c;
    f64* d;
    f64* Ceq;
    f64* Deq;
    f64* ceq;
    f64* deq;
    f64* H;
    f64* h;
    f64* b;
    struct blasfeo_dmat* A;
    struct blasfeo_dmat* B;
    f64* w;
    struct blasfeo_dmat* Q;
    struct blasfeo_dmat* R;
    struct blasfeo_dmat* S;
    struct blasfeo_dmat* Luu;
    struct blasfeo_dmat* Lue;
    struct blasfeo_dmat* Lee;
    struct blasfeo_dmat* Ku;
    struct blasfeo_dmat* Ke;
    struct blasfeo_dmat* P;
    f64* Pw;
    f64* x_lqr;
    f64* u_lqr;
    f64* eta_lqr;
    f64* Dx_lqr;
    f64* Cu_lqr;
    struct blasfeo_dmat* riccati_tmp1;
    struct blasfeo_dmat* riccati_tmp2;
    f64* solver_tmp;
    f64* GEtmp;
    u32 return_status;
    u32 dH_singular;
    u32 nx;
    u32 nu;
    u32* mx;
    u32* cmx;
    u32* mu;
    u32* cmu;
    u32* eqx;
    u32* equ;
    u32* ceqx;
    u32* cequ;
    u32* rho;
    u32* crho;
    u32 neq_x0;
    u32 N;
    u32 nc;
    u32 W_stride;
    u32 neta;
    u32 max_iter;
} workspace;

u32 solve(workspace* wrk);
void update_problem_data(
    workspace* wrk, f64* x0, f64* q, f64* r, 
    f64* A, f64* B, f64* w, f64* Q, f64* R, 
    f64* S, u32* mx, u32* mu, f64* D, 
    f64* C, f64* d, f64* c, u32* eqx,
    u32* equ, f64* Deq, f64* Ceq, 
    f64* deq, f64* ceq) ;
void solve_dual_qp(workspace* wrk);
u32 get_descent_dir(workspace* wrk);
u32 drop_component(f64* xi, f64* p, u32 n);
void add_constraint(workspace* wrk, constraint_t* constr);
void remove_constraint(workspace* wrk, u32 idx);
void update_working_set_add(workspace* wrk, constraint_t* constr);
void update_working_set_remove(workspace* wrk, u32 idx);
void get_M_row(workspace* wrk, constraint_t* constr, u32 M_idx);
void get_dH_row(workspace* wrk, constraint_t* constr);
void update_dH_chol_add(workspace* wrk);
void update_dH_chol_remove(workspace* wrk, u32 idx);
u32 get_L_from_scratch(workspace* wrk);
void get_b(workspace* wrk);
void reset_working_set(workspace* wrk);
u32 is_dual_feasible(f64* p, u32 n);
void check_primal_feasibility_most_violated(workspace* wrk, constraint_t* constr);
void check_primal_feasibility_greedy(workspace* wrk, constraint_t* constr);
void compute_du_deta(workspace* wrk);
u32 check_constraints(workspace* wrk, u32 t, u32 is_state, constraint_t* constr);
void get_most_violated(workspace* wrk, constraint_t* constr);
u32 check_infeasibility(f64* p, u32 n);
void get_Cu_Dx(workspace* wrk);
u32 is_active(workspace* wrk, constraint_t* constr);
void set_active(workspace* wrk, constraint_t* constr);
void set_inactive(workspace* wrk, constraint_t* constr);
u32 get_ncx(workspace* wrk);
u32 get_ncu(workspace* wrk);
u32 get_neqx(workspace* wrk);
u32 get_nequ(workspace* wrk);
void compute_prefix_sum(u32* ca, u32* a, u32 n);
void workspace_init(
    workspace* wrk, f64* A, f64* B, 
    f64* w, f64* Q, f64* R, f64* S,
    f64* q, f64* r, f64* D, f64* C, 
    f64* d, f64* c, f64* Deq, f64* Ceq,
    f64* deq, f64* ceq, f64* x0, u32 N, 
    u32 nx, u32 nu, u32* mx, u32* mu, 
    u32* eqx, u32* equ, u32 max_iter,
    u32 greedy);
void allocate_inequalities_workspace(workspace* wrk, u32* mx, u32* mu);
void allocate_equalities_workspace(workspace* wrk, u32* eqx, u32* equ);
void workspace_free(workspace* wrk);
void solve_riccati(workspace* wrk);
void solve_lqr(workspace* wrk);
u32 eqcon_infeasible(workspace* wrk);
void add_lqrsol(workspace* wrk);
void fma_mv(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride);
void fms_mv(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride);
void fma_mv_t(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride);
void fms_mv_t(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride);
void trsv(f64* x, f64* L, u32 n, u32 stride);
void trsv_t(f64* x, f64* L, u32 n, u32 stride);
void transpose(f64* dst, f64* src, u32 nrs, u32 ncs);
void fma_mm_nt(f64* C, f64* A, f64* B, u32 nr, u32 nc, u32 k, u32 ostride);
void fms_mm_nt(f64* C, f64* A, f64* B, u32 nr, u32 nc, u32 k, u32 ostride);
void fma_mm_nn(f64* C, f64* A, f64* B, u32 nr, u32 nc, u32 k, u32 ostride);
void syrk_nt_lo(f64* C, f64* A, f64* B, u32 nrc, u32 k, u32 ostride);
void syrk_nt(f64* C, f64* A, f64* B, u32 nrc, u32 k, u32 ostride);
void nsyrk_nt(f64* C, f64* A, f64* B, u32 nrc, u32 k, u32 ostride);
void cholesky(f64* L, u32 n);
void trsm_rt(f64* X, f64* L, u32 nv, u32 nsys);
u32 gaussian_elimination(f64* A, f64* tmp, u32 nr, u32 nc, u32 nctot, u32 R);
void negate(f64* v, u32 n);
void add(f64* v, f64* w, u32 n);
f64 dot(f64* v, f64* w, u32 n);
void swap(f64** a, f64** b);

u32 solve(workspace* wrk) {
    constraint_t constr;
    i32 removed_constr;
    u32 iters = wrk->max_iter;

    // Check feasibility on x0
    if (eqcon_infeasible(wrk)) {
        wrk->return_status = INFEASIBLE;
        iters = 0;
        goto solve_ret;
    }

    for (u32 k=0; k<wrk->max_iter; ++k) {
        if (!wrk->dH_singular) {
            solve_dual_qp(wrk);
            if (is_dual_feasible(wrk->p, wrk->as.n_active)) {
                memcpy(wrk->xi, wrk->p, wrk->as.n_active*sizeof(f64));
                wrk->constraint_selection_rule(wrk, &constr);
                if (constr.t < 0) {
                    wrk->return_status = SOLVED;
                    iters = k;
                    goto solve_ret;
                }
                add_constraint(wrk, &constr); 
            } else {
                // Form descent direction
                for (u32 i=0; i<wrk->as.n_active; ++i) wrk->p[i] -= wrk->xi[i];
                removed_constr = drop_component(wrk->xi, wrk->p, wrk->as.n_active);
                remove_constraint(wrk, removed_constr);
            }
        } else {
            if (get_descent_dir(wrk)) {
                wrk->return_status = ILL_CONDITIONED;
                iters = k;
                goto solve_ret;
            }
            if (check_infeasibility(wrk->p, wrk->as.n_active)) {
                wrk->return_status = INFEASIBLE;
                iters = k;
                goto solve_ret;
            }
            removed_constr = drop_component(wrk->xi, wrk->p, wrk->as.n_active);
            remove_constraint(wrk, removed_constr);
        }
    }
    wrk->return_status = MAX_ITER;

solve_ret:
    // Add unconstrained minimizer
    add_lqrsol(wrk); 
    return iters;
}

void update_problem_data(
    workspace* wrk, f64* x0, f64* q, f64* r, 
    f64* A, f64* B, f64* w, f64* Q, f64* R, 
    f64* S, u32* mx, u32* mu, f64* D, 
    f64* C, f64* d, f64* c, u32* eqx,
    u32* equ, f64* Deq, f64* Ceq, 
    f64* deq, f64* ceq) 
{   
    /*
    For simplicity:
    - If the user wants to change any of A, B, w, it must provide again all three.
    - If the user wants to change any of Q, R, S, it must provide again all three.
    - If the user wants to change any of D, d, C, c, it must provide again all four, 
      toghether with mx, mu.
    - If the user wants to change any of Deq, deq, Ceq, ceq, it must provide again all four, 
      toghether with eqx, equ.
    */
    u32 N = wrk->N;
    u32 nx = wrk->nx;
    u32 nu = wrk->nu;
    u32 constraints_were_updated = D != 0 || C != 0;
    u32 equalities_were_updated = Deq != 0 || Ceq != 0;
    u32 dynamics_was_updated = A != 0;
    u32 cost_was_updated = Q != 0;

    memcpy(wrk->x_lqr, x0, nx*sizeof(f64));
    if (q) memcpy(wrk->q, q, N*nx*sizeof(f64));
    if (r) memcpy(wrk->r, r, N*nu*sizeof(f64));
    if (constraints_were_updated) {
        // Deallocate and reallocate workspace.
        free(wrk->ineq_memory);
        allocate_inequalities_workspace(wrk, mx, mu);

        memcpy(wrk->D, D, get_ncx(wrk)*nx*sizeof(f64));
        memcpy(wrk->C, C, get_ncu(wrk)*nu*sizeof(f64));
        memcpy(wrk->d, d, get_ncx(wrk)*sizeof(f64));
        memcpy(wrk->c, c, get_ncu(wrk)*sizeof(f64));

        // Previous working set is discarded.
        reset_working_set(wrk);
    }
    if (equalities_were_updated) {
        // Deallocate and reallocate workspace.
        free(wrk->eq_memory);
        allocate_equalities_workspace(wrk, eqx, equ);

        memcpy(wrk->Deq, Deq, get_neqx(wrk)*nx*sizeof(f64));
        memcpy(wrk->Ceq, Ceq, get_nequ(wrk)*nu*sizeof(f64));
        memcpy(wrk->deq, deq, get_neqx(wrk)*sizeof(f64));
        memcpy(wrk->ceq, ceq, get_nequ(wrk)*sizeof(f64));
    }
    if (cost_was_updated) {
        for (u32 t=0; t<N; ++t) {
            blasfeo_pack_tran_dmat(nx, nx, Q+t*nx*nx, nx, wrk->Q+t, 0, 0);
            blasfeo_pack_tran_dmat(nu, nu, R+t*nu*nu, nu, wrk->R+t, 0, 0);
            blasfeo_pack_dmat(nx, nu, S+t*nx*nu, nx, wrk->S+t, 0, 0);
        }
    }
    if (dynamics_was_updated) {
        for (u32 t=0; t<N; ++t) {
            blasfeo_pack_dmat(nx, nx, A+t*nx*nx, nx, wrk->A+t, 0, 0);    
            blasfeo_pack_dmat(nu, nx, B+t*nu*nx, nu, wrk->B+t, 0, 0);    
        }
        memcpy(wrk->w, w, wrk->N*wrk->nx*sizeof(f64));
    }

    if (dynamics_was_updated || cost_was_updated || equalities_were_updated)
        solve_riccati(wrk);
    
    // If constraints were not updated but dH changed, rebuild
    // from scratch. If new dH turns out to be singular, reset WS.
    u32 corrupt_workspace = wrk->dH_singular; // Corrupt workspace due to non "SOLVE" return status.
    u32 new_dH_singular = !wrk->dH_singular && !constraints_were_updated && 
        (cost_was_updated || dynamics_was_updated || equalities_were_updated)
        && get_L_from_scratch(wrk);
    if (corrupt_workspace || new_dH_singular) reset_working_set(wrk);

    memcpy(wrk->r_wrk, wrk->r, nu*N*sizeof(f64));
    memcpy(wrk->q_wrk, wrk->q, nx*N*sizeof(f64));
    solve_lqr(wrk);
    get_b(wrk);
}

void solve_dual_qp(workspace* wrk) {
    /*
        Solve linear system dH p = - d
    */
    u32 n_active = wrk->as.n_active;
    
    // copy d into p
    for (u32 i=0; i<n_active; ++i) wrk->p[i] = -wrk->dl[i];

    // Solve L y = -d
    trsv(wrk->p, wrk->L, n_active, wrk->W_stride);
    // Solve L'p = y
    trsv_t(wrk->p, wrk->L, n_active, wrk->W_stride);
}

u32 get_descent_dir(workspace* wrk) {
    u32 n_active = wrk->as.n_active;
    u32 W_stride = wrk->W_stride;

    // Solve LL'p = 0, p != 0. Assume L_{n_active, n_active} = 0.
    memset(wrk->p, 0, n_active*sizeof(f64));
    wrk->p[n_active-1] = 1.0;
    for (u32 i=0; i<n_active-1; ++i)
        wrk->p[i] = -wrk->L[(n_active-1)*W_stride + i];
    trsv_t(wrk->p, wrk->L, n_active-1, W_stride);

    // Enforce p' b < 0.
    f64 dotv = dot(wrk->p, wrk->dl, n_active);
    if (dotv >= -ZERO_TOL && dotv <= ZERO_TOL) return 1;
    if (dotv > ZERO_TOL)
        negate(wrk->p, n_active);
    return 0;
}

u32 drop_component(f64* xi, f64* p, u32 n) {
    /*
        Line search along p < 0:
            xi + t*p = 0 \iff
            t = -xi / p
    */
    f64 t = INFINITY;
    u32 argmin = 0;
    for (u32 i=0; i<n; ++i) {
        if (p[i] > -ZERO_TOL) continue;
        f64 tau = - xi[i] / p[i];
        if (tau < t) {
            t = tau;
            argmin = i;
        }
    }
    for (u32 i=0; i<n; ++i) xi[i] += t*p[i];
    return argmin;
}

void add_constraint(workspace* wrk, constraint_t* constr) {
    u32 t = constr->t;
    u32 idx = constr->idx;
    u32 is_state = constr->is_state;
    get_dH_row(wrk, constr);
    update_dH_chol_add(wrk);
    /*
        Update dual linear term
    */
    if (is_state) {
        wrk->dl[wrk->as.n_active] = 
            wrk->d[wrk->cmx[t] + idx] - wrk->Dx_lqr[wrk->cmx[t] + idx];
    } else {
        wrk->dl[wrk->as.n_active] = 
            wrk->c[wrk->cmu[t] + idx] - wrk->Cu_lqr[wrk->cmu[t] + idx];
    }

    update_working_set_add(wrk, constr);
}

void remove_constraint(workspace* wrk, u32 idx) {
    // Update cholesky of dH
    update_dH_chol_remove(wrk, idx);
    // Update linear term
    for (u32 i=idx+1; i<wrk->as.n_active; ++i)
        wrk->dl[i-1] = wrk->dl[i];
    // Update Mu and Meta
    for (u32 i=idx+1; i<wrk->as.n_active; ++i) {
        memcpy(wrk->Mu + (i-1)*wrk->N*wrk->nu,
               wrk->Mu + i*wrk->N*wrk->nu,
               wrk->N*wrk->nu*sizeof(f64));
        memcpy(wrk->Meta + (i-1)*wrk->neta,
               wrk->Meta + i*wrk->neta,
               wrk->neta*sizeof(f64));
    }

    // Update active set
    update_working_set_remove(wrk, idx);
}

void update_working_set_add(workspace* wrk, constraint_t* constr) {
    u32 t = constr->t;
    u32 idx = constr->idx;
    u32 is_state = constr->is_state;

    // Update ( xi_idx -> constraint ) map.
    constraint_t new_constraint = {(i32)t, (i32)idx, is_state};
    wrk->as.xi2con[wrk->as.n_active] = new_constraint;

    wrk->xi[wrk->as.n_active] = 0.0;
    set_active(wrk, &wrk->as.xi2con[wrk->as.n_active]);
    wrk->as.n_active += 1;
    wrk->as.max_t = MAX(wrk->as.max_t, t);
}

void update_working_set_remove(workspace* wrk, u32 xi_idx) {
    // Retrieve constraint info.
    u32 t = wrk->as.xi2con[xi_idx].t;
    u32 idx = wrk->as.xi2con[xi_idx].idx;
    u32 is_state = wrk->as.xi2con[xi_idx].is_state;

    // Update (xi_idx -> constraint info) map.
    for (u32 i=xi_idx+1; i<wrk->as.n_active; ++i) 
        wrk->as.xi2con[(i-1)] = wrk->as.xi2con[i]; 

    // Compact xi.
    for (u32 i=xi_idx+1; i < wrk->as.n_active; ++i)
        wrk->xi[i-1] = wrk->xi[i];
    constraint_t removed_constraint = {(i32)t, (i32)idx, is_state};
    set_inactive(wrk, &removed_constraint);
    wrk->as.n_active -= 1;
    // Recompute max_t
    wrk->as.max_t = 0;
    for (u32 i=0; i<wrk->as.n_active; ++i)
        wrk->as.max_t = MAX(wrk->as.max_t, wrk->as.xi2con[i].t);
}

void get_M_row(workspace* wrk, constraint_t* constr, u32 M_idx) {
    /*
    Mu and Meta contain the positive and negative signature components
    of the constraint square-root response. CU rows are given by the du, 
    deta recursion initialized with r[t] = C_{t,i}, r[tau!=t]=0, q=0, w=0.
    DX rows are given by the same recursion initialized with
    q[t] = D_{t,i}, q[tau!=t]=0, r=0, w=0.
    */
    u32 t = constr->t;
    u32 idx = constr->idx;
    f64* tmp1 = wrk->riccati_tmp1->pA;
    f64* tmp2 = wrk->riccati_tmp2->pA;
    u32 N = wrk->N;
    u32 nx = wrk->nx;
    u32 nu = wrk->nu;
    f64* Mu = wrk->Mu + N*nu*M_idx;
    f64* Meta = wrk->Meta + wrk->neta*M_idx;
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1;
    struct blasfeo_dvec v2;
    memset(Mu, 0, N*nu*sizeof(f64));
    memset(Meta, 0, wrk->neta*sizeof(f64));
    i32 tau;
    
    if (constr->is_state) { // Need DX
        // Initialize p
        memcpy(tmp1, wrk->D + wrk->cmx[t]*nx + idx*nx, nx*sizeof(f64));
        tau = t;
    } else { // Need CU
        tau = t-1;
        f64* u = Mu + t*nu;
        f64* eta = Meta + wrk->crho[t];
        u32 rho = wrk->rho[t];
        memcpy(u, wrk->C+wrk->cmu[t]*nu+idx*nu, nu*sizeof(f64));
        v0.pa = u; v2.pa = eta;
        TRSVLQR(v0, v2, wrk->Luu+t, wrk->Lue+t, wrk->Lee+t, nu, rho);
        
        // Initialize p.
        if (t>0) {
            v1.pa = tmp1;
            blasfeo_dgemv_n(nx, nu, -1.0, wrk->Ku+t, 0, 0, &v0, 0, 0.0, &v1, 0, &v1, 0);
            v0.pa = eta;
            blasfeo_dgemv_n(nx, rho, 1.0, wrk->Ke+t, 0, 0, &v0, 0, 1.0, &v1, 0, &v1, 0);
        }
    }

    for (; tau>=0; --tau) {
        u32 rho = wrk->rho[tau];
        f64* u = Mu + tau*nu;
        f64* eta = Meta + wrk->crho[tau];
        // Compute B' p
        v0.pa = u; v1.pa = tmp1;
        blasfeo_dgemv_n(nu, nx, 1.0, wrk->B+tau, 0, 0, &v1, 0, 0.0, &v0, 0, &v0, 0);
        // Solve Lu [du; deta] = [B' p; 0]
        v2.pa = eta;
        TRSVLQR(v0, v2, wrk->Luu+tau, wrk->Lue+tau, wrk->Lee+tau, nu, rho);
        // p = A[tau]' p - Ku du + Keta deta
        v0.pa = tmp2; v1.pa = tmp1;
        blasfeo_dgemv_n(nx, nx, 1.0, wrk->A+tau, 0, 0, &v1, 0, 0.0, &v0, 0, &v0, 0);
        v1.pa = u;
        blasfeo_dgemv_n(nx, nu, -1.0, wrk->Ku+tau, 0, 0, &v1, 0, 1.0, &v0, 0, &v0, 0);
        v1.pa = eta;
        blasfeo_dgemv_n(nx, rho, 1.0, wrk->Ke+tau, 0, 0, &v1, 0, 1.0, &v0, 0, &v0, 0);
        swap(&tmp1, &tmp2);
    }
}

void get_dH_row(workspace* wrk, constraint_t* constr) {
    /*
        dH = Mu Mu' - Meta Meta'.
        1) First compute the new rows of Mu and Meta.
        2) Compute their signed products with existing rows.
    */
    u32 n_active = wrk->as.n_active;
    u32 N = wrk->N;
    u32 nu = wrk->nu;

    get_M_row(wrk, constr, n_active);
    
    // Write the new row of dH into the new row of L.
    memset(wrk->L + n_active*wrk->W_stride, 0, (n_active+1)*sizeof(f64));
    f64* mu_ptr = wrk->Mu + n_active*N*nu;
    f64* meta_ptr = wrk->Meta + n_active*wrk->neta;
    u32 nu_cols = (wrk->as.max_t+1)*nu;
    u32 eta_cols = wrk->crho[wrk->as.max_t] + wrk->rho[wrk->as.max_t];
    fma_mv(wrk->L + n_active*wrk->W_stride,
           wrk->Mu, mu_ptr, n_active, nu_cols, N*nu);
    fms_mv(wrk->L + n_active*wrk->W_stride,
           wrk->Meta, meta_ptr, n_active, eta_cols, wrk->neta);
    wrk->L[n_active*wrk->W_stride + n_active] =
        dot(mu_ptr, mu_ptr, N*nu) - dot(meta_ptr, meta_ptr, wrk->neta);
}

void update_dH_chol_add(workspace* wrk) {
    u32 n_active = wrk->as.n_active;
    // Solve
    trsv(wrk->L + n_active*wrk->W_stride, wrk->L, n_active, wrk->W_stride);
    // Diagonal
    wrk->L[n_active*wrk->W_stride + n_active] -= 
            dot(wrk->L+n_active*wrk->W_stride, wrk->L+n_active*wrk->W_stride, n_active);
    if (wrk->L[n_active*wrk->W_stride + n_active] < ZERO_TOL) {
        wrk->dH_singular = 1;
        wrk->L[n_active*wrk->W_stride + n_active] = 0.0;
    }
    wrk->L[n_active*wrk->W_stride + n_active] = sqrt(wrk->L[n_active*wrk->W_stride + n_active]);
}

void update_dH_chol_remove(workspace* wrk, u32 idx) {
    u32 n_active = wrk->as.n_active;
    u32 W_stride = wrk->W_stride;
    f64* l = wrk->solver_tmp;

    // Remove row at idx.
    for (u32 i=idx+1; i<n_active; ++i)
        memcpy(wrk->L+(i-1)*W_stride, wrk->L+i*W_stride, (i+1)*sizeof(f64));

    // Extract column[idx] at l. Fix bottom-right lower triangle
    for (u32 i=idx; i<n_active-1; ++i) {
        l[i-idx] = wrk->L[i*W_stride + idx];
        for (u32 j=idx+1; j<=i+1; ++j)
            wrk->L[i*W_stride + j-1] = wrk->L[i*W_stride + j];
    }

    // Perform rank1 update of bottom-right lower triangle
    f64 lii, lii_new, a, b;
    for (u32 i=idx; i<n_active-1; ++i) {
        lii = wrk->L[i*W_stride+i];
        lii_new = sqrt(PW2(lii) + PW2(l[i-idx])); 
        wrk->L[i*W_stride+i] = lii_new;
        a = l[i-idx] / lii_new;
        b = lii / lii_new;
        for (u32 j=i+1; j<n_active-1; ++j) {
            lii = l[j-idx];
            lii_new = wrk->L[j*W_stride + i];
            l[j-idx] = lii * b - lii_new * a;
            wrk->L[j*W_stride + i] = b * lii_new + a * lii;
        }
    }

    wrk->dH_singular = 0;
}

u32 get_L_from_scratch(workspace* wrk) {
    u32 n_active = wrk->as.n_active;
    u32 W_stride = wrk->W_stride;

    // Compute Mu and Meta from scratch
    for (u32 ci=0; ci<n_active; ++ci) {
        constraint_t* constr = wrk->as.xi2con + ci;
        get_M_row(wrk, constr, ci);
    }

    // Compute dH = Mu Mu' - Meta Meta'
    memset(wrk->L, 0, n_active*W_stride*sizeof(f64));
    // Specialized syrk algorithm.
    u32 nu_cols = (wrk->as.max_t+1)*wrk->nu;
    u32 eta_cols = wrk->crho[wrk->as.max_t] + wrk->rho[wrk->as.max_t];
    for (u32 i=0; i<n_active; ++i)
        for (u32 j=0; j<=i; ++j) {
            for (u32 k=0; k<nu_cols; ++k)
                wrk->L[i*W_stride+j] +=
                    wrk->Mu[i*wrk->N*wrk->nu + k] * wrk->Mu[j*wrk->N*wrk->nu + k];
            for (u32 k=0; k<eta_cols; ++k)
                wrk->L[i*W_stride+j] -=
                    wrk->Meta[i*wrk->neta + k] * wrk->Meta[j*wrk->neta + k];
        }

    // Compute chol(dH), checking for singularity.
    for (u32 i=0; i<n_active; ++i) {
        f64* lii = wrk->L + i*W_stride + i;
        if (*lii < ZERO_TOL) return 1; // Detected singularity.
        *lii = sqrt(*lii);
        for (u32 j=i+1; j<n_active; ++j) wrk->L[j*W_stride + i] /= *lii;
        for (u32 j=i+1; j<n_active; ++j)
            for (u32 k=i+1; k<n_active; ++k)
                wrk->L[j*W_stride + k] -= wrk->L[j*W_stride + i] * wrk->L[k*W_stride + i];
    }
    return 0;
}

void get_b(workspace* wrk) {
    for (u32 i=0; i<wrk->as.n_active; ++i) {
        u32 t = wrk->as.xi2con[i].t;
        u32 idx = wrk->as.xi2con[i].idx;
        u32 is_state = wrk->as.xi2con[i].is_state;
        if (is_state) 
            wrk->dl[i] = wrk->d[wrk->cmx[t]+idx] - wrk->Dx_lqr[wrk->cmx[t]+idx];
        else
            wrk->dl[i] = wrk->c[wrk->cmu[t]+idx] - wrk->Cu_lqr[wrk->cmu[t]+idx];
    }
}

void reset_working_set(workspace* wrk) {
    for (u32 i=0; i<wrk->nc; ++i) wrk->as.as_members[i] = 0;
    wrk->as.n_active = 0;
    wrk->dH_singular = 0;
    wrk->as.max_t = 0;
}

u32 is_dual_feasible(f64* p, u32 n) {
    /*
        Check that all components are non-negative.   
    */
    for (u32 i=0; i<n; ++i) {
        if (p[i] < -ZERO_TOL) return 0;
    }
    return 1;
}

void check_primal_feasibility_most_violated(workspace* wrk, constraint_t* constr) {
    /*
        Computes argmin L(x, u, \lambda, \mu) and correspnding constraint slacks.
    */

    u32 N = wrk->N;
    u32 nu = wrk->nu;
    u32 nx = wrk->nx;
    f64* u = wrk->u;
    f64* eta = wrk->eta;
    f64* x = wrk->x;
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1;
    
    // Compute (du, deta) feedforwards.
    compute_du_deta(wrk);
    
    // Forward recursion.
    v0.pa = u; v1.pa = eta;
    blasfeo_dvecsc(nu, -1.0, &v0, 0);
    TRSVLQR_T(v0, v1, wrk->Luu, wrk->Lue, wrk->Lee, nu, wrk->rho[0]);
    v1.pa = x+nx;
    blasfeo_dgemv_t(nu, nx, 1.0, wrk->B, 0, 0, &v0, 0, 0.0, &v1, 0, &v1, 0); 
    for (u32 t=1; t<N; ++t) {
        u32 rho = wrk->rho[t];
        f64* u = wrk->u+t*nu;
        f64* eta = wrk->eta+wrk->crho[t];
        f64* x = wrk->x+t*nx;
        v0.pa = u; v1.pa = x;
        blasfeo_dgemv_t(nx, nu, -1.0, wrk->Ku+t, 0, 0, &v1, 0, -1.0, &v0, 0, &v0, 0);
        v0.pa = eta;
        blasfeo_dgemv_t(nx, rho, 1.0, wrk->Ke+t, 0, 0, &v1, 0, 1.0, &v0, 0, &v0, 0);
        v0.pa = u; v1.pa = eta;
        TRSVLQR_T(v0, v1, wrk->Luu+t, wrk->Lue+t, wrk->Lee+t, nu, rho);
        v1.pa = x+nx;
        blasfeo_dgemv_t(nu, nx, 1.0, wrk->B+t, 0, 0, &v0, 0, 0.0, &v1, 0, &v1, 0);
        v0.pa = x;
        blasfeo_dgemv_t(nx, nx, 1.0, wrk->A+t, 0, 0, &v0, 0, 1.0, &v1, 0, &v1, 0);
    }

    // Compute constraint image and get most violated constraint
    get_Cu_Dx(wrk);
    get_most_violated(wrk, constr);
}

void check_primal_feasibility_greedy(workspace* wrk, constraint_t* constr) {
    u32 N = wrk->N;
    u32 nx = wrk->nx;
    u32 nu = wrk->nu;

    // Compute feedforward terms
    compute_du_deta(wrk);

    // Run forward recursion and stop as soon as a violated constraint
    // is detected.
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1;
    constr->t = -1;

    // First iteration (x0 = 0)
    v0.pa = wrk->u; v1.pa = wrk->eta;
    blasfeo_dvecsc(nu, -1.0, &v0, 0);
    TRSVLQR_T(v0, v1, wrk->Luu, wrk->Lue, wrk->Lee, nu, wrk->rho[0]);
    if (check_constraints(wrk, 0, 0, constr)) return;
    v1.pa = wrk->x+nx;
    blasfeo_dgemv_t(nu, nx, 1.0, wrk->B, 0, 0, &v0, 0, 0.0, &v1, 0, &v1, 0);
    if (check_constraints(wrk, 0, 1, constr)) return;

    for (u32 t=1; t<N; ++t) {
        u32 rho = wrk->rho[t];
        f64* u = wrk->u+t*nu;
        f64* eta = wrk->eta+wrk->crho[t];
        f64* x = wrk->x+t*nx;
        
        // Compute control
        v0.pa = u; v1.pa = x;
        blasfeo_dgemv_t(nx, nu, -1.0, wrk->Ku+t, 0, 0, &v1, 0, -1.0, &v0, 0, &v0, 0);
        v0.pa = eta;
        blasfeo_dgemv_t(nx, rho, 1.0, wrk->Ke+t, 0, 0, &v1, 0, 1.0, &v0, 0, &v0, 0);
        v0.pa = u; v1.pa = eta;
        TRSVLQR_T(v0, v1, wrk->Luu+t, wrk->Lue+t, wrk->Lee+t, nu, rho);

        // Check control constraints
        if (check_constraints(wrk, t, 0, constr)) return; 

        // Propagate state dynamics
        v0.pa = x; v1.pa = x+nx;
        blasfeo_dgemv_t(nx, nx, 1.0, wrk->A+t, 0, 0, &v0, 0, 0.0, &v1, 0, &v1, 0);
        v0.pa = u;
        blasfeo_dgemv_t(nx, nu, 1.0, wrk->B+t, 0, 0, &v0, 0, 1.0, &v1, 0, &v1, 0);

        // Check state constraints
        if (check_constraints(wrk, t, 1, constr)) return;
    }
}

void compute_du_deta(workspace* wrk) {
    u32 N = wrk->N;
    u32 nu = wrk->nu;
    f64* u = wrk->u;
    f64* eta = wrk->eta;
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1;
    memset(u, 0, N*nu*sizeof(f64));
    memset(eta, 0, wrk->neta*sizeof(f64));
    // Get sqrt feedforwards (du, deta).
    u32 u_cols = (wrk->as.max_t+1)*nu;
    u32 eta_cols = wrk->crho[wrk->as.max_t] + wrk->rho[wrk->as.max_t];
    v0.pa = u;
    for (u32 i=0; i<wrk->as.n_active; ++i) {
        v1.pa = wrk->Mu + i*N*nu;
        blasfeo_daxpy(u_cols, wrk->xi[i], &v1, 0, &v0, 0, &v0, 0);
    }
    v0.pa = eta;
    for (u32 i=0; i<wrk->as.n_active; ++i) {
        v1.pa = wrk->Meta + i*wrk->neta;
        blasfeo_daxpy(eta_cols, wrk->xi[i], &v1, 0, &v0, 0, &v0, 0);
    }
}

u32 check_constraints(workspace* wrk, u32 t, u32 is_state, constraint_t* constr) {
    u32 m, cm, n;
    f64 *xu, *CD, *cd, *CuDx, *CuDx_lqr;
    if (is_state) {
        m = wrk->mx[t];
        cm = wrk->cmx[t];
        n = wrk->nx;
        xu = wrk->x+t*n+n;
        CuDx = wrk->Dx + cm;
        CuDx_lqr = wrk->Dx_lqr + cm;
        CD = wrk->D+cm*n;
        cd = wrk->d + cm;
    } else {
        m = wrk->mu[t];
        cm = wrk->cmu[t];
        n = wrk->nu;
        xu = wrk->u+t*n;
        CuDx = wrk->Cu+cm;
        CuDx_lqr = wrk->Cu_lqr+cm;
        CD = wrk->C+cm*n;
        cd = wrk->c+cm;
    }
    struct blasfeo_dvec v0, v1;
    v0.pa = xu;

    for (u32 i=0; i<m; ++i) {
        constraint_t tmp = (constraint_t) {(i32) t, (i32) i, is_state};
        if (!is_active(wrk, &tmp)) {
            v1.pa = CD+i*n;
            CuDx[i] = CuDx_lqr[i] + blasfeo_ddot(n, &v0, 0, &v1, 0);
            if (CuDx[i] - cd[i] > ZERO_TOL) {
                constr->t = t;
                constr->idx = i;
                constr->is_state = is_state;
                return 1;
            }
        }
    }
    
    return 0;
}

void get_most_violated(workspace* wrk, constraint_t* constr) {
    // Return most violated constraint
    i32 tu = -1;
    i32 tx = -1;
    i32 idxu = -1;
    i32 idxx = -1;
    f64 maxu = -1;
    f64 maxx = -1;

    /*
        Check Cu
    */
    for (u32 tau=0; tau<wrk->N; ++tau)
        for (u32 i=0; i<wrk->mu[tau]; ++i) {
            constraint_t tmpcon = (constraint_t) {(i32) tau, (i32) i, 0};
            if (is_active(wrk, &tmpcon)) continue;
            f64 tmp = wrk->Cu[wrk->cmu[tau] + i] + wrk->Cu_lqr[wrk->cmu[tau] + i] - wrk->c[wrk->cmu[tau] + i];
            if (tmp > ZERO_TOL && tmp > maxu) {
                maxu = tmp;
                idxu = i;
                tu = tau;
            }
    }
    
    /*
        Check Dx
    */
    for (u32 tau=0; tau<wrk->N; ++tau)
        for (u32 i=0; i<wrk->mx[tau]; ++i) {
            constraint_t tmpcon = (constraint_t){(i32) tau, (i32) i, 1};
            if (is_active(wrk, &tmpcon)) continue;
            f64 tmp = wrk->Dx[wrk->cmx[tau] + i] + wrk->Dx_lqr[wrk->cmx[tau] + i] - wrk->d[wrk->cmx[tau] + i];
            if (tmp > ZERO_TOL && tmp > maxx) {
                maxx = tmp;
                idxx = i;
                tx = tau;
            }
    }

    if (maxx > maxu) {
        constr->idx = idxx;
        constr->t = tx;
        constr->is_state = 1;
    } else {
        constr->idx = idxu;
        constr->t = tu;
        constr->is_state = 0;
    }
}

u32 check_infeasibility(f64* p, u32 n) {
    for (u32 i=0; i<n; ++i)
        if (p[i] < -ZERO_TOL) return 0;

    return 1; 
}

void get_Cu_Dx(workspace* wrk) {
    u32 N = wrk->N;
    u32 nx = wrk->nx;
    u32 nu = wrk->nu;
    u32* mx = wrk->mx;
    u32* cmx = wrk->cmx;
    u32* mu = wrk->mu;
    u32* cmu = wrk->cmu;
    f64* u = wrk->u;
    f64* x = wrk->x;
    f64* Cu = wrk->Cu;
    f64* Dx = wrk->Dx;
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1;

    for (u32 t=0; t<N; ++t) {
        for (u32 i=0; i<mu[t]; ++i) {
            constraint_t constraint = {(i32)t, (i32)i, 0};
            if (!is_active(wrk, &constraint)) {
                v0.pa = wrk->C + cmu[t]*nu + i*nu;
                v1.pa = u + t*nu;
                Cu[cmu[t]+i] = blasfeo_ddot(nu, &v0, 0, &v1, 0);
            }
        }
        for (u32 i=0; i<mx[t]; ++i) {
            constraint_t constraint = {(i32)t, (i32)i, 1};
            if (!is_active(wrk, &constraint))  {
                v0.pa = wrk->D + cmx[t]*nx + i*nx;
                v1.pa = x + t*nx + nx;
                Dx[cmx[t]+i] = blasfeo_ddot(nx, &v0, 0, &v1, 0); 
            }
        }
    }
}

u32 is_active(workspace* wrk, constraint_t* constr) {
    u32 t = constr->t;
    u32 i = constr->idx;
    u32 is_state = constr->is_state;
    return wrk->as.as_members[wrk->cmx[t]+wrk->cmu[t] + is_state*wrk->mu[t] + i];
}

void set_active(workspace* wrk, constraint_t* constr) {
    u32 t = constr->t;
    u32 i = constr->idx;
    u32 is_state = constr->is_state;
    wrk->as.as_members[wrk->cmx[t]+wrk->cmu[t] + is_state*wrk->mu[t] + i] = 1;
}

void set_inactive(workspace* wrk, constraint_t* constr) {
    u32 t = constr->t;
    u32 i = constr->idx;
    u32 is_state = constr->is_state;
    wrk->as.as_members[wrk->cmx[t]+wrk->cmu[t] + is_state*wrk->mu[t] + i] = 0;
}

u32 get_ncx(workspace* wrk) {
    return wrk->cmx[wrk->N-1]+wrk->mx[wrk->N-1];
}

u32 get_ncu(workspace* wrk) {
    return wrk->cmu[wrk->N-1]+wrk->mu[wrk->N-1];
}

u32 get_neqx(workspace* wrk) {
    return wrk->ceqx[wrk->N-1]+wrk->eqx[wrk->N-1];
}

u32 get_nequ(workspace* wrk) {
    return wrk->cequ[wrk->N-1]+wrk->equ[wrk->N-1];
}

void compute_prefix_sum(u32* ca, u32* a, u32 n) {
    u32 sum = 0;
    for (u32 i=0; i<n; ++i) {
        ca[i] = sum;
        sum += a[i];
    }
}

void workspace_init(
    workspace* wrk, f64* A, f64* B, 
    f64* w, f64* Q, f64* R, f64* S,
    f64* q, f64* r, f64* D, f64* C, 
    f64* d, f64* c, f64* Deq, f64* Ceq,
    f64* deq, f64* ceq, f64* x0, u32 N, 
    u32 nx, u32 nu, u32* mx, u32* mu, 
    u32* eqx, u32* equ, u32 max_iter,
    u32 greedy)
{   
    wrk->N = N;
    wrk->nx = nx;
    wrk->nu = nu;
    wrk->dH_singular = 0;
    wrk->max_iter = max_iter;
    wrk->constraint_selection_rule = greedy ? check_primal_feasibility_greedy : 
                                        check_primal_feasibility_most_violated;

    /*
        Allocate statically sized memory
    */
    u32 nfloats = 6*N*nx+2*nx + // q, q_wrk, x, x_lqr, w, Pw
                  7*N*nu;     // u, eta, u_lqr, eta_lqr, r, r_wrk, b
    u32 nints = 10*N; // mx, cmx, mu, cmu, eqx, equ, ceqx, cequ, rho, crho
    u32 nmat = 11*N + 2; // A, B, Q, R, S, P, Luu, Lue, Lee, Ku, Ke, tmp1, tmp2
    wrk->smemory = malloc(
        nfloats*sizeof(f64) +
        nmat*sizeof(struct blasfeo_dmat) +
        nints*sizeof(u32)
    );
    f64* mem = (f64*) wrk->smemory;
    wrk->w = mem; mem+=N*nx;
    wrk->Pw = mem; mem+=N*nx;
    wrk->x = mem; mem+=N*nx+nx;
    wrk->x_lqr = mem; mem+=N*nx+nx;
    wrk->u = mem; mem+=N*nu;
    wrk->eta = mem; mem+=N*nu;
    wrk->u_lqr = mem; mem+=N*nu;
    wrk->eta_lqr = mem; mem+=N*nu;
    wrk->q = mem; mem+=N*nx;
    wrk->q_wrk = mem; mem+=N*nx;
    wrk->r = mem; mem+=N*nu;
    wrk->r_wrk = mem; mem+=N*nu;
    wrk->b = mem; mem+=N*nu;
    unsigned char* u32mem = (unsigned char*) mem; 
    wrk->mx = (u32*) u32mem; u32mem+=N*sizeof(u32);
    wrk->cmx = (u32*) u32mem; u32mem+=N*sizeof(u32);
    wrk->mu = (u32*) u32mem; u32mem+=N*sizeof(u32);
    wrk->cmu = (u32*) u32mem; u32mem+=N*sizeof(u32);
    wrk->eqx = (u32*) u32mem; u32mem+=N*sizeof(u32);
    wrk->equ = (u32*) u32mem; u32mem+=N*sizeof(u32);
    wrk->ceqx = (u32*) u32mem; u32mem+=N*sizeof(u32);
    wrk->cequ = (u32*) u32mem; u32mem+=N*sizeof(u32);
    wrk->rho = (u32*) u32mem; u32mem+=N*sizeof(u32);
    wrk->crho = (u32*) u32mem; u32mem+=N*sizeof(u32);
    struct blasfeo_dmat* matmem = (struct blasfeo_dmat*) u32mem;
    wrk->A = matmem; matmem += N;
    wrk->B = matmem; matmem += N;
    wrk->Q = matmem; matmem += N;
    wrk->R = matmem; matmem += N;
    wrk->S = matmem; matmem += N;
    wrk->P = matmem; matmem += N;
    wrk->Luu = matmem; matmem += N;
    wrk->Lue = matmem; matmem += N;
    wrk->Lee = matmem; matmem += N;
    wrk->Ku = matmem; matmem += N;
    wrk->Ke = matmem; matmem += N;
    wrk->riccati_tmp1 = matmem++;
    wrk->riccati_tmp2 = matmem++;

    /*
        Allocate dynamically sized memory
    */
    allocate_inequalities_workspace(wrk, mx, mu);
    allocate_equalities_workspace(wrk, eqx, equ);
    
    u32 ncx = get_ncx(wrk);
    u32 ncu = get_ncu(wrk);
    u32 neqx = get_neqx(wrk);
    u32 nequ = get_nequ(wrk);
    for (u32 t=0; t<N; ++t) {
        // Store A'
        blasfeo_allocate_dmat(nx, nx, wrk->A+t);
        blasfeo_pack_dmat(nx, nx, A+t*nx*nx, nx, wrk->A+t, 0, 0);
        // Store B'
        blasfeo_allocate_dmat(nu, nx, wrk->B+t);
        blasfeo_pack_dmat(nu, nx, B+t*nx*nu, nu, wrk->B+t, 0, 0);
        
        blasfeo_allocate_dmat(nx, nx, wrk->Q+t);
        blasfeo_pack_tran_dmat(nx, nx, Q+t*nx*nx, nx, wrk->Q+t, 0, 0);
        blasfeo_allocate_dmat(nu, nu, wrk->R+t);
        blasfeo_pack_tran_dmat(nu, nu, R+t*nu*nu, nu, wrk->R+t, 0, 0);
        // Store S'
        blasfeo_allocate_dmat(nx, nu, wrk->S+t);
        blasfeo_pack_dmat(nx, nu, S+t*nx*nu, nx, wrk->S+t, 0, 0);

        blasfeo_allocate_dmat(nu, nu, wrk->Luu+t);
        blasfeo_allocate_dmat(nu, nu, wrk->Lue+t);
        blasfeo_allocate_dmat(nu, nu, wrk->Lee+t);
        blasfeo_allocate_dmat(nx, nu, wrk->Ku+t);
        blasfeo_allocate_dmat(nx, nu, wrk->Ke+t);
        blasfeo_allocate_dmat(nx, nx, wrk->P+t);
    }
    blasfeo_allocate_dmat(MAX(nx, nu), MAX(nx, nu), wrk->riccati_tmp1);
    blasfeo_allocate_dmat(MAX(nx, nu), MAX(nx, nu), wrk->riccati_tmp2);
    memcpy(wrk->w, w, N*nx*sizeof(f64));
    memcpy(wrk->q, q, N*nx*sizeof(f64));
    memcpy(wrk->r, r, N*nu*sizeof(f64));
    memcpy(wrk->D, D, ncx*nx*sizeof(f64));
    memcpy(wrk->C, C, ncu*nu*sizeof(f64));
    memcpy(wrk->d, d, ncx*sizeof(f64));
    memcpy(wrk->c, c, ncu*sizeof(f64));
    memcpy(wrk->Deq, Deq, neqx*nx*sizeof(f64));
    memcpy(wrk->Ceq, Ceq, nequ*nu*sizeof(f64));
    memcpy(wrk->deq, deq, neqx*sizeof(f64));
    memcpy(wrk->ceq, ceq, nequ*sizeof(f64));
    memset(wrk->x, 0, nx*sizeof(f64));
    memcpy(wrk->x_lqr, x0, nx*sizeof(f64));

    // Initialize Working Set.
    reset_working_set(wrk);

    // Initialize LQR/Riccati data
    solve_riccati(wrk);
    solve_lqr(wrk);
}

void allocate_inequalities_workspace(workspace* wrk, u32* mx, u32* mu) {
    u32 N = wrk->N;
    u32 nx = wrk->nx;
    u32 nu = wrk->nu;

    memcpy(wrk->mx, mx, N*sizeof(u32));
    memcpy(wrk->mu, mu, N*sizeof(u32));
    // Compute prexif sums of mx and mu.
    compute_prefix_sum(wrk->cmx, wrk->mx, N);
    compute_prefix_sum(wrk->cmu, wrk->mu, N);
    u32 ncx = get_ncx(wrk);
    u32 ncu = get_ncu(wrk);
    u32 nc = wrk->nc = ncx + ncu;
    u32 W_stride = wrk->W_stride = MIN(nc, N*nu+1);

    u32 tmp_size = W_stride;
    u32 nfloats = 2*nc +          // Dx, Cu, Dx_lqr, Cu_lqr
                3*W_stride +      // xi, p, dl
                ncx*nx+ncu*nu + // D, C
                ncx+ncu +       // d, c
                W_stride*W_stride + // L
                W_stride*2*N*nu +       // Mu, Meta
                tmp_size;       // solver_tmp
    wrk->ineq_memory = malloc(
        nfloats*sizeof(f64) +
        nc*sizeof(u32) +              // active_set.as_members
        W_stride*sizeof(constraint_t)       // active_set.xi2con
    );
    f64* mem = (f64*) wrk->ineq_memory;
    wrk->Dx = mem; mem+=ncx;
    wrk->Dx_lqr = mem; mem+=ncx;
    wrk->Cu = mem; mem+=ncu;
    wrk->Cu_lqr = mem; mem+=ncu;
    wrk->xi = mem; mem+=W_stride;
    wrk->p = mem; mem+=W_stride;
    wrk->dl = mem; mem+=W_stride;
    wrk->D = mem; mem+=ncx*nx;
    wrk->C = mem; mem+=ncu*nu;
    wrk->d = mem; mem+=ncx;
    wrk->c = mem; mem+=ncu;
    wrk->L = mem; mem+=W_stride*W_stride;
    wrk->Mu = mem; mem+=W_stride*N*nu;
    wrk->Meta = mem; mem+=W_stride*N*nu;
    wrk->solver_tmp = mem; mem+=tmp_size;
    unsigned char* vmem = (unsigned char*) mem;
    wrk->as.as_members = (u32*) vmem; vmem+=nc*sizeof(u32);
    wrk->as.xi2con = (constraint_t*) vmem; vmem+=W_stride*sizeof(constraint_t);
}

void allocate_equalities_workspace(workspace* wrk, u32* eqx, u32* equ) {
    u32 N = wrk->N;
    u32 nx = wrk->nx;
    u32 nu = wrk->nu;

    memcpy(wrk->eqx, eqx, N*sizeof(u32));
    memcpy(wrk->equ, equ, N*sizeof(u32));
    // Compute prexif sums of mx and mu.
    compute_prefix_sum(wrk->ceqx, wrk->eqx, N);
    compute_prefix_sum(wrk->cequ, wrk->equ, N);
    u32 neqx = get_neqx(wrk);
    u32 nequ = get_nequ(wrk);

    u32 nfloats = neqx*nx+nequ*nu + // Deq, Ceq
                neqx+nequ +         // deq, ceq
                (neqx+nequ)*(nx+1+nx+nu+1) + // H, GEtmp, h
                nx+nu+1; // last row of GEtmp 
    wrk->eq_memory = malloc(nfloats*sizeof(f64));
    f64* mem = (f64*) wrk->eq_memory;
    wrk->Deq = mem; mem+=neqx*nx;
    wrk->Ceq = mem; mem+=nequ*nu;
    wrk->deq = mem; mem+=neqx;
    wrk->ceq = mem; mem+=nequ;
    wrk->H = mem; mem+=(neqx+nequ)*nx;
    wrk->GEtmp = mem; mem+=(neqx+nequ+1)*(nx+nu+1);
    wrk->h = mem; mem+=neqx+nequ;
}

void workspace_free(workspace* wrk) {
    for (u32 t=0; t<wrk->N; ++t) {
        blasfeo_free_dmat(wrk->A+t);
        blasfeo_free_dmat(wrk->B+t);
        blasfeo_free_dmat(wrk->Q+t);
        blasfeo_free_dmat(wrk->S+t);
        blasfeo_free_dmat(wrk->R+t);
        blasfeo_free_dmat(wrk->P+t);
        blasfeo_free_dmat(wrk->Luu+t);
        blasfeo_free_dmat(wrk->Lue+t);
        blasfeo_free_dmat(wrk->Lee+t);
        blasfeo_free_dmat(wrk->Ku+t);
        blasfeo_free_dmat(wrk->Ke+t);
    }
    blasfeo_free_dmat(wrk->riccati_tmp1);
    blasfeo_free_dmat(wrk->riccati_tmp2);
    free(wrk->smemory);
    free(wrk->ineq_memory);
    free(wrk->eq_memory);
}

void solve_riccati(workspace* wrk) {
    u32 N = wrk->N;
    u32 nx = wrk->nx;
    u32 nu = wrk->nu;
    u32* eqx = wrk->eqx;
    u32* equ = wrk->equ;
    u32* ceqx = wrk->ceqx;
    u32* cequ = wrk->cequ;
    struct blasfeo_dmat* tmp1 = wrk->riccati_tmp1;
    struct blasfeo_dmat* tmp2 = wrk->riccati_tmp2;
    f64* GEtmp = wrk->GEtmp;
    blasfeo_dgecp(nx, nx, wrk->Q+N-1, 0, 0, wrk->P+N-1, 0, 0);

    u32 neq_x0 = 0;
    for (i32 t=N-1; t>=0; t--) {
        // Compute Lu00 = chol(R + B'PB)
        blasfeo_dgemm_nt(nu, nx, nx, 1.0, wrk->B+t, 0, 0, wrk->P+t, 0, 0, 0.0, tmp1, 0, 0, tmp1, 0, 0);
        blasfeo_dsyrk_dpotrf_ln(nu, nx, tmp1, 0, 0, wrk->B+t, 0, 0, wrk->R+t, 0, 0, wrk->Luu+t, 0, 0); 

        /* 
            Gaussian elimination to propagate constraints
        */
        
        // Form [C; DB; HB | 0; DA; HA | c; d-Dw; h-Hw]
        f64* A = tmp2->pA;
        blasfeo_unpack_dmat(nx, nx, wrk->A+t, 0, 0, A, nx);
        memset(GEtmp, 0, (nx+nu+1)*(equ[t]+eqx[t]+neq_x0)*sizeof(f64));
        for (u32 i=0; i<equ[t]; ++i) memcpy(GEtmp+i*(nx+nu+1), wrk->Ceq+(cequ[t]+i)*nu, nu*sizeof(f64));
        for (u32 i=0; i<equ[t]; ++i) GEtmp[i*(nx+nu+1)+nx+nu] = wrk->ceq[cequ[t]+i];
        fma_mm_nn(GEtmp+equ[t]*(nx+nu+1)+nu, wrk->Deq+ceqx[t]*nx, A, eqx[t], nx, nx, nx+nu+1);
        fma_mm_nn(GEtmp+(equ[t]+eqx[t])*(nx+nu+1)+nu, wrk->H, A, neq_x0, nx, nx, nx+nu+1);
        for (u32 i=0; i<eqx[t]; ++i) GEtmp[(equ[t]+i)*(nx+nu+1)+nx+nu] = wrk->deq[ceqx[t]+i];
        fms_mv(GEtmp+equ[t]*(nx+nu+1)+nx+nu, wrk->Deq+ceqx[t]*nx, wrk->w+t*nx, eqx[t], nx, nx+nu+1);
        f64* B = tmp2->pA;
        blasfeo_unpack_dmat(nu, nx, wrk->B+t, 0, 0, B, nu);
        fma_mm_nn(GEtmp+equ[t]*(nx+nu+1), wrk->Deq+ceqx[t]*nx, B, eqx[t], nu, nx, nx+nu+1);
        fma_mm_nn(GEtmp+(equ[t]+eqx[t])*(nx+nu+1), wrk->H, B, neq_x0, nu, nx, nx+nu+1);
        for (u32 i=0; i<neq_x0; ++i) GEtmp[(eqx[t]+equ[t]+i)*(nx+nu+1)+nx+nu] = wrk->h[i];
        fms_mv(GEtmp+(equ[t]+eqx[t])*(nx+nu+1)+nx+nu, wrk->H, wrk->w+t*nx, neq_x0, nx, nx+nu+1);
        // Gaussian elimination
        u32 rho = gaussian_elimination(GEtmp, GEtmp + (equ[t]+eqx[t]+neq_x0)*(nx+nu+1),
            equ[t]+eqx[t]+neq_x0, nu, nx+nu+1, nu);
        // Copy new H, h.
        for (u32 i=rho; i<equ[t]+eqx[t]+neq_x0; ++i) {
            memcpy(wrk->H + (i-rho)*nx, GEtmp + i*(nx+nu+1) + nu, nx*sizeof(f64));
            wrk->h[i-rho] = GEtmp[i*(nx+nu+1)+nx+nu];
        }
        // Store -b.
        for (u32 i=0; i<rho; ++i) wrk->b[t*nu+i] = -GEtmp[i*(nx+nu+1)+nx+nu];
        neq_x0 = neq_x0+equ[t]+eqx[t] - rho;
        wrk->rho[t] = rho;

        /* 
            Complete LDL of KKT matrix [(R+B'PB) G'; G 0]
        */
        blasfeo_pack_tran_dmat(nu, rho, GEtmp, nx+nu+1, wrk->Lue+t, 0, 0);
        blasfeo_dtrsm_rltn(rho, nu, 1.0, wrk->Luu+t, 0, 0, wrk->Lue+t, 0, 0, wrk->Lue+t, 0, 0);
        blasfeo_dgesc(rho, rho, 0.0, wrk->Lee+t, 0, 0);
        blasfeo_dsyrk_dpotrf_ln(rho, nu, wrk->Lue+t, 0, 0, wrk->Lue+t, 0, 0, wrk->Lee+t, 0, 0, wrk->Lee+t, 0, 0);

        /*
            Compute partial feedback gains.
        */

        // Compute K0 = (A'PB + S')Lu00^{-T}
        blasfeo_dgemm_nt(nx, nu, nx, 1.0, wrk->A+t, 0, 0, tmp1, 0, 0, 1.0, wrk->S+t, 0, 0, wrk->Ku+t, 0, 0);
        blasfeo_dtrsm_rltn(nx, nu, 1.0, wrk->Luu+t, 0, 0, wrk->Ku+t, 0, 0, wrk->Ku+t, 0, 0);
        // Compute K1 = (M' - K0 Lu01')Lu11^{-T}
        blasfeo_pack_dmat(nx, rho, GEtmp+nu, nx+nu+1, wrk->Ke+t, 0, 0);
        blasfeo_dgemm_nt(nx, rho, nu, -1.0, wrk->Ku+t, 0, 0, wrk->Lue+t, 0, 0, 1.0, wrk->Ke+t, 0, 0, wrk->Ke+t, 0, 0);
        blasfeo_dtrsm_rltn(nx, rho, 1.0, wrk->Lee+t, 0, 0, wrk->Ke+t, 0, 0, wrk->Ke+t, 0, 0);

        if (t==0) break;
        /*
            Update P = Q + A'PA - Ku Ku' + Keta Keta'
        */
        blasfeo_dgemm_nt(nx, nx, nx, 1.0, wrk->A+t, 0, 0, wrk->P+t, 0, 0, 0.0, tmp1, 0, 0, tmp1, 0, 0);
        blasfeo_dsyrk_ln(nx, nx, 1.0, tmp1, 0, 0, wrk->A+t, 0, 0, 1.0, wrk->Q+t-1, 0, 0, wrk->P+t-1, 0, 0);
        blasfeo_dsyrk_ln(nx, nu, -1.0, wrk->Ku+t, 0, 0, wrk->Ku+t, 0, 0, 1.0, wrk->P+t-1, 0, 0, wrk->P+t-1, 0, 0);
        blasfeo_dsyrk_ln(nx, rho, 1.0, wrk->Ke+t, 0, 0, wrk->Ke+t, 0, 0, 1.0, wrk->P+t-1, 0, 0, wrk->P+t-1, 0, 0);
        blasfeo_dtrtr_l(nx, wrk->P+t-1, 0, 0, wrk->P+t-1, 0, 0);
    }
    // Set number of affine constraints on x0.
    wrk->neq_x0 = neq_x0;
    // Compute prexif sum of equality constraints.
    compute_prefix_sum(wrk->crho, wrk->rho, N);
    wrk->neta = wrk->crho[N-1] + wrk->rho[N-1];
}

void solve_lqr(workspace* wrk) {
    u32 N = wrk->N;
    u32 nu = wrk->nu;
    u32 nx = wrk->nx;
    f64* tmp1 = wrk->riccati_tmp1->pA;
    f64* tmp2 = wrk->riccati_tmp2->pA;
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1;
    struct blasfeo_dvec v2;

    // Initialize costate
    memcpy(tmp1, wrk->q + (N-1)*nx, nx*sizeof(f64)); 
    memset(wrk->Cu_lqr, 0, get_ncu(wrk)*sizeof(f64));
    memset(wrk->Dx_lqr, 0, get_ncx(wrk)*sizeof(f64));

    // Backward recursion
    for (i32 t=N-1; t>=0; t--) {
        /*
            Compute du = Lu^{-1}(r + B'(p + Pw))
        */
        u32 rho = wrk->rho[t];
        f64* du = wrk->u_lqr + t*nu;
        f64* deta = wrk->eta_lqr + wrk->crho[t];
        
        v0.pa = tmp1; v1.pa = wrk->w+t*nx;
        blasfeo_dsymv_l(nx, 1.0, wrk->P+t, 0, 0, &v1, 0, 1.0, &v0, 0, &v0, 0);
        v1.pa = du; v2.pa = wrk->r+t*nu;
        blasfeo_dgemv_n(nu, nx, 1.0, wrk->B+t, 0, 0, &v0, 0, 1.0, &v2, 0, &v1, 0);
        memcpy(deta, wrk->b+t*nu, rho*sizeof(f64));
        v2.pa = deta;
        TRSVLQR(v1, v2, wrk->Luu+t, wrk->Lue+t, wrk->Lee+t, nu, rho);

        if (t==0) break;
        /*
            Compute p = A' (p + Pw) - Ku du + Keta deta + q 
        */
        v2.pa = tmp2; v1.pa = wrk->q+(t-1)*nx;
        blasfeo_dgemv_n(nx, nx, 1.0, wrk->A+t, 0, 0, &v0, 0, 1.0, &v1, 0, &v2, 0);
        v1.pa = du;
        blasfeo_dgemv_n(nx, nu, -1.0, wrk->Ku+t, 0, 0, &v1, 0, 1.0, &v2, 0, &v2, 0);
        v1.pa = deta;
        blasfeo_dgemv_n(nx, rho, 1.0, wrk->Ke+t, 0, 0, &v1, 0, 1.0, &v2, 0, &v2, 0);
        swap(&tmp1, &tmp2);
    }

    // Forward recursion
    for (u32 t=0; t<N; ++t) {
        u32 rho = wrk->rho[t];
        f64* u = wrk->u_lqr+t*nu;
        f64* eta = wrk->eta_lqr+wrk->crho[t];
        f64* x = wrk->x_lqr+t*nx;
        /*
            u = Lu^{-T}\Simga[-Ku' x - du; Keta' x0 + deta]
        */
        v0.pa = u; v1.pa = x;
        blasfeo_dgemv_t(nx, nu, 1.0, wrk->Ku+t, 0, 0, &v1, 0, 1.0, &v0, 0, &v0, 0);
        blasfeo_dvecsc(nu, -1.0, &v0, 0);
        v0.pa = eta;
        blasfeo_dgemv_t(nx, rho, 1.0, wrk->Ke+t, 0, 0, &v1, 0, 1.0, &v0, 0, &v0, 0);
        v0.pa = u; v1.pa = eta;
        TRSVLQR_T(v0, v1, wrk->Luu+t, wrk->Lue+t, wrk->Lee+t, nu, rho);
        
        /*
            x = Ax + Bu + w
        */
        v1.pa = x+nx; v2.pa = wrk->w+t*nx;
        blasfeo_dgemv_t(nu, nx, 1.0, wrk->B+t, 0, 0, &v0, 0, 1.0, &v2, 0, &v1, 0);
        v2.pa = x;
        blasfeo_dgemv_t(nx, nx, 1.0, wrk->A+t, 0, 0, &v2, 0, 1.0, &v1, 0, &v1, 0);
        
        /*
            Cu, Dx
        */
        fma_mv(wrk->Cu_lqr+wrk->cmu[t], wrk->C+wrk->cmu[t]*nu, u, wrk->mu[t], nu, nu);
        fma_mv(wrk->Dx_lqr+wrk->cmx[t], wrk->D+wrk->cmx[t]*nx, x+nx, wrk->mx[t], nx, nx);
    }
}

u32 eqcon_infeasible(workspace* wrk) {
    memset(wrk->GEtmp, 0, wrk->neq_x0*sizeof(f64));
    fma_mv(wrk->GEtmp, wrk->H, wrk->x_lqr, wrk->neq_x0, wrk->nx, wrk->nx);
    for (u32 i=0; i<wrk->neq_x0; ++i)
        if (ABS(wrk->GEtmp[i] - wrk->h[i]) > ZERO_TOL) return 1;
    return 0;
}

void add_lqrsol(workspace* wrk) {
    // Add unconstrained minimizer.
    u32 N = wrk->N;
    u32 nu = wrk->nu;
    u32 nx = wrk->nx;
    struct blasfeo_dvec v0, v1;

    v0.pa = wrk->u; v1.pa = wrk->u_lqr;
    blasfeo_daxpy(N*nu, 1.0, &v1, 0, &v0, 0, &v0, 0); 
    v0.pa = wrk->eta, v1.pa = wrk->eta_lqr;
    blasfeo_daxpy(wrk->neta, 1.0, &v1, 0, &v0, 0, &v0, 0); 
    v0.pa = wrk->x + nx; v1.pa = wrk->x_lqr + nx;
    blasfeo_daxpy(N*nx, 1.0, &v1, 0, &v0, 0, &v0, 0); 
}

void fma_mv(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride) {
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1; v1.pa = x;
    for (u32 i=0; i<ny; ++i) {
        v0.pa = A + i*stride;
        y[i] += blasfeo_ddot(nx, &v0, 0, &v1, 0);
    }
}

void fms_mv(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride) {
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1; v1.pa = x;
    for (u32 i=0; i<ny; ++i) {
        v0.pa = A + i*stride;
        y[i] -= blasfeo_ddot(nx, &v0, 0, &v1, 0);
    }
}

void fma_mv_t(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride) {
    for (u32 i=0; i<nx; ++i)
        for (u32 j=0; j<ny; ++j) y[j] += A[i*stride + j] * x[i];
}

void fms_mv_t(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride) {
    for (u32 i=0; i<nx; ++i)
        for (u32 j=0; j<ny; ++j) y[j] -= A[i*stride + j] * x[i];
}

void trsv(f64* x, f64* L, u32 n, u32 stride) {
    for (u32 i=0; i<n; ++i) {
        x[i] /= L[i*stride + i];
        for (u32 j=i+1; j<n; ++j) x[j] -= L[j*stride + i] * x[i];
    }
}

void trsv_t(f64* x, f64* L, u32 n, u32 stride) {
    for (i32 i=n-1; i>=0; --i) {
        x[i] /= L[i*stride + i];
        for (i32 j=i-1; j>=0; --j) x[j] -= L[i*stride + j] * x[i];
    }
}

void transpose(f64* dst, f64* src, u32 nrs, u32 ncs) {
    for (u32 i=0; i<nrs; ++i)
        for (u32 j=0; j<ncs; ++j) dst[j*nrs + i] = src[i*ncs + j];
}

void fma_mm_nt(f64* C, f64* A, f64* B, u32 nr, u32 nc, u32 k, u32 ostride) {
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1;
    for (u32 i=0; i<nr; ++i)
        for (u32 j=0; j<nc; ++j) {
            v0.pa = A + i*k; v1.pa = B + j*k;
            C[i*ostride + j] += blasfeo_ddot(k, &v0, 0, &v1, 0);
        }
}

void fms_mm_nt(f64* C, f64* A, f64* B, u32 nr, u32 nc, u32 k, u32 ostride) {
    struct blasfeo_dvec v0;
    struct blasfeo_dvec v1;
    for (u32 i=0; i<nr; ++i)
        for (u32 j=0; j<nc; ++j) {
            v0.pa = A + i*k; v1.pa = B + j*k;
            C[i*ostride + j] -= blasfeo_ddot(k, &v0, 0, &v1, 0);
        }
}

void fma_mm_nn(f64* C, f64* A, f64* B, u32 nr, u32 nc, u32 k, u32 ostride) {
    for (u32 i=0; i<nr; ++i)
        for (u32 j=0; j<nc; ++j)
            for (u32 l=0; l<k; ++l) 
                C[i*ostride + j] += A[i*k + l] * B[l*nc + j];
}

void syrk_nt_lo(f64* C, f64* A, f64* B, u32 nrc, u32 k, u32 ostride) {
    for (u32 i=0; i<nrc; ++i)
        for (u32 j=0; j<=i; ++j)
            for (u32 l=0; l<k; ++l) 
                C[i*ostride + j] += A[i*k + l] * B[j*k + l];
}

void syrk_nt(f64* C, f64* A, f64* B, u32 nrc, u32 k, u32 ostride) {
    for (u32 i=0; i<nrc; ++i)
        for (u32 j=0; j<=i; ++j) {
            for (u32 l=0; l<k; ++l) C[i*ostride + j] += A[i*k + l] * B[j*k + l];
            C[j*ostride + i] = C[i*ostride + j];
        }
}

void nsyrk_nt(f64* C, f64* A, f64* B, u32 nrc, u32 k, u32 ostride) {
    for (u32 i=0; i<nrc; ++i)
        for (u32 j=0; j<=i; ++j) {
            for (u32 l=0; l<k; ++l) C[i*ostride + j] -= A[i*k + l] * B[j*k + l];
            C[j*ostride + i] = C[i*ostride + j];
        }
}

void cholesky(f64* L, u32 n) {
    for (u32 i=0; i<n; ++i) {
        L[i*n + i] = sqrt(L[i*n + i]);
        for (u32 j=i+1; j<n; ++j) L[j*n + i] /= L[i*n + i];
        for (u32 j=i+1; j<n; ++j)
            for (u32 k=i+1; k<n; ++k) L[j*n + k] -= L[i + j*n] * L[i + k*n];
    }
}

void trsm_rt(f64* X, f64* L, u32 nv, u32 nsys) {
    for (u32 i=0; i<nv; ++i) {
        for (u32 j=0; j<nsys; ++j) X[j*nv + i] /= L[i*nv + i];
        for (u32 k=i+1; k<nv; ++k)
            for (u32 j=0; j<nsys; ++j) X[j*nv + k] -= L[k*nv + i] * X[j*nv + i];
    }
}

u32 gaussian_elimination(f64* A, f64* tmp, u32 nr, u32 nc, u32 nctot, u32 R) {
    u32 rho = 0;
    for (u32 i=0; i<nc; ++i) {
        if (rho == R) break;
        // Find pivot
        u32 pi = rho-1;
        f64 p = 0;
        for (u32 j=rho; j<nr; ++j)
            if (ABS(A[j*nctot+i]) > ABS(p)) {
                pi = j;
                p = A[j*nctot+i];
            }
        if (pi==rho-1) continue;

        // Swap rows pi and rho
        if (rho != pi) {
            memcpy(tmp, A+rho*nctot, nctot*sizeof(f64));
            memcpy(A+rho*nctot, A+pi*nctot, nctot*sizeof(f64));
            memcpy(A+pi*nctot, tmp, nctot*sizeof(f64));
        }

        // Perform elimination step
        for (u32 j=rho+1; j<nr; ++j) {
            f64 alpha = A[j*nctot + i] / p;
            for (u32 k=i; k<nctot; ++k) A[j*nctot + k] -= alpha * A[rho*nctot + k];
        }
        
        // Increase rank
        rho += 1;
    }
    return rho;
}

void negate(f64* v, u32 n) {
    for (u32 i=0; i<n; ++i) v[i] *= -1.0;
}

void add(f64* v, f64* w, u32 n) {
    for (u32 i=0; i<n; ++i) v[i] += w[i];
}

f64 dot(f64* v, f64* w, u32 n) {
    f64 acc = 0.0;
    for (u32 i=0; i<n; ++i) acc += v[i] * w[i];
    return acc;
}

void swap(f64** a, f64** b) {
    f64* tmp = *a;
    *a = *b;
    *b = tmp;
}
