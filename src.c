#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
typedef uint32_t u32;
typedef double f64;
typedef int32_t i32;

#define ZERO_TOL 1e-12
#define SOLVED 1
#define INFEASIBLE 2
#define MAX_ITER 3
#define ILL_CONDITIONED 4
#define PW2(x) x*x

typedef struct {
    i32 t;
    i32 idx;
    u32 is_state;
} constraint_t;

typedef struct {
    i32* active_u; // For each time-index: (xi index active at t, corresponding constraint index).
    i32* active_x;
    constraint_t* xi2con; // For each dual variable, the corresponding constraint.
    u32* as_members;
    u32 n_active;
} active_set;

typedef struct {
    active_set as;
    void* smemory;
    void* dmemory;
    f64* xi;
    f64* x;
    f64* u;
    f64* Dx;
    f64* Cu;
    f64* p;
    f64* b_wrk;
    f64* r_wrk;
    f64* q_wrk;
    f64* M;
    f64* L;
    f64* r;
    f64* q;
    f64* C;
    f64* D;
    f64* c;
    f64* d;
    f64* A;
    f64* B;
    f64* w;
    f64* Q;
    f64* R;
    f64* S;
    f64* Lu;
    f64* K;
    f64* P;
    f64* Acl;
    f64* Pw;
    f64* x_lqr;
    f64* u_lqr;
    f64* Dx_lqr;
    f64* Cu_lqr;
    f64* tmp1;
    f64* tmp2;
    u32 return_status;
    u32 dH_singular;
    u32 nx;
    u32 nu;
    u32* mx;
    u32* cmx;
    u32* mu;
    u32* cmu;
    u32 N;
    u32 nc;
    u32 max_iter;
} workspace;

u32 solve(workspace* wrk);
void update_problem_data(
    workspace* wrk, f64* x0, f64* q, f64* r, 
    f64* A, f64* B, f64* w, f64* Q, f64* R, 
    f64* S, u32* mx, u32* mu, f64* D, 
    f64* C, f64* d, f64* c) ;
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
void compute_slacks(workspace* wrk, constraint_t* constr);
void get_violated_constraint(workspace* wrk, constraint_t* constr);
u32 check_infeasibility(f64* p, u32 n);
void solve_lqr(workspace* wrk, f64* x, f64* u);
void get_Cu_Dx(workspace* wrk, u32 all, f64* x, f64* u, f64* Dx, f64* Cu);
void get_lqr_qr(workspace* wrk);
void get_dual_linear_terms(f64* ref, i32* map, f64* mat, f64* xi, u32 N, u32 n, u32* m, u32* cumm);
u32 is_active(workspace* wrk, constraint_t* constr);
void set_active(workspace* wrk, constraint_t* constr);
void set_inactive(workspace* wrk, constraint_t* constr);
u32 get_ncx(workspace* wrk);
u32 get_ncu(workspace* wrk);
void compute_prefix_sum(u32* ca, u32* a, u32 n);
void workspace_init(
    workspace* wrk, f64* A, f64* B, 
    f64* w, f64* Q, f64* R, f64* S,
    f64* q, f64* r, f64* D, f64* C, 
    f64* d, f64* c, f64* x0, u32 N, 
    u32 nx, u32 nu, u32* mx, u32* mu, 
    u32 max_iter);
void allocate_dynamic_workspace(workspace* wrk, u32* mx, u32* mu);
void workspace_free(workspace* wrk);
void solve_riccati(workspace* wrk);
void fma_mv(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride);
void fma_mv_t(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride);
void trsv(f64* x, f64* L, u32 n, u32 stride);
void trsv_t(f64* x, f64* L, u32 n, u32 stride);
void transpose(f64* dst, f64* src, u32 nrs, u32 ncs);
void fma_mm_nt(f64* C, f64* A, f64* B, u32 nr, u32 nc, u32 k, u32 ostride);
void cholesky(f64* L, u32 n);
void trsm(f64* X, f64* L, u32 nv, u32 nsys);
void trsm_t(f64* X, f64* L, u32 nv, u32 nsys);
void negate(f64* v, u32 n);
f64 dot(f64* v, f64* w, u32 n);
void swap(f64** a, f64** b);

u32 solve(workspace* wrk) {
    constraint_t constr;
    i32 removed_constr;

    for (u32 k=0; k<wrk->max_iter; ++k) {
        if (!wrk->dH_singular) {
            solve_dual_qp(wrk);
            if (is_dual_feasible(wrk->p, wrk->as.n_active)) {
                memcpy(wrk->xi, wrk->p, wrk->as.n_active*sizeof(f64));
                compute_slacks(wrk, &constr);
                if (constr.t < 0) {
                    wrk->return_status = SOLVED;
                    return k;
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
                return k;
            }
            if (check_infeasibility(wrk->p, wrk->as.n_active)) {
                wrk->return_status = INFEASIBLE;
                return k;
            }
            removed_constr = drop_component(wrk->xi, wrk->p, wrk->as.n_active);
            remove_constraint(wrk, removed_constr);
        }
    }
    
    wrk->return_status = MAX_ITER;
    return wrk->max_iter;
}

void update_problem_data(
    workspace* wrk, f64* x0, f64* q, f64* r, 
    f64* A, f64* B, f64* w, f64* Q, f64* R, 
    f64* S, u32* mx, u32* mu, f64* D, 
    f64* C, f64* d, f64* c) 
{   
    /*
    For simplicity:
    - If the user wants to change any of A, B, w, it must provide again all three.
    - If the user wants to change any of Q, R, S, it must provide again all three.
    - If the user wants to change any of D, d, C, c, it must provide again all four, 
      toghether with mx, mu.
    */
    u32 constraints_were_updated = D != 0;
    u32 dynamics_was_updated = A != 0;
    u32 cost_was_updated = Q != 0;

    memcpy(wrk->x, x0, wrk->nx*sizeof(f64));
    memcpy(wrk->x_lqr, x0, wrk->nx*sizeof(f64));
    if (q) memcpy(wrk->q, q, wrk->N*wrk->nx*sizeof(f64));
    if (r) memcpy(wrk->r, r, wrk->N*wrk->nu*sizeof(f64));
    if (constraints_were_updated) {
        // Deallocate and reallocate dynamic workspace.
        free(wrk->dmemory);
        allocate_dynamic_workspace(wrk, mx, mu);

        memcpy(wrk->D, D, get_ncx(wrk)*wrk->nx*sizeof(f64));
        memcpy(wrk->C, C, get_ncu(wrk)*wrk->nu*sizeof(f64));
        memcpy(wrk->d, d, get_ncx(wrk)*sizeof(f64));
        memcpy(wrk->c, c, get_ncu(wrk)*sizeof(f64));

        // Previous working set is discarded.
        reset_working_set(wrk);
    }
    if (cost_was_updated) {
        memcpy(wrk->Q, Q, wrk->N*wrk->nx*wrk->nx*sizeof(f64));
        memcpy(wrk->R, R, wrk->N*wrk->nu*wrk->nu*sizeof(f64));
        memcpy(wrk->S, S, wrk->N*wrk->nx*wrk->nu*sizeof(f64));
    }
    if (dynamics_was_updated) {
        memcpy(wrk->A, A, wrk->N*wrk->nx*wrk->nx*sizeof(f64));
        memcpy(wrk->B, B, wrk->N*wrk->nu*wrk->nx*sizeof(f64));
        memcpy(wrk->w, w, wrk->N*wrk->nx*sizeof(f64));
    }

    if (dynamics_was_updated || cost_was_updated) {
        memcpy(wrk->P, wrk->Q, wrk->N*wrk->nx*wrk->nx*sizeof(f64));
        solve_riccati(wrk);
    }
    
    // If constraints were not updated but dH changed, rebuild
    // from scratch. If new dH turns out to be singular, reset WS.ù
    u32 corrupt_workspace = wrk->dH_singular; // Corrupt workspace due to non "SOLVE" return status.
    u32 new_dH_singular = !wrk->dH_singular && !constraints_were_updated && (cost_was_updated || dynamics_was_updated) && get_L_from_scratch(wrk);
    if (corrupt_workspace || new_dH_singular) reset_working_set(wrk);

    memcpy(wrk->r_wrk, wrk->r, wrk->nu*wrk->N*sizeof(f64));
    memcpy(wrk->q_wrk, wrk->q, wrk->nx*wrk->N*sizeof(f64));
    solve_lqr(wrk, wrk->x_lqr, wrk->u_lqr);
    get_Cu_Dx(wrk, 1, wrk->x_lqr, wrk->u_lqr, wrk->Dx_lqr, wrk->Cu_lqr);
    get_b(wrk);
}

void solve_dual_qp(workspace* wrk) {
    /*
        Solve linear system dH p = - d
    */
    u32 n_active = wrk->as.n_active;
    
    // copy d into p
    for (u32 i=0; i<n_active; ++i) wrk->p[i] = -wrk->b_wrk[i];

    // Solve L y = -d
    trsv(wrk->p, wrk->L, n_active, wrk->nc);
    // Solve L'p = y
    trsv_t(wrk->p, wrk->L, n_active, wrk->nc);
}

u32 get_descent_dir(workspace* wrk) {
    u32 n_active = wrk->as.n_active;
    u32 nc = wrk->nc;

    // Solve LL'p = 0, p != 0. Assume L_{n_active, n_active} = 0.
    memset(wrk->p, 0, n_active*sizeof(f64));
    wrk->p[n_active-1] = 1.0;
    for (u32 i=0; i<n_active-1; ++i)
        wrk->p[i] = -wrk->L[(n_active-1)*nc + i];
    trsv_t(wrk->p, wrk->L, n_active-1, nc);

    // Enforce p' b < 0.
    f64 dotv = dot(wrk->p, wrk->b_wrk, n_active);
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
        wrk->b_wrk[wrk->as.n_active] = 
            wrk->d[wrk->cmx[t] + idx] - wrk->Dx_lqr[wrk->cmx[t] + idx];
    } else {
        wrk->b_wrk[wrk->as.n_active] = 
            wrk->c[wrk->cmu[t] + idx] - wrk->Cu_lqr[wrk->cmu[t] + idx];
    }

    update_working_set_add(wrk, constr);
}

void remove_constraint(workspace* wrk, u32 idx) {
    // Update cholesky of dH
    update_dH_chol_remove(wrk, idx);
    // Update linear term
    for (u32 i=idx+1; i<wrk->as.n_active; ++i)
        wrk->b_wrk[i-1] = wrk->b_wrk[i];
    // Update M
    for (u32 i=idx+1; i<wrk->as.n_active; ++i)
        memcpy(wrk->M + (i-1)*wrk->N*wrk->nu, wrk->M + i*wrk->N*wrk->nu, wrk->N*wrk->nu*sizeof(f64));

    // Update active set
    update_working_set_remove(wrk, idx);
}

void update_working_set_add(workspace* wrk, constraint_t* constr) {
    u32 t = constr->t;
    u32 idx = constr->idx;
    u32 is_state = constr->is_state;

    // Update ( t -> active constraints ) map.
    if (is_state) {
        u32 i=0;
        for (; i<2*wrk->mx[t] && wrk->as.active_x[2*wrk->cmx[t] + i] >= 0; i+=2);
        wrk->as.active_x[2*wrk->cmx[t] + i] = wrk->as.n_active;
        wrk->as.active_x[2*wrk->cmx[t] + i + 1] = idx;
    } else {
        u32 i=0;
        for (; i<2*wrk->mu[t] && wrk->as.active_u[2*wrk->cmu[t] + i] >= 0; i+=2);
        wrk->as.active_u[2*wrk->cmu[t] + i] = wrk->as.n_active;
        wrk->as.active_u[2*wrk->cmu[t] + i + 1] = idx;
    }

    // Update ( xi_idx -> constraint ) map.
    constraint_t new_constraint = {(i32)t, (i32)idx, is_state};
    wrk->as.xi2con[wrk->as.n_active] = new_constraint;

    wrk->xi[wrk->as.n_active] = 0.0;
    set_active(wrk, &wrk->as.xi2con[wrk->as.n_active]);
    wrk->as.n_active += 1;
}

void update_working_set_remove(workspace* wrk, u32 xi_idx) {
    // Retrieve constraint info.
    u32 t = wrk->as.xi2con[xi_idx].t;
    u32 idx = wrk->as.xi2con[xi_idx].idx;
    u32 is_state = wrk->as.xi2con[xi_idx].is_state;

    // Update (xi_idx -> constraint info) map.
    for (u32 i=xi_idx+1; i<wrk->as.n_active; ++i) 
        wrk->as.xi2con[(i-1)] = wrk->as.xi2con[i]; 

    /* 
        Update (t -> active constraints) map.
    */
    // Remove xi_idx entry from active_x/u.
    i32* map = is_state ? wrk->as.active_x : wrk->as.active_u;
    u32* nc = is_state ? wrk->mx : wrk->mu;
    u32* cumc = is_state ? wrk->cmx : wrk->cmu;
    map += 2*cumc[t];
    u32 i = 0;
    for (i=0; i<2*nc[t] && map[i] != xi_idx; i+=2) ;
    for (; i<2*(nc[t]-1) && map[i] != -1; i+=2) {
        map[i] = map[i+2];
        map[i+1] = map[i+3];
    }
    map[2*(nc[t]-1)] = map[2*nc[t]-1] = -1;

    // Update active_x/u to reflect new xi indexing;
    nc = wrk->mu;
    cumc = wrk->cmu;
    map = wrk->as.active_u;
    for (u32 t=0; t<wrk->N; ++t)
        for (u32 i=0; i<2*nc[t] && map[2*cumc[t] + i] >= 0; i+=2) 
            if (map[2*cumc[t] + i] > xi_idx) map[2*cumc[t] + i] -= 1;
    nc = wrk->mx;
    cumc = wrk->cmx;
    map = wrk->as.active_x;
    for (u32 t=0; t<wrk->N; ++t)
        for (u32 i=0; i<2*nc[t] && map[2*cumc[t] + i] >= 0; i+=2) 
            if (map[2*cumc[t] + i] > xi_idx) map[2*cumc[t] + i] -= 1;

    // Compact xi.
    for (u32 i=xi_idx+1; i < wrk->as.n_active; ++i)
        wrk->xi[i-1] = wrk->xi[i];
    constraint_t removed_constraint = {(i32)t, (i32)idx, is_state};
    set_inactive(wrk, &removed_constraint);
    wrk->as.n_active -= 1;
}

void get_M_row(workspace* wrk, constraint_t* constr, u32 M_idx) {
    /*
        M = [CU; DX]
        where:
            - CU[t, k] = C Lu[t]^{-T} if t=k, -C K[t]Acl[t-1] .. Acl[k] B if k < t.
            - DX[t, k] = D Acl[t] ... Acl[k] B if k <= t.
    */
    u32 t = constr->t;
    u32 idx = constr->idx;
    f64* tmp1 = wrk->tmp1;
    f64* tmp2 = wrk->tmp2;
    u32 N = wrk->N;
    u32 nx = wrk->nx;
    u32 nu = wrk->nu;
    memset(wrk->M + N*nu*M_idx, 0, N*nu*sizeof(f64));
    f64* m_ptr = wrk->M + N*nu*M_idx + t*nu;
    
    if (constr->is_state) { // Need DX
        // Initialize recursion state
        memcpy(tmp1, wrk->D + wrk->cmx[t]*nx + idx*nx, nx*sizeof(f64));
        
        // Compute B' Di
        memset(m_ptr, 0, nu*sizeof(f64));
        fma_mv_t(m_ptr, wrk->B + t*nx*nu, tmp1, nu, nx, nu);
        // Solve Lu x = B' Di
        trsv(m_ptr, wrk->Lu + t*nu*nu, nu, nu);
        // walk back along row of M.
        m_ptr -= nu;

        // Propagate through Acl recursion state
        for (i32 tau=t-1; tau>=0; --tau) {
            // state = Acl[tau]' state
            memset(tmp2, 0, nx*sizeof(f64));
            fma_mv_t(tmp2, wrk->Acl+(tau+1)*nx*nx, tmp1, nx, nx, nx);
            swap(&tmp1, &tmp2);

            // Compute B' state
            memset(m_ptr, 0, nu*sizeof(f64));
            fma_mv_t(m_ptr, wrk->B+tau*nx*nu, tmp1, nu, nx, nu);
            // Solve Lu x = B' state
            trsv(m_ptr, wrk->Lu+tau*nu*nu, nu, nu);
            // walk back along row of M.
            m_ptr -= nu;            
        }
    } else { // Need DU
        memcpy(m_ptr, wrk->C + wrk->cmu[t]*nu + idx*nu, nu*sizeof(f64));
        trsv(m_ptr, wrk->Lu+nu*nu*t, nu, nu);
        
        // Initialize backward recursion state.
        if (t>0) {
            memset(tmp1, 0, nx*sizeof(f64));
            fma_mv_t(tmp1, wrk->K+t*nu*nx, wrk->C+wrk->cmu[t]*nu+idx*nu, nx, nu, nx);
            negate(tmp1, nx);
        }
        // Walk back along row of M.
        m_ptr -= nu;
        
        // Start recursion
        for (i32 tau=t-1; tau>=0; --tau) {
            // Compute B' state
            memset(m_ptr, 0, nu*sizeof(f64));
            fma_mv_t(m_ptr, wrk->B+tau*nx*nu, tmp1, nu, nx, nu);
            // Solve Lu x = B' state
            trsv(m_ptr, wrk->Lu+tau*nu*nu, nu, nu);
            // walk back along row of M.
            m_ptr -= nu;  

            // state = Acl[tau]' state
            memset(tmp2, 0, nx*sizeof(f64));
            fma_mv_t(tmp2, wrk->Acl+tau*nx*nx, tmp1, nx, nx, nx);
            swap(&tmp1, &tmp2);          
        }
    }
}

void get_dH_row(workspace* wrk, constraint_t* constr) {
    /*
        dH = M M', M = [CU; DX]
        1) First compute m, new row of M.
        2) We then compute it's product with M.
    */
    u32 n_active = wrk->as.n_active;
    u32 N = wrk->N;
    u32 nu = wrk->nu;

    get_M_row(wrk, constr, n_active);
    
    // Computation of new row of dH, given by (M' m, m'm).
    // Write result into new row of L.
    memset(wrk->L + n_active*wrk->nc, 0, (n_active+1)*sizeof(f64));
    f64* m_ptr = wrk->M + n_active*N*nu;
    fma_mv(wrk->L + n_active*wrk->nc, wrk->M, m_ptr, n_active, N*nu, N*nu);
    wrk->L[n_active*wrk->nc + n_active] = dot(m_ptr, m_ptr, N*nu);
}

void update_dH_chol_add(workspace* wrk) {
    u32 n_active = wrk->as.n_active;
    // Solve
    trsv(wrk->L + n_active*wrk->nc, wrk->L, n_active, wrk->nc);
    // Diagonal
    wrk->L[n_active*wrk->nc + n_active] -= 
            dot(wrk->L+n_active*wrk->nc, wrk->L+n_active*wrk->nc, n_active);
    if (wrk->L[n_active*wrk->nc + n_active] < ZERO_TOL) {
        wrk->dH_singular = 1;
        wrk->L[n_active*wrk->nc + n_active] = 0.0;
    }
    wrk->L[n_active*wrk->nc + n_active] = sqrt(wrk->L[n_active*wrk->nc + n_active]);
}

void update_dH_chol_remove(workspace* wrk, u32 idx) {
    u32 n_active = wrk->as.n_active;
    u32 nc = wrk->nc;
    f64* l = wrk->tmp1;

    // Remove row at idx.
    for (u32 i=idx+1; i<n_active; ++i)
        memcpy(wrk->L+(i-1)*nc, wrk->L+i*nc, (i+1)*sizeof(f64));

    // Extract column[idx] at l. Fix bottom-right lower triangle
    for (u32 i=idx; i<n_active-1; ++i) {
        l[i-idx] = wrk->L[i*nc + idx];
        for (u32 j=idx+1; j<=i+1; ++j)
            wrk->L[i*nc + j-1] = wrk->L[i*nc + j];
    }

    // Perform rank1 update of bottom-right lower triangle
    f64 lii, lii_new, a, b;
    for (u32 i=idx; i<n_active-1; ++i) {
        lii = wrk->L[i*nc+i];
        lii_new = sqrt(PW2(lii) + PW2(l[i-idx])); 
        wrk->L[i*nc+i] = lii_new;
        a = l[i-idx] / lii_new;
        b = lii / lii_new;
        for (u32 j=i+1; j<n_active-1; ++j) {
            lii = l[j-idx];
            lii_new = wrk->L[j*nc + i];
            l[j-idx] = lii * b - lii_new * a;
            wrk->L[j*nc + i] = b * lii_new + a * lii;
        }
    }

    wrk->dH_singular = 0;
}

u32 get_L_from_scratch(workspace* wrk) {
    u32 n_active = wrk->as.n_active;
    u32 nc = wrk->nc;

    // Compute M from scratch
    for (u32 ci=0; ci<n_active; ++ci) {
        constraint_t* constr = wrk->as.xi2con + ci;
        get_M_row(wrk, constr, ci);
    }

    // Compute dH = MM'
    memset(wrk->L, 0, n_active*wrk->nc*sizeof(f64));
    fma_mm_nt(wrk->L, wrk->M, wrk->M, n_active, n_active, wrk->N*wrk->nu, nc);

    // Compute chol(dH), checking for singularity.
    for (u32 i=0; i<n_active; ++i) {
        f64* lii = wrk->L + i*nc + i;
        if (*lii < ZERO_TOL) return 1; // Detected singularity.
        *lii = sqrt(*lii);
        for (u32 j=i+1; j<n_active; ++j) wrk->L[j*nc + i] /= *lii;
        for (u32 j=i+1; j<n_active; ++j)
            for (u32 k=i+1; k<n_active; ++k)
                wrk->L[j*nc + k] -= wrk->L[j*nc + i] * wrk->L[k*nc + i];
    }
    return 0;
}

void get_b(workspace* wrk) {
    for (u32 i=0; i<wrk->as.n_active; ++i) {
        u32 t = wrk->as.xi2con[i].t;
        u32 idx = wrk->as.xi2con[i].idx;
        u32 is_state = wrk->as.xi2con[i].is_state;
        if (is_state) 
            wrk->b_wrk[i] = wrk->d[wrk->cmx[t]+idx] - wrk->Dx_lqr[wrk->cmx[t]+idx];
        else
            wrk->b_wrk[i] = wrk->c[wrk->cmu[t]+idx] - wrk->Cu_lqr[wrk->cmu[t]+idx];
    }
}

void reset_working_set(workspace* wrk) {
    for (u32 i=0; i<2*get_ncx(wrk); ++i) wrk->as.active_x[i] = -1;
    for (u32 i=0; i<2*get_ncu(wrk); ++i) wrk->as.active_u[i] = -1;
    for (u32 i=0; i<wrk->nc; ++i) wrk->as.as_members[i] = 0;
    wrk->as.n_active = 0;
    wrk->dH_singular = 0;
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

void compute_slacks(workspace* wrk, constraint_t* constr) {
    /*
        Computes argmin L(x, u, \lambda, \mu) and correspnding constraint slacks.
    */

    get_lqr_qr(wrk);
    solve_lqr(wrk, wrk->x, wrk->u);
    get_Cu_Dx(wrk, 0, wrk->x, wrk->u, wrk->Dx, wrk->Cu);
    get_violated_constraint(wrk, constr);
}

void get_violated_constraint(workspace* wrk, constraint_t* constr) {
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
            f64 tmp = wrk->Cu[wrk->cmu[tau] + i] - wrk->c[wrk->cmu[tau] + i];
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
            f64 tmp = wrk->Dx[wrk->cmx[tau] + i] - wrk->d[wrk->cmx[tau] + i];
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

void solve_lqr(workspace* wrk, f64* x, f64* u) {
    u32 N = wrk->N;
    u32 nu = wrk->nu;
    u32 nx = wrk->nx;
    f64* tmp1 = wrk->tmp1;
    f64* tmp2 = wrk->tmp2;

    // Initialize states with disturbances
    memcpy(x + nx, wrk->w, N*nx*sizeof(f64));
    memcpy(u, wrk->r_wrk, N*nu*sizeof(f64));
    // Initialize costate
    memcpy(tmp1, wrk->q_wrk + (N-1)*nx, nx*sizeof(f64)); 

    // Backward recursion
    for (i32 t=N-1; t>=0; t--) {
        /*
            Compute u = - (R + B'PB)^{-1}(r + B'(p + Pw))
        */
        for (u32 i=0; i<nx; ++i) tmp1[i] += wrk->Pw[t*nx + i];
        fma_mv_t(u+t*nu, wrk->B+t*nu*nx, tmp1, nu, nx, nu);
        negate(u + t*nu, nu);
        trsv(u+t*nu, wrk->Lu+t*nu*nu, nu, nu);
        trsv_t(u+t*nu, wrk->Lu+t*nu*nu, nu, nu);

        if (t==0) break;
        /*
            Compute p = (A - BK)' (p + Pw) - K'r + q 
        */
        for (u32 i=0; i<nx; ++i) tmp2[i] = wrk->q_wrk[(t-1)*nx + i];
        fma_mv_t(tmp2, wrk->Acl+t*nx*nx, tmp1, nx, nx, nx);
        swap(&tmp1, &tmp2);
        for (u32 j=0; j<nu; ++j) 
            for (u32 i=0; i<nx; ++i) tmp1[i] -= wrk->K[t*nx*nu + j*nx + i] * wrk->r_wrk[t*nu + j];
    }

    // Forward recursion
    for (u32 t=0; t<N; ++t) {
        /*
            u = -K x + du
        */
        for (u32 i=0; i<nu; ++i)
            for (u32 j=0; j<nx; ++j) u[t*nu + i] -= wrk->K[t*nu*nx + i*nx + j] * x[t*nx + j];
        
        /*
            x = Ax + Bu + w
        */
        fma_mv(x+t*nx+nx, wrk->A+t*nx*nx, x+t*nx, nx, nx, nx);
        fma_mv(x+t*nx+nx, wrk->B+t*nx*nu, u+t*nu, nx, nu, nu);
    }
}

void get_Cu_Dx(workspace* wrk, u32 all, f64* x, f64* u, f64* Dx, f64* Cu) {
    u32 N = wrk->N;
    u32 nx = wrk->nx;
    u32 nu = wrk->nu;
    u32* mx = wrk->mx;
    u32* cmx = wrk->cmx;
    u32* mu = wrk->mu;
    u32* cmu = wrk->cmu;

    // Set s to zero
    memset(Cu, 0, get_ncu(wrk)*sizeof(f64));
    memset(Dx, 0, get_ncx(wrk)*sizeof(f64));

    if (!all) {
        for (u32 t=0; t<N; ++t) {
            for (u32 i=0; i<mu[t]; ++i) {
                constraint_t constraint = {(i32)t, (i32)i, 0};
                if (is_active(wrk, &constraint)) Cu[cmu[t] + i] = wrk->c[cmu[t] + i];
                else for (u32 j=0; j<nu; ++j) Cu[cmu[t] + i] += wrk->C[cmu[t]*nu + i*nu + j] * u[t*nu + j];
            }
            for (u32 i=0; i<mx[t]; ++i) {
                constraint_t constraint = {(i32)t, (i32)i, 1};
                if (is_active(wrk, &constraint)) Dx[cmx[t] + i] = wrk->d[cmx[t] + i];
                else for (u32 j=0; j<nx; ++j) Dx[cmx[t] + i] += wrk->D[cmx[t]*nx + i*nx + j] * x[t*nx+nx + j];
            }
        }
    } else {
        for (u32 t=0; t<N; ++t) {
            fma_mv(Cu + cmu[t], wrk->C + cmu[t]*nu, u + t*nu, mu[t], nu, nu);
            fma_mv(Dx + cmx[t], wrk->D + cmx[t]*nx, x + t*nx + nx, mx[t], nx, nx);
        }
    }
}

void get_lqr_qr(workspace* wrk) {
    /*
        Compute terms r = C' \mu, q = D' \lam.
    */
    i32* active_u = wrk->as.active_u;
    i32* active_x = wrk->as.active_x;
    memcpy(wrk->r_wrk, wrk->r, wrk->N*wrk->nu*sizeof(f64));
    memcpy(wrk->q_wrk, wrk->q, wrk->N*wrk->nx*sizeof(f64));
    get_dual_linear_terms(wrk->r_wrk, active_u, wrk->C, wrk->xi, wrk->N, wrk->nu, wrk->mu, wrk->cmu);
    get_dual_linear_terms(wrk->q_wrk, active_x, wrk->D, wrk->xi, wrk->N, wrk->nx, wrk->mx, wrk->cmx);
}

void get_dual_linear_terms(f64* ref, i32* map, f64* mat, f64* xi, u32 N, u32 n, u32* m, u32* cumm) {
    for (u32 t=0; t<N; ++t) {
        for (u32 i=0; i<2*m[t]; i+=2) {
            // Retrieve dual variable
            i32 xi_idx = map[2*cumm[t] + i];
            u32 mat_idx = map[2*cumm[t] + i + 1];
            if (xi_idx < 0) break;
            
            // Accumulate ref += mat[mat_idx]' * dual_var
            for (u32 j=0; j<n; ++j)
                ref[t*n + j] += mat[cumm[t]*n + mat_idx*n + j] * xi[xi_idx];
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
    f64* d, f64* c, f64* x0, u32 N, 
    u32 nx, u32 nu, u32* mx, u32* mu, 
    u32 max_iter)
{   
    wrk->N = N;
    wrk->nx = nx;
    wrk->nu = nu;
    wrk->dH_singular = 0;
    wrk->max_iter = max_iter;

    /*
        Allocate statically sized memory
    */
    u32 nfloats = 4*N*nx*nx + // A, P, Q, Acl
                  3*N*nx*nu + // B, K, S
                  2*N*nu*nu +   // Lu, R
                  6*N*nx+2*nx + // q, q_wrk, x, x_lqr, w, Pw
                  4*N*nu;     // u, u_lqr, r, r_wrk
    wrk->smemory = malloc(
        nfloats*sizeof(f64) +
        4*N*sizeof(u32) // mx, cmx, mu, cmu
    );
    f64* mem = (f64*) wrk->smemory;
    wrk->A = mem; mem+=N*nx*nx;
    wrk->Acl = mem; mem+=N*nx*nx;
    wrk->P = mem; mem+=N*nx*nx;
    wrk->Q = mem; mem+=N*nx*nx;
    wrk->B = mem; mem+=N*nx*nu;
    wrk->K = mem; mem+=N*nx*nu;
    wrk->S = mem; mem+=N*nx*nu;
    wrk->Lu = mem; mem+=N*nu*nu;
    wrk->R = mem; mem+=N*nu*nu;
    wrk->w = mem; mem+=N*nx;
    wrk->Pw = mem; mem+=N*nx;
    wrk->x = mem; mem+=N*nx+nx;
    wrk->x_lqr = mem; mem+=N*nx+nx;
    wrk->u = mem; mem+=N*nu;
    wrk->u_lqr = mem; mem+=N*nu;
    wrk->q = mem; mem+=N*nx;
    wrk->q_wrk = mem; mem+=N*nx;
    wrk->r = mem; mem+=N*nu;
    wrk->r_wrk = mem; mem+=N*nu;
    unsigned char* vmem = (unsigned char*) mem; 
    wrk->mx = (u32*) vmem; vmem+=N*sizeof(u32);
    wrk->cmx = (u32*) vmem; vmem+=N*sizeof(u32);
    wrk->mu = (u32*) vmem; vmem+=N*sizeof(u32);
    wrk->cmu = (u32*) vmem; vmem+=N*sizeof(u32);


    /*
        Allocate dynamically sized memory
    */
    allocate_dynamic_workspace(wrk, mx, mu);
    
    u32 ncx = get_ncx(wrk);
    u32 ncu = get_ncu(wrk);
    memcpy(wrk->A, A, N*nx*nx*sizeof(f64));
    memcpy(wrk->B, B, N*nx*nu*sizeof(f64));
    memcpy(wrk->w, w, N*nx*sizeof(f64));
    memcpy(wrk->P, Q, N*nx*nx*sizeof(f64));
    memcpy(wrk->Q, Q, N*nx*nx*sizeof(f64));
    memcpy(wrk->S, S, N*nu*nx*sizeof(f64));
    memcpy(wrk->R, R, N*nu*nu*sizeof(f64));
    memcpy(wrk->q, q, N*nx*sizeof(f64));
    memcpy(wrk->r, r, N*nu*sizeof(f64));
    memcpy(wrk->D, D, ncx*nx*sizeof(f64));
    memcpy(wrk->C, C, ncu*nu*sizeof(f64));
    memcpy(wrk->d, d, ncx*sizeof(f64));
    memcpy(wrk->c, c, ncu*sizeof(f64));
    memcpy(wrk->x, x0, nx*sizeof(f64));
    memcpy(wrk->x_lqr, x0, nx*sizeof(f64));

    // Initialize Working Set.
    reset_working_set(wrk);

    // Initialize LQR/Riccati data
    solve_riccati(wrk);
    get_lqr_qr(wrk);
    solve_lqr(wrk, wrk->x_lqr, wrk->u_lqr);
    get_Cu_Dx(wrk, 0, wrk->x_lqr, wrk->u_lqr, wrk->Dx_lqr, wrk->Cu_lqr);
}

void allocate_dynamic_workspace(workspace* wrk, u32* mx, u32* mu) {
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

    u32 tmp2_size = nx * (nx > nu ? nx : nu);
    u32 tmp1_size = nc > tmp2_size ? nc : tmp2_size;
    u32 nfloats = 5*nc +          // Dx, Cu, Dx_lqr, Cu_lqr, xi, p, b_wrk
                ncx*nx+ncu*nu + // D, C
                ncx+ncu +       // d, c
                nc*nc +         // L
                nc*N*nu +       // M
                tmp1_size +     // tmp1
                tmp2_size;      // tmp2
    wrk->dmemory = malloc(
        nfloats*sizeof(f64) +
        nc*sizeof(u32) +              // active_set.as_members
        (2*ncx + 2*ncu)*sizeof(i32) + // active_set.active_x/active_u
        nc*sizeof(constraint_t)       // active_set.xi2con
    );
    f64* mem = (f64*) wrk->dmemory;
    wrk->Dx = mem; mem+=ncx;
    wrk->Dx_lqr = mem; mem+=ncx;
    wrk->Cu = mem; mem+=ncu;
    wrk->Cu_lqr = mem; mem+=ncu;
    wrk->xi = mem; mem+=nc;
    wrk->p = mem; mem+=nc;
    wrk->b_wrk = mem; mem+=nc;
    wrk->D = mem; mem+=ncx*nx;
    wrk->C = mem; mem+=ncu*nu;
    wrk->d = mem; mem+=ncx;
    wrk->c = mem; mem+=ncu;
    wrk->L = mem; mem+=nc*nc;
    wrk->M = mem; mem+=nc*N*nu;
    wrk->tmp1 = mem; mem+=tmp1_size;
    wrk->tmp2 = mem; mem+=tmp2_size;
    unsigned char* vmem = (unsigned char*) mem;
    wrk->as.as_members = (u32*) vmem; vmem+=nc*sizeof(u32);
    wrk->as.active_x = (i32*) vmem; vmem+=2*ncx*sizeof(i32); 
    wrk->as.active_u = (i32*) vmem; vmem+=2*ncu*sizeof(i32); 
    wrk->as.xi2con = (constraint_t*) vmem; vmem+=nc*sizeof(constraint_t);
}

void workspace_free(workspace* wrk) {
    free(wrk->smemory);
    free(wrk->dmemory);
}

void solve_riccati(workspace* wrk) {
    u32 N = wrk->N;
    u32 nx = wrk->nx;
    u32 nu = wrk->nu;
    f64* tmp1 = wrk->tmp1;
    f64* tmp2 = wrk->tmp2;

    memcpy(wrk->Lu, wrk->R, N*nu*nu*sizeof(f64));
    memcpy(wrk->K, wrk->S, N*nu*nx*sizeof(f64));
    memcpy(wrk->Acl, wrk->A, N*nx*nx*sizeof(f64));
    memset(wrk->Pw, 0, N*nx*sizeof(f64));
    for (i32 t=N-1; t>=0; t--) {
        // Compute Lu = chol(R + B'PB)
        transpose(tmp2, wrk->B + t*nx*nu, nx, nu);
        memset(tmp1, 0, nx*nu*sizeof(f64));
        fma_mm_nt(tmp1, tmp2, wrk->P + t*nx*nx, nu, nx, nx, nx);
        fma_mm_nt(wrk->Lu + t*nu*nu, tmp2, tmp1, nu, nu, nx, nu);
        cholesky(wrk->Lu + t*nu*nu, nu);

        // Compute K = Lu^{-T}Lu^{-1}(S + B'PA)
        transpose(tmp2, wrk->A + t*nx*nx, nx, nx);
        fma_mm_nt(wrk->K+t*nx*nu, tmp1, tmp2, nu, nx, nx, nx);
        trsm(wrk->K + t*nx*nu, wrk->Lu + t*nu*nu, nu, nx);
        trsm_t(wrk->K + t*nx*nu, wrk->Lu + t*nu*nu, nu, nx);

        // Compute Acl = A - BK
        for (u32 i=0; i<nx; ++i)
            for (u32 j=0; j<nx; ++j)
                for (u32 k=0; k<nu; ++k) 
                    wrk->Acl[t*nx*nx + i*nx+j] -= wrk->B[t*nx*nu + i*nu + k] * wrk->K[t*nx*nu + k*nx + j];

        // Compute Pw
        fma_mv(wrk->Pw + t*nx, wrk->P + t*nx*nx, wrk->w + t*nx, nx, nx, nx);

        if (t==0) return;
        // Compute P = Q + A'P Acl - S'K
        memset(tmp1, 0, nx*nx*sizeof(f64));
        fma_mm_nt(tmp1, tmp2, wrk->P+t*nx*nx, nx, nx, nx, nx);
        for (u32 i=0; i<nx; ++i)
            for (u32 j=0; j<nx; ++j)
                for (u32 k=0; k<nx; ++k)
                    wrk->P[(t-1)*nx*nx + i*nx+j] += tmp1[i*nx + k] * wrk->Acl[t*nx*nx + k*nx + j];
        for (u32 i=0; i<nx; ++i)
            for (u32 j=0; j<nx; ++j)
                for (u32 k=0; k<nu; ++k)
                    wrk->P[(t-1)*nx*nx + i*nx+j] -= wrk->S[t*nx*nu + k*nx + i] * wrk->K[t*nx*nu + k*nx + j];
        transpose(tmp1, wrk->P+(t-1)*nx*nx, nx, nx);
        for (u32 i=0; i<nx*nx; ++i) 
            wrk->P[(t-1)*nx*nx + i] = 0.5*(wrk->P[(t-1)*nx*nx + i] + tmp1[i]);

    }
}

void fma_mv(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride) {
    for (u32 i=0; i<ny; ++i)
        for (u32 j=0; j<nx; ++j) y[i] += A[i*stride + j] * x[j];
}

void fma_mv_t(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride) {
    for (u32 i=0; i<nx; ++i)
        for (u32 j=0; j<ny; ++j) y[j] += A[i*stride + j] * x[i];
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
    for (u32 i=0; i<nr; ++i)
        for (u32 j=0; j<nc; ++j)
            for (u32 l=0; l<k; ++l) 
                C[i*ostride + j] += A[i*k + l] * B[j*k + l];
}

void cholesky(f64* L, u32 n) {
    for (u32 i=0; i<n; ++i) {
        L[i*n + i] = sqrt(L[i*n + i]);
        for (u32 j=i+1; j<n; ++j) L[j*n + i] /= L[i*n + i];
        for (u32 j=i+1; j<n; ++j)
            for (u32 k=i+1; k<n; ++k) L[j*n + k] -= L[i + j*n] * L[i + k*n];
    }
}

void trsm(f64* X, f64* L, u32 nv, u32 nsys) {
    for (u32 i=0; i<nv; ++i) {
        for (u32 vi=0; vi<nsys; ++vi) X[i*nsys + vi] /= L[i*nv+i];
        for (u32 j=i+1; j<nv; ++j)
            for (u32 vi=0; vi<nsys; ++vi) X[j*nsys + vi] -= L[j*nv + i] * X[i*nsys + vi];
    }
}

void trsm_t(f64* X, f64* L, u32 nv, u32 nsys) {
    for (i32 i=nv-1; i>=0; --i) {
        for (u32 vi=0; vi<nsys; ++vi) X[i*nsys + vi] /= L[i*nv + i];
        for (i32 j=i-1; j>=0; --j)
            for (u32 vi=0; vi<nsys; ++vi) X[j*nsys + vi] -= L[i*nv + j] * X[i*nsys + vi];
    }
}

void negate(f64* v, u32 n) {
    for (u32 i=0; i<n; ++i) v[i] *= -1.0;
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
