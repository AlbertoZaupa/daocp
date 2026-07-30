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
    f64* xi;
    f64* x;
    f64* u;
    f64* sx;
    f64* su;
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
    f64* Lu;
    f64* K;
    f64* P;
    f64* Acl;
    f64* x_lqr;
    f64* u_lqr;
    f64* sx_lqr;
    f64* su_lqr;
    f64* tmp1;
    f64* tmp2;
    u32 return_status;
    u32 dH_singular;
    u32 nx;
    u32 nu;
    u32 mx;
    u32 mu;
    u32 N;
    u32 nc;
    u32 max_iter;
} workspace;

void solve(workspace* wrk);
void solve_dual_qp(workspace* wrk);
u32 get_descent_dir(workspace* wrk);
u32 drop_component(f64* xi, f64* p, u32 n);
void add_constraint(workspace* wrk, constraint_t* constr);
void remove_constraint(workspace* wrk, u32 idx);
void update_working_set_add(workspace* wrk, constraint_t* constr);
void update_working_set_remove(workspace* wrk, u32 idx);
void get_dH_row(workspace* wrk, constraint_t* constr);
void update_dH_chol_add(workspace* wrk);
void update_dH_chol_remove(workspace* wrk, u32 idx);
u32 is_dual_feasible(f64* p, u32 n);
void compute_slacks(workspace* wrk, constraint_t* constr);
void get_violated_constraint(workspace* wrk, constraint_t* constr);
u32 check_infeasibility(f64* p, u32 n);
void solve_lqr(workspace* wrk, f64* x, f64* u);
void get_Cu_Dx(workspace* wrk, f64* x, f64* u, f64* sx, f64* su);
void get_lqr_qr(workspace* wrk);
void get_dual_linear_terms(f64* ref, i32* map, f64* mat, f64* xi, u32 N, u32 n, u32 m);
u32 is_active(workspace* wrk, constraint_t* constr);
void set_active(workspace* wrk, constraint_t* constr);
void set_inactive(workspace* wrk, constraint_t* constr);
void fma_mv(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride);
void fma_mv_t(f64* y, f64* A, f64* x, u32 ny, u32 nx, u32 stride);
void trsv(f64* x, f64* L, u32 n, u32 stride);
void trsv_t(f64* x, f64* L, u32 n, u32 stride);
void negate(f64* v, u32 n);
f64 dot(f64* v, f64* w, u32 n);
void swap(f64** a, f64** b);

void solve(workspace* wrk) {
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
                    return;
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
                return;
            }
            if (check_infeasibility(wrk->p, wrk->as.n_active)) {
                wrk->return_status = INFEASIBLE;
                return;
            }
            removed_constr = drop_component(wrk->xi, wrk->p, wrk->as.n_active);
            remove_constraint(wrk, removed_constr);
        }
    }
    
    wrk->return_status = MAX_ITER;
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
    get_dH_row(wrk, constr);
    update_dH_chol_add(wrk);
    /*
        Update dual linear term
    */
    if (constr->is_state) {
        wrk->b_wrk[wrk->as.n_active] = 
            wrk->d[constr->idx] - wrk->sx_lqr[constr->t*wrk->mx + constr->idx];
    } else {
        wrk->b_wrk[wrk->as.n_active] = 
            wrk->c[constr->idx] - wrk->su_lqr[constr->t*wrk->mu + constr->idx];
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
        for (; i<2*wrk->mx && wrk->as.active_x[t*2*wrk->mx + i] >= 0; i+=2);
        wrk->as.active_x[t*2*wrk->mx + i] = wrk->as.n_active;
        wrk->as.active_x[t*2*wrk->mx + i + 1] = idx;
    } else {
        u32 i=0;
        for (; i<2*wrk->mu && wrk->as.active_u[t*2*wrk->mu + i] >= 0; i+=2);
        wrk->as.active_u[t*2*wrk->mu + i] = wrk->as.n_active;
        wrk->as.active_u[t*2*wrk->mu + i + 1] = idx;
    }

    // Update ( xi_idx -> constraint ) map.
    wrk->as.xi2con[wrk->as.n_active] = (constraint_t) {t, idx, is_state};

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
    u32 nc = is_state ? wrk->mx : wrk->mu;
    map += t*2*nc;
    u32 i = 0;
    for (i=0; i<2*nc && map[i] != xi_idx; i+=2) ;
    for (; i<2*(nc-1) && map[i] != -1; i+=2) {
        map[i] = map[i+2];
        map[i+1] = map[i+3];
    }
    map[2*(nc-1)] = map[2*nc-1] = -1;

    // Update active_x/u to reflect new xi indexing;
    nc = wrk->mu;
    map = wrk->as.active_u;
    for (u32 t=0; t<wrk->N; ++t)
        for (u32 i=0; i<2*nc && map[2*t*nc + i] >= 0; i+=2) 
            if (map[2*t*nc + i] > xi_idx) map[2*t*nc + i] -= 1;
    nc = wrk->mx;
    map = wrk->as.active_x;
    for (u32 t=0; t<wrk->N; ++t)
        for (u32 i=0; i<2*nc && map[2*t*nc + i] >= 0; i+=2) 
            if (map[2*t*nc + i] > xi_idx) map[2*t*nc + i] -= 1;

    // Compact xi.
    for (u32 i=xi_idx+1; i < wrk->as.n_active; ++i)
        wrk->xi[i-1] = wrk->xi[i];
    set_inactive(wrk, &(constraint_t) {t, idx, is_state});
    wrk->as.n_active -= 1;
}

void get_dH_row(workspace* wrk, constraint_t* constr) {
    /*
        H = M M', M = [CU; DX]
        where:
            - CU[t, k] = C Lu[t]^{-T} if t=k, -C K[t]Acl[t-1] .. Acl[k] B if k < t.
            - DX[t, k] = D Acl[t] ... Acl[k] B if k <= t.

        1) We first compute the approprate row of CU or DX.
        2) We then compute it's product with M.
    */
    f64* tmp1 = wrk->tmp1;
    f64* tmp2 = wrk->tmp2;
    u32 N = wrk->N;
    u32 nx = wrk->nx;
    u32 nu = wrk->nu;
    u32 n_active = wrk->as.n_active;
    u32 t = constr->t;
    memset(wrk->M + N*nu*n_active, 0, N*nu*sizeof(f64));
    f64* m_ptr = wrk->M + N*nu*n_active + t*nu;
    
    if (constr->is_state) { // Need DX
        // Initialize recursion state
        memcpy(tmp1, wrk->D + constr->idx*nx, nx*sizeof(f64));
        
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
        memcpy(m_ptr, wrk->C + constr->idx*nu, nu*sizeof(f64));
        trsv(m_ptr, wrk->Lu+nu*nu*t, nu, nu);
        
        // Initialize backward recursion state.
        if (t>0) {
            memset(tmp1, 0, nx*sizeof(f64));
            fma_mv_t(tmp1, wrk->K+t*nu*nx, wrk->C+constr->idx*nu, nx, nu, nx);
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

    /*
        Computation of new row of dH. Given by (M' m, m'm).
        Write result into new row of L.
    */
    memset(wrk->L + n_active*wrk->nc, 0, (n_active+1)*sizeof(f64));
    m_ptr = wrk->M + n_active*N*nu;
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
    get_Cu_Dx(wrk, wrk->x, wrk->u, wrk->sx, wrk->su);
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
        Check su
    */
    for (u32 tau=0; tau<wrk->N; ++tau)
        for (u32 i=0; i<wrk->mu; ++i) {
            f64 tmp = wrk->su[tau*wrk->mu + i] - wrk->c[i];
            if (tmp > ZERO_TOL && tmp > maxu) {
                maxu = tmp;
                idxu = i;
                tu = tau;
            }
    }
    
    /*
        Check sx
    */
    for (u32 tau=0; tau<wrk->N; ++tau)
        for (u32 i=0; i<wrk->mx; ++i) {
            f64 tmp = wrk->sx[tau*wrk->mx + i] - wrk->d[i];
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

    memset(x + nx, 0, N*nx*sizeof(f64));
    // Initialize costate
    memcpy(tmp1, wrk->q_wrk + (N-1)*nx, nx*sizeof(f64)); 

    // Backward recursion
    for (i32 t=N-1; t>=0; t--) {
        /*
            Compute u = - (R + B'PB)^{-1}(r + B'p)
        */
        for (u32 i=0; i<nu; ++i) u[t*nu + i] = wrk->r_wrk[t*nu + i];
        fma_mv_t(u+t*nu, wrk->B+t*nu*nx, tmp1, nu, nx, nu);
        negate(u + t*nu, nu);
        trsv(u+t*nu, wrk->Lu+t*nu*nu, nu, nu);
        trsv_t(u+t*nu, wrk->Lu+t*nu*nu, nu, nu);

        if (t==0) break;
        /*
            Compute p = (A - BK)' p - K'r + q 
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
            x = Ax + Bu
        */
        fma_mv(x+t*nx+nx, wrk->A+t*nx*nx, x+t*nx, nx, nx, nx);
        fma_mv(x+t*nx+nx, wrk->B+t*nx*nu, u+t*nu, nx, nu, nu);
    }
}

void get_Cu_Dx(workspace* wrk, f64* x, f64* u, f64* sx, f64* su) {
    u32 N = wrk->N;
    u32 nx = wrk->nx;
    u32 nu = wrk->nu;
    u32 mx = wrk->mx;
    u32 mu = wrk->mu;

    // Set s to zero
    memset(su, 0, N*mu*sizeof(f64));
    memset(sx, 0, N*mx*sizeof(f64));

    for (u32 t=0; t<wrk->N; ++t) {
        for (u32 i=0; i<mu; ++i) {
            if (is_active(wrk, &(constraint_t) {t, i, 0})) su[t*mu + i] = wrk->c[i];
            else for (u32 j=0; j<nu; ++j) su[t*mu + i] += wrk->C[i*nu + j] * u[t*nu + j]; 
        }
        for (u32 i=0; i<mx; ++i) {
            if (is_active(wrk, &(constraint_t) {t, i, 1})) sx[t*mx + i] = wrk->d[i];
            else for (u32 j=0; j<nx; ++j) sx[t*mx + i] += wrk->D[i*nx + j] * x[t*nx+nx + j]; 
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
    get_dual_linear_terms(wrk->r_wrk, active_u, wrk->C, wrk->xi, wrk->N, wrk->nu, wrk->mu);
    get_dual_linear_terms(wrk->q_wrk, active_x, wrk->D, wrk->xi, wrk->N, wrk->nx, wrk->mx);
}

void get_dual_linear_terms(f64* ref, i32* map, f64* mat, f64* xi, u32 N, u32 n, u32 m) {
    for (u32 t=0; t<N; ++t) {
        for (u32 i=0; i<2*m; i+=2) {
            // Retrieve dual variable
            i32 xi_idx = map[2*t*m + i];
            u32 mat_idx = map[2*t*m + i + 1];
            if (xi_idx < 0) break;
            
            // Accumulate ref += mat[mat_idx]' * dual_var
            for (u32 j=0; j<n; ++j)
                ref[t*n + j] += mat[mat_idx*n + j] * xi[xi_idx];
        }
    }
}

u32 is_active(workspace* wrk, constraint_t* constr) {
    u32 t = constr->t;
    u32 i = constr->idx;
    u32 is_state = constr->is_state;
    return wrk->as.as_members[t*(wrk->mx+wrk->mu) + is_state*wrk->mu + i];
}

void set_active(workspace* wrk, constraint_t* constr) {
    u32 t = constr->t;
    u32 i = constr->idx;
    u32 is_state = constr->is_state;
    wrk->as.as_members[t*(wrk->mx+wrk->mu) + is_state*wrk->mu + i] = 1;
}

void set_inactive(workspace* wrk, constraint_t* constr) {
    u32 t = constr->t;
    u32 i = constr->idx;
    u32 is_state = constr->is_state;
    wrk->as.as_members[t*(wrk->mx+wrk->mu) + is_state*wrk->mu + i] = 0;
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
