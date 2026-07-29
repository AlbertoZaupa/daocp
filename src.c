#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
typedef uint32_t u32;
typedef double f64;
typedef int32_t i32;

#define ZERO_TOL 1e-15
#define SOLVED 1
#define INFEASIBLE 2
#define MAX_ITER 3
#define PW2(x) x*x

typedef struct {
    i32 t;
    i32 idx;
    u32 is_state;
} constraint_t;

typedef struct {
    i32* active_u; // For each time-index: (xi index active at t, corresponding constraint index).
    i32* active_x;
    u32* xi2con; // For each dual variable, the corresponding constraint.
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
void get_descent_dir(workspace* wrk);
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
                removed_constr = drop_component(wrk->xi, wrk->p, wrk->as.n_active);
                remove_constraint(wrk, removed_constr);
            }
        } else {
            get_descent_dir(wrk);
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
    for (u32 i=0; i<n_active; ++i) {
        wrk->p[i] /= wrk->L[i*wrk->nc + i];
        for (u32 j=i+1; j<n_active; ++j) 
            wrk->p[j] -= wrk->L[j*wrk->nc + i] * wrk->p[i];
    }

    // Solve L'p = y
    for (i32 i=n_active-1; i>=0; --i) {
        wrk->p[i] /= wrk->L[i*wrk->nc + i];
        for (i32 j=i-1; j>=0; --j)
            wrk->p[j] -= wrk->L[i*wrk->nc + j] * wrk->p[i];
    }
}

void get_descent_dir(workspace* wrk) {
    u32 n_active = wrk->as.n_active;
    u32 nc = wrk->nc;

    // Solve LL'p = 0, p != 0. Assume L_{n_active, n_active} = 0.
    memset(wrk->p, 0, n_active*sizeof(f64));
    wrk->p[n_active-1] = 1.0;
    wrk->p[n_active-2] = wrk->L[(n_active-1)*nc + n_active-2];
    for (i32 i=n_active-2; i>=0; i--) {
        wrk->p[i] /= wrk->L[i*nc + i];
        for (i32 j=i-1; j>=0; j--)
            wrk->p[j] -= wrk->L[i*nc + j] * wrk->p[i];
    }

    // Enforce p' b < 0.
    f64 dot = 0.0;
    for (u32 i=0; i<n_active; ++i) dot += wrk->p[i] * wrk->b_wrk[i];
    if (dot > ZERO_TOL) {
        for (u32 i=0; i<n_active; ++i) wrk->p[i] *= -1.0;
    }
}

u32 drop_component(f64* xi, f64* p, u32 n) {
    /*
        Line search along p < 0:
            xi + t(p-xi) = 0 \iff
            t = xi / (xi-p)
    */
    f64 t = 1.0;
    f64 argmin = -1;
    for (u32 i=0; i<n; ++i) {
        if (p[i] > ZERO_TOL) continue;
        f64 tau = xi[i] / (xi[i] - p[i]);
        if (tau < t) {
            t = tau;
            argmin = i;
        }
    }
    for (u32 i=0; i<n; ++i) xi[i] += t*(p[i]-xi[i]);
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

    // Update ( t -> active constraints ) map.
    if (constr->is_state) {
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
    wrk->as.xi2con[3*wrk->as.n_active] = t;
    wrk->as.xi2con[3*wrk->as.n_active+1] = idx;
    wrk->as.xi2con[3*wrk->as.n_active+2] = constr->is_state;

    wrk->xi[wrk->as.n_active] = 0.0;
    wrk->as.n_active += 1;
}

void update_working_set_remove(workspace* wrk, u32 xi_idx) {
    // Retrieve constraint info.
    u32 t = wrk->as.xi2con[xi_idx*3];
    u32 idx = wrk->as.xi2con[xi_idx*3+1];
    u32 is_state = wrk->as.xi2con[xi_idx*3+2];

    // Update (xi_idx -> constraint info) map.
    for (u32 i=xi_idx+1; i<wrk->as.n_active; ++i) {
        wrk->as.xi2con[(i-1)*3] = wrk->as.xi2con[i*3]; 
        wrk->as.xi2con[(i-1)*3+1] = wrk->as.xi2con[i*3+1]; 
        wrk->as.xi2con[(i-1)*3+2] = wrk->as.xi2con[i*3+2]; 
    }

    // Update (t -> active constraints) map.
    i32* map = is_state ? wrk->as.active_x : wrk->as.active_u;
    u32 nc = is_state ? wrk->mx : wrk->mu;
    map += t*2*nc;
    u32 i = 0;
    for (i=0; i<2*nc && map[i] != idx; i+=2) ;
    for (; i<2*(nc-1) && map[i] != -1; i+=2) {
        map[i] = map[i+2];
        map[i+1] = map[i+3];
    }
    map[2*(nc-1)] = map[2*nc-1] = -1;

    wrk->as.n_active -= 1;
}

void get_dH_row(workspace* wrk, constraint_t* constr) {
    /*
        H = M M', M = [CU; DX]
        where:
            - CU[t, k] = C Lu[t]^{-T} if t=k, -C K[t]Acl[t-1] .. Acl[k] B if k < t.
            - DX[t, k] = D Acl[t-1] ... Acl[k] B if k <= t.

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
    memset(wrk->M + N*nu*n_active, 0, N*nu*n_active*sizeof(f64));
    f64* m_ptr = wrk->M + N*nu*n_active + t*nu;
    
    if (constr->is_state) { // Need DX
        // Initialize recursion state
        memcpy(tmp1, wrk->D + constr->idx*nx, nx*sizeof(f64));
        
        // Compute B' Di
        memset(m_ptr, 0, nu*sizeof(f64));
        for (u32 j=0; j<nx; ++j)
            for (u32 i=0; i<nu; ++i) m_ptr[i] += wrk->B[t*nx*nu + j*nu + i] * tmp1[j];
        // Solve Lu x = B' Di
        for (u32 i=0; i<nu; ++i) {
            m_ptr[i] /= wrk->Lu[t*nu*nu + i*nu + i];
            for (u32 j=i+1; j<nu; ++j) m_ptr[j] -= wrk->Lu[t*nu*nu + j*nu + i] * m_ptr[i];
        }
        // walk back along row of M.
        m_ptr -= nu;

        // Propagate through Acl recursion state
        for (i32 tau=t-1; tau>=0; --tau) {
            // state = Acl[tau]' state
            memset(tmp2, 0, nx*sizeof(f64));
            for (u32 j=0; j<nx; ++j)
                for (u32 i=0; i<nx; ++i) tmp2[i] += wrk->Acl[tau*nx*nx + j*nx + i] * tmp1[j];
            swap(&tmp1, &tmp2);

            // Compute B' state
            memset(m_ptr, 0, nu*sizeof(f64));
            for (u32 j=0; j<nx; ++j)
                for (u32 i=0; i<nu; ++i) m_ptr[i] += wrk->B[tau*nx*nu + j*nu + i] * tmp1[j];
            // Solve Lu x = B' state
            for (u32 i=0; i<nu; ++i) {
                m_ptr[i] /= wrk->Lu[tau*nu*nu + i*nu + i];
                for (u32 j=i+1; j<nu; ++j) m_ptr[j] -= wrk->Lu[tau*nu*nu + j*nu + i] * m_ptr[i];
            }
            // walk back along row of M.
            m_ptr -= nu;            
        }
    } else { // Need DU
        memcpy(m_ptr, wrk->C + constr->idx*nu, nu*sizeof(f64));
        for (u32 i=0; i<nu; ++i) {
            m_ptr[i] /= wrk->Lu[t*nu*nu + i*nu + i];
            for (u32 j=i+1; j<nu; ++j) m_ptr[j] -= wrk->Lu[t*nu*nu + j*nu + i] * m_ptr[i];
        }
        
        // Initialize backward recursion state.
        if (t>0) {
            memset(tmp1, 0, nx*sizeof(f64));
            for (u32 j=0; j<nu; ++j)
                for (u32 i=0; i<nx; ++i) tmp1[i] -= wrk->K[t*nu*nx + j*nx+i] *
                                                    wrk->C[constr->idx*nu + j];
        }
        // Walk back along row of M.
        m_ptr -= nu;
        
        // Start recursion
        for (i32 tau=t-1; tau>=0; --tau) {
            // Compute B' state
            memset(m_ptr, 0, nu*sizeof(f64));
            for (u32 j=0; j<nx; ++j)
                for (u32 i=0; i<nu; ++i) m_ptr[i] += wrk->B[tau*nx*nu + j*nu + i] * tmp1[j];
            // Solve Lu x = B' state
            for (u32 i=0; i<nu; ++i) {
                m_ptr[i] /= wrk->Lu[tau*nu*nu + i*nu + i];
                for (u32 j=i+1; j<nu; ++j) m_ptr[j] -= wrk->Lu[tau*nu*nu + j*nu + i] * m_ptr[i];
            }
            // walk back along row of M.
            m_ptr -= nu;  

            // state = Acl[tau]' state
            memset(tmp2, 0, nx*sizeof(f64));
            for (u32 j=0; j<nx; ++j)
                for (u32 i=0; i<nx; ++i) tmp2[i] += wrk->Acl[tau*nx*nx + j*nx + i] * tmp1[j];
            swap(&tmp1, &tmp2);          
        }
    }

    /*
        Computation of new row of dH. Given by (M' m, m'm).
        Write result into new row of L.
    */
    memset(wrk->L + n_active*wrk->nc, 0, (n_active+1)*sizeof(f64));
    m_ptr = wrk->M + n_active*N*nu;
    for (u32 i=0; i<n_active; ++i)
        for (u32 j=0; j<N*nu; ++j) wrk->L[n_active*wrk->nc+i] += 
                                    wrk->M[i*N*nu + j] * m_ptr[j]; 
    for (u32 i=0; i<n_active; ++i)
        wrk->L[n_active*wrk->nc + n_active] += PW2(m_ptr[i]);
}

void update_dH_chol_add(workspace* wrk) {
    u32 n_active = wrk->as.n_active;
    // Solve
    for (u32 i=0; i<n_active; ++i) {
        wrk->L[n_active*wrk->nc + i] /= wrk->L[i*wrk->nc + i];
        for (u32 j=i+1; j<n_active; ++j)
            wrk->L[n_active*wrk->nc + j] -= wrk->L[j*wrk->nc + i] * wrk->L[n_active*wrk->nc + i];
    }

    // Diagonal
    for (u32 i=0; i<n_active; ++i)
        wrk->L[n_active*wrk->nc + n_active] -= wrk->L[n_active*wrk->nc + i] *
                                               wrk->L[n_active*wrk->nc + i];
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
        memcpy(wrk->L+(i-1)*nc, wrk->L+i*nc, i*sizeof(f64));

    // Extract column[idx] at l. Fix bottom-right lower triangle
    for (u32 i=idx; i<n_active-1; ++i) {
        l[i-idx] = wrk->L[i*nc + idx];
        for (u32 j=idx+1; j<n_active; ++j)
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
            l[j-idx] = l[j-idx] * b - wrk->L[j*nc + i] * a;
            wrk->L[j*nc + i] *= b;
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
    u32 tu = -1;
    u32 tx = -1;
    u32 idxu = -1;
    u32 idxx = -1;
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
    u32 mu = wrk->mu;
    u32 mx = wrk->mx;
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
        for (u32 i=0; i<nu; ++i) u[t*nu + i] = -wrk->r_wrk[t*nu + i];
        for (u32 j=0; j<nx; ++j) 
            for (u32 i=0; i<nu; ++i) u[t*nu + i] -= wrk->B[t*nu*nx + j*nu + i] * tmp1[j];
        for (u32 i=0; i<nu; ++i) {
            u[t*nu + i] /= wrk->Lu[t*nu*nu + i*nu + i];
            for (u32 j=i+1; j<nu; ++j) u[t*nu + j] -= wrk->Lu[t*nu*nu + j*nu + i] * u[t*nu + i];
        }
        for (i32 i=nu-1; i>=0; i--) {
            u[t*nu + i] /= wrk->Lu[t*nu*nu + i*nu + i];
            for (i32 j=i-1; j>=0; j--) u[t*nu + j] -= wrk->Lu[t*nu*nu + i*nu + j] * u[t*nu + i];
        }

        if (t==0) break;
        /*
            Compute p = (A - BK)' p - K'r + q 
        */
        for (u32 i=0; i<nx; ++i) tmp2[i] = wrk->q_wrk[(t-1)*nx + i];
        for (u32 j=0; j<nx; ++j)
            for (u32 i=0; i<nx; ++i) tmp2[i] += wrk->Acl[t*nx*nx + j*nx + i] * tmp1[j];
        for (u32 i=0; i<nx; ++i) tmp1[i] = tmp2[i];
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
        for (u32 i=0; i<nx; ++i) {
            for (u32 j=0; j<nx; ++j) x[t*nx+nx+i] += wrk->A[t*nx*nx+i*nx+j] * x[t*nx+j];
            for (u32 j=0; j<nu; ++j) x[t*nx+nx+i] += wrk->B[t*nx*nu+i*nu+j] * u[t*nu+j];
        }
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
        /*
            su = C u
        */
        for (u32 i=0; i<mu; ++i)
            for (u32 j=0; j<nu; ++j) su[t*mu + i] += wrk->C[i*nu + j] * u[t*nu + j];

        /*
            sx = D x
        */
        for (u32 i=0; i<mx; ++i) 
            for (u32 j=0; j<nx; ++j) sx[t*mx + i] += wrk->D[nx*i + j] * x[t*nx + nx + j]; 
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

void swap(f64** a, f64** b) {
    f64* tmp = *a;
    *a = *b;
    *b = tmp;
}
