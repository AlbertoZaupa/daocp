/*
 * Copyright (c) 2026 Alberto Zaupa
 * SPDX-License-Identifier: MIT
 * See LICENSE in the project root for license information.
 */

#include <internal.h>

void daocp_solve_riccati(daocp_workspace* wrk, daocp_qp* qp) {
    u32 N = qp->dims.N;
    u32* nx = qp->dims.nx;
    u32* nu = qp->dims.nu;
    u32* ne = qp->dims.ne;
    u32* m = wrk->m;
    struct blasfeo_dmat* tmp1 = &wrk->tmp2;
    f64* GEtmp = wrk->GEtmp;
    f64* ABtmp = wrk->ABtmp;
    f64* H = wrk->H; f64* h = wrk->h;
    f64** Dx = qp->Dx; f64** Du = qp->Du;
    f64** J = wrk->J; f64* Jmem = J[N-1];

    // Initialize recursion
    blasfeo_dgecp(nx[N], nx[N], qp->RSQrq+N, 0, 0, wrk->P+N-1, 0, 0);
    blasfeo_dtrtr_l(nx[N], wrk->P+N-1, 0, 0, wrk->P+N-1, 0, 0);
    u32 neq_x0 = qp->dims.ne[N];
    f64* tmp = H;
    for (u32 i=0; i<neq_x0; ++i) {
        memcpy(tmp, qp->Dx[N]+i*nx[N], nx[N]*sizeof(f64));
        tmp += nx[N];
    }
    memcpy(h, qp->d[N], ne[N]*sizeof(f64));

    for (i32 t=N-1; t>=0; t--) {
        // Compute Lu00 = chol(R + B'PB)
        blasfeo_dgemm_nt(nu[t], nx[t+1], nx[t+1], 1.0, qp->BAwt+t, 0, 0, wrk->P+t, 0, 0, 0.0, tmp1, 0, 0, tmp1, 0, 0);
        blasfeo_dsyrk_dpotrf_ln(nu[t], nx[t+1], tmp1, 0, 0, qp->BAwt+t, 0, 0, qp->RSQrq+t, 0, 0, wrk->Luu+t, 0, 0); 

        /* 
            Gaussian elimination to propagate constraints
        */
        
        m[t] = ne[t] + neq_x0;
        // Form [Du; HB | Dx; HA | d; h-Hw]
        memset(GEtmp, 0, (nx[t]+nu[t]+1)*(ne[t]+neq_x0)*sizeof(f64));
        for (u32 i=0; i<ne[t]; ++i) memcpy(GEtmp+i*(nx[t]+nu[t]+1), Du[t]+i*nu[t], nu[t]*sizeof(f64));
        for (u32 i=0; i<ne[t]; ++i) memcpy(GEtmp+i*(nx[t]+nu[t]+1)+nu[t], Dx[t]+i*nx[t], nx[t]*sizeof(f64));
        for (u32 i=0; i<ne[t]; ++i) GEtmp[i*(nx[t]+nu[t]+1)+nx[t]+nu[t]] = qp->d[t][i];
        blasfeo_unpack_tran_dmat(nu[t], nx[t+1], qp->BAwt+t, 0, 0, ABtmp, nx[t+1]);
        daocp_fma_mm_nt(GEtmp+ne[t]*(nx[t]+nu[t]+1), H, ABtmp, neq_x0, nu[t], nx[t+1], nx[t]+nu[t]+1);
        blasfeo_unpack_tran_dmat(nx[t], nx[t+1], qp->BAwt+t, nu[t], 0, ABtmp, nx[t+1]);
        daocp_fma_mm_nt(GEtmp+ne[t]*(nx[t]+nu[t]+1)+nu[t], H, ABtmp, neq_x0, nx[t], nx[t+1], nx[t]+nu[t]+1);
        for (u32 i=0; i<neq_x0; ++i) GEtmp[(ne[t]+i)*(nx[t]+nu[t]+1)+nx[t]+nu[t]] = h[i];
        for (u32 i=0; i<nx[t+1]; ++i)
            ABtmp[i] = BLASFEO_DMATEL(qp->BAwt+t, nu[t]+nx[t], i); 
        for (u32 i=0; i<neq_x0; ++i)
            GEtmp[(ne[t]+i)*(nx[t]+nu[t]+1)+nx[t]+nu[t]] -= daocp_dot(H+i*nx[t+1], ABtmp, nx[t+1]);

        // Gaussian elimination
        u32 rho = wrk->rho[t] = daocp_gaussian_elimination(GEtmp, J[t], GEtmp + (ne[t]+neq_x0)*(nx[t]+nu[t]+1), ne[t]+neq_x0, nu[t], nx[t]+nu[t]+1, nu[t]);
        Jmem += (rho*(2*m[t]+1-rho)) >> 1;
        if (t>0) J[t-1] = Jmem;
        // Copy new H, h.
        for (u32 i=rho; i<ne[t]+neq_x0; ++i) {
            memcpy(wrk->H + (i-rho)*nx[t], GEtmp + i*(nx[t]+nu[t]+1) + nu[t], nx[t]*sizeof(f64));
            wrk->h[i-rho] = GEtmp[i*(nx[t]+nu[t]+1)+nx[t]+nu[t]];
        }
        // Store -b.
        for (u32 i=0; i<rho; ++i) wrk->b[t].pa[i] = -GEtmp[i*(nx[t]+nu[t]+1)+nx[t]+nu[t]];
        neq_x0 = neq_x0+ne[t]-rho;

        /* 
            Complete LDL of KKT matrix [(R+B'PB) G'; G 0]
        */
        if (rho) {
            blasfeo_pack_tran_dmat(nu[t], rho, GEtmp, nx[t]+nu[t]+1, wrk->Lue+t, 0, 0);
            blasfeo_dtrsm_rltn(rho, nu[t], 1.0, wrk->Luu+t, 0, 0, wrk->Lue+t, 0, 0, wrk->Lue+t, 0, 0);
            blasfeo_dgese(rho, rho, 0.0, wrk->Lee+t, 0, 0);
            blasfeo_dsyrk_dpotrf_ln(rho, nu[t], wrk->Lue+t, 0, 0, wrk->Lue+t, 0, 0, wrk->Lee+t, 0, 0, wrk->Lee+t, 0, 0);
        }

        /*
            Compute partial feedback gains.
        */

        // Compute K0 = (A'PB + S')Lu00^{-T}
        blasfeo_dgemm_nt(nx[t], nu[t], nx[t+1], 1.0,
                        qp->BAwt+t, nu[t], 0, tmp1, 0, 0,
                        1.0, qp->RSQrq+t, nu[t], 0,
                        wrk->Ku+t, 0, 0);
        blasfeo_dtrsm_rltn(nx[t], nu[t], 1.0, wrk->Luu+t, 0, 0, wrk->Ku+t, 0, 0, wrk->Ku+t, 0, 0);
        // Compute K1 = (M' - K0 Lu01')Lu11^{-T}
        if (rho) {
            blasfeo_pack_dmat(nx[t], rho, GEtmp+nu[t], nx[t]+nu[t]+1, wrk->Ke+t, 0, 0);
            blasfeo_dgemm_nt(nx[t], rho, nu[t], -1.0, wrk->Ku+t, 0, 0, wrk->Lue+t, 0, 0, 1.0, wrk->Ke+t, 0, 0, wrk->Ke+t, 0, 0);
            blasfeo_dtrsm_rltn(nx[t], rho, 1.0, wrk->Lee+t, 0, 0, wrk->Ke+t, 0, 0, wrk->Ke+t, 0, 0);
        }

        if (t==0) break;
        /*
            Update P = Q + A'PA - Ku Ku' + Keta Keta'
        */
        blasfeo_dgemm_nt(nx[t], nx[t+1], nx[t+1], 1.0, qp->BAwt+t, nu[t], 0, wrk->P+t, 0, 0, 0.0, tmp1, 0, 0, tmp1, 0, 0);
        // POSSIBLE ALIGNMENT ISSUE WITH BLASFEO (On Mac, to be verified)
        // When the A and Q operands below are not aligned, reversing the
        // A, tmp1=A'P order creates a correctness issue.
        blasfeo_dsyrk_ln(nx[t], nx[t+1], 1.0,
                         qp->BAwt+t, nu[t], 0, tmp1, 0, 0,
                         1.0, qp->RSQrq+t, nu[t], nu[t],
                         wrk->P+t-1, 0, 0);
        blasfeo_dsyrk_ln(nx[t], nu[t], -1.0, wrk->Ku+t, 0, 0, wrk->Ku+t, 0, 0, 1.0, wrk->P+t-1, 0, 0, wrk->P+t-1, 0, 0);
        if (rho) blasfeo_dsyrk_ln(nx[t], rho, 1.0, wrk->Ke+t, 0, 0, wrk->Ke+t, 0, 0, 1.0, wrk->P+t-1, 0, 0, wrk->P+t-1, 0, 0);
        blasfeo_dtrtr_l(nx[t], wrk->P+t-1, 0, 0, wrk->P+t-1, 0, 0);
    }
    // Set number of affine constraints on x0.
    wrk->nH0 = neq_x0;
    // Compute prexif sum of equality constraints.
    daocp_compute_prefix_sum(wrk->crho, wrk->rho, N);
    wrk->neta = wrk->crho[N-1] + wrk->rho[N-1];
    // Initialize eta pointers;
    f64* memory = wrk->eta[0];
    for (u32 t=1; t<N; ++t) wrk->eta[t] = memory + wrk->crho[t];
}

void daocp_solve_lqr(daocp_workspace* wrk, daocp_qp* qp) {
    u32 N = qp->dims.N;
    u32* nu = qp->dims.nu;
    u32* nx = qp->dims.nx;
    u32* rho = wrk->rho;
    struct blasfeo_dvec* p = &wrk->costate0;
    struct blasfeo_dvec* ptmp = &wrk->costate1;

    // Initialize costate
    blasfeo_drowex(nx[N], 1.0, qp->RSQrq+N, nx[N], 0, p, 0);

    // Backward recursion
    for (i32 t=N-1; t>=0; t--) {
        /*
            Solve:
            [Luu 0; Lue Lee][du; deta] = [r+B'(p+Pw); -b]
        */
        blasfeo_drowex(nx[t+1], 1.0, &qp->BAwt[t], nu[t]+nx[t], 0, ptmp, 0);
        blasfeo_dsymv_l(nx[t+1], 1.0, wrk->P+t, 0, 0, ptmp, 0, 1.0, p, 0, p, 0);
        blasfeo_drowex(nu[t], 1.0, &qp->RSQrq[t], nu[t]+nx[t], 0, ptmp, 0);
        blasfeo_dgemv_n(nu[t], nx[t+1], 1.0, qp->BAwt+t, 0, 0, p, 0, 1.0, ptmp, 0, &wrk->ux_lqr[t], 0);
        blasfeo_dveccp(rho[t], &wrk->b[t], 0, &wrk->eta_lqr[t], 0);
        DAOCP_TRSVLQR(wrk->ux_lqr[t], wrk->eta_lqr[t], wrk->Luu+t, wrk->Lue+t, wrk->Lee+t, nu[t], rho[t]);

        if (t==0) break;
        /*
            Compute p = A' (p + Pw) - Ku du + Keta deta + q 
        */
        blasfeo_drowex(nx[t], 1.0, &qp->RSQrq[t], nu[t]+nx[t], nu[t], ptmp, 0);
        blasfeo_dgemv_n(nx[t], nx[t+1], 1.0, qp->BAwt+t, nu[t], 0, p, 0, 1.0, ptmp, 0, ptmp, 0);
        blasfeo_dgemv_n(nx[t], nu[t], -1.0, wrk->Ku+t, 0, 0, &wrk->ux_lqr[t], 0, 1.0, ptmp, 0, ptmp, 0);
        if (rho[t]) blasfeo_dgemv_n(nx[t], rho[t], 1.0, wrk->Ke+t, 0, 0, &wrk->eta_lqr[t], 0, 1.0, ptmp, 0, ptmp, 0);
        daocp_pointer_swap((unsigned char**)&p, (unsigned char**)&ptmp);
    }

    // Forward recursion
    memcpy(wrk->ux_lqr[0].pa+nu[0], qp->x0, nx[0]*sizeof(f64));
    for (u32 t=0; t<N; ++t) {
        /*
            [Luu' Lue'; 0 Lee'][u; eta] = [-Ku'x - du; Keta'x0 + deta]
        */
        blasfeo_dgemv_t(nx[t], nu[t], -1.0, wrk->Ku+t, 0, 0, &wrk->ux_lqr[t], nu[t], -1.0, &wrk->ux_lqr[t], 0, &wrk->ux_lqr[t], 0);
        if (rho[t]) blasfeo_dgemv_t(nx[t], rho[t], 1.0, wrk->Ke+t, 0, 0, &wrk->ux_lqr[t], nu[t], 1.0, &wrk->eta_lqr[t], 0, &wrk->eta_lqr[t], 0);
        DAOCP_TRSVLQR_T(wrk->ux_lqr[t], wrk->eta_lqr[t], wrk->Luu+t, wrk->Lue+t, wrk->Lee+t, nu[t], rho[t]);
        
        /*
            x = Ax + Bu + w
        */
        blasfeo_drowex(nx[t+1], 1.0, &qp->BAwt[t], nu[t]+nx[t], 0, &wrk->ux_lqr[t+1], nu[t+1]);
        blasfeo_dgemv_t(nu[t], nx[t+1], 1.0, qp->BAwt+t, 0, 0, &wrk->ux_lqr[t], 0, 1.0, &wrk->ux_lqr[t+1], nu[t+1], &wrk->ux_lqr[t+1], nu[t+1]);
        blasfeo_dgemv_t(nx[t], nx[t+1], 1.0, qp->BAwt+t, nu[t], 0, &wrk->ux_lqr[t], nu[t], 1.0, &wrk->ux_lqr[t+1], nu[t+1], &wrk->ux_lqr[t+1], nu[t+1]);
    }

    // Evaluate constraints and adjust right/left-hand sides.
    u32* nbx = qp->dims.nbx; u32* nbu = qp->dims.nbu;
    u32* ng = qp->dims.ng;
    u32** idxbx = qp->idxbx; u32** idxbu = qp->idxbu;
    daocp_constraint_type** types = wrk->contypes;
    f64** Cx = qp->Cx; f64** Cu = qp->Cu;
    struct blasfeo_dvec v;
    for (u32 t=0; t<=N; ++t) {
        // Evaluate input bounds
        for (u32 i=0; i<nbu[t]; ++i) {
            f64 uval = wrk->ux_lqr[t].pa[idxbu[t][i]];
            wrk->lbu_wrk[t][i] = qp->lbu[t][i] - uval;
            wrk->ubu_wrk[t][i] = qp->ubu[t][i] - uval;
        }
        // Evaluate state bounds
        for (u32 i=0; i<nbx[t]; ++i) {
            f64 xval = wrk->ux_lqr[t].pa[nu[t]+idxbx[t][i]];
            wrk->lbx_wrk[t][i] = qp->lbx[t][i] - xval;
            wrk->ubx_wrk[t][i] = qp->ubx[t][i] - xval;
        }

        // Evaluate general constraints
        for (u32 i=0; i<ng[t]; ++i) {
            f64 val = 0;
            if (types[t][i] == DAOCP_ONLY_U || types[t][i] == DAOCP_MIXED) {
                v.pa = Cu[t]+i*nu[t];
                val += blasfeo_ddot(nu[t], &v, 0, &wrk->ux_lqr[t], 0);
            }
            if (types[t][i] == DAOCP_ONLY_X || types[t][i] == DAOCP_MIXED) {
                v.pa = Cx[t]+i*nx[t];
                val += blasfeo_ddot(nx[t], &v, 0, &wrk->ux_lqr[t], nu[t]);
            }
            wrk->lg_wrk[t][i] = qp->cl[t][i] - val;
            wrk->ug_wrk[t][i] = qp->cu[t][i] - val;
        }
    }
}

void daocp_reconstruct_eqcon_dual(daocp_workspace* wrk, daocp_sol* sol) {
    /*
    The following algorithm reconstructs the dual solution for the original
    equality constraints.
    - Initialize mu = [eta[0]; 0], m[0]-vector.
    - Transform mu through transpose of rank-revealing J[0]: mu = J[0]' mu.
    - Output eta_sol[0] = mu[:ne[0]]. 
    - Set mu = [eta[1]; mu[ne[0]:]].
    - Iterate until stage N, where eta_sol[N] = mu[ne[N-1]:]. 
    */
    u32* ne = wrk->dims->ne;
    f64** eta = wrk->eta;
    struct blasfeo_dvec* eta_lqr = wrk->eta_lqr;
    f64* mu = wrk->GEtmp;
    u32 N = wrk->dims->N;
    f64** J = wrk->J;
    u32* m = wrk->m;
    u32* rho = wrk->rho;

    // Initialize mu
    memset(mu, 0, m[0]*sizeof(f64));
    
    for (u32 t=0; t<N; ++t) {
        // Copy eta + eta_lqr in mu
        for (u32 i=0; i<rho[t]; ++i)
            mu[i] = eta[t][i] + BLASFEO_DVECEL(eta_lqr+t, i);
        // Transform mu
        daocp_GE_transpose(J[t], mu, m[t], rho[t]);
        // Write first ne[t] components in the solution
        memcpy(sol->eta + wrk->crho[t], mu, ne[t]*sizeof(f64));
        // Forward remaining m[t]-ne[t] components
        if (t<N-1) {
            if (rho[t+1] > ne[t]) {
                for (u32 i=m[t]-1; i>=ne[t]; --i) mu[rho[t+1]+i-ne[t]] = mu[i];
            } else if (rho[t+1] < ne[t]) {
                for (u32 i=ne[t]; i<m[t]; ++i) mu[rho[t+1]+i-ne[t]] = mu[i];
            }
        }
    }
    memcpy(sol->eta + wrk->crho[N-1]+rho[N-1], mu + ne[N-1], m[N]*sizeof(f64));
}