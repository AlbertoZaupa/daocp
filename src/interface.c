#include <internal.h>
#define DAOCP_MEMORY_ALIGNMENT 64

static char* daocp_align_memory(char* memory) {
    uintptr_t address = (uintptr_t) memory;
    address = (address + DAOCP_MEMORY_ALIGNMENT - 1)
            & ~((uintptr_t) DAOCP_MEMORY_ALIGNMENT - 1);
    return (char*) address;
}


static void daocp_get_totals(
    daocp_dims* dims, u32* nx_tot, u32* nu_tot, u32* nb_tot,
    u32* ng_tot, u32* ne_tot, u32* max_nx, u32* max_nu)
{
    u32 N = dims->N;

    *nx_tot = 0;
    *nu_tot = 0;
    *nb_tot = 0;
    *ng_tot = 0;
    *ne_tot = 0;
    *max_nx = 0;
    *max_nu = 0;

    for (u32 t = 0; t <= N; ++t) {
        *nx_tot += dims->nx[t];
        if (t>0 && t<N) *nb_tot += dims->nbu[t] + dims->nbx[t];
        else if (t>0) *nb_tot += dims->nbx[t];
        else if (t<N) *nb_tot += dims->nbu[t];
        *ng_tot += dims->ng[t];
        *ne_tot += dims->ne[t];
        *max_nx = DAOCP_MAX(*max_nx, dims->nx[t]);

        if (t < N) {
            *nu_tot += dims->nu[t];
            *max_nu = DAOCP_MAX(*max_nu, dims->nu[t]);
        }
    }
}

void daocp_args_set_default(daocp_args* args) {
    args->max_iter = 1000;
    args->selection = DAOCP_SELECT_GREEDY;
}

u32 daocp_qp_memsize(daocp_dims* dims) {
    u32 N = dims->N;
    u32 size = 0;

    // BLASFEO matrix descriptors.
    size += N * sizeof(struct blasfeo_dmat);       // BAwt
    size += (N + 1) * sizeof(struct blasfeo_dmat); // RSQrq

    // Stage pointer arrays.
    size += (11*N + 7) * sizeof(f64*);
    size += (2*N + 1) * sizeof(u32*);

    // Dense, non-BLASFEO data.
    size += (u32) dims->nx[0] * sizeof(f64); // x0
    for (u32 t = 0; t <= N; ++t) {
        if (t>0) size += dims->ng[t] * dims->nx[t] * sizeof(f64); // Cx
        if (t < N) size += dims->ng[t] * dims->nu[t] * sizeof(f64); // Cu
        size += dims->ne[t] * dims->nx[t] * sizeof(f64); // Dx
        if (t<N) size += dims->ne[t] * dims->nu[t] * sizeof(f64); // Du
        size += 2*dims->ng[t] * sizeof(f64);          // cl, cu
        size += dims->ne[t] * sizeof(f64);               // d
        if (t<N) size += 2*dims->nbu[t] * sizeof(f64);         // lbu, ubu
        if (t>0) size += 2*dims->nbx[t] * sizeof(f64);         // lbx, ubx
        if (t>0 && t<N) size += (dims->nbu[t] + dims->nbx[t]) * sizeof(u32);
        else if (t<N) size += dims->nbu[t] * sizeof(u32);
        else if (t>0) size += dims->nbx[t] * sizeof(u32);
    }

    // The BLASFEO backing store starts at a cache-line-aligned address.  The
    // caller's allocation need not itself be cache-line aligned.
    size += DAOCP_MEMORY_ALIGNMENT - 1;
    for (u32 t = 0; t < N; ++t)
        size += blasfeo_memsize_dmat(
            dims->nu[t] + dims->nx[t] + 1, dims->nx[t + 1]);
    for (u32 t = 0; t <= N; ++t)
        size += blasfeo_memsize_dmat(
            dims->nu[t] + dims->nx[t] + 1,
            dims->nu[t] + dims->nx[t]);

    return size;
}

void daocp_qp_memory_assign(daocp_dims* dims, daocp_qp* qp, void* memory) {
    u32 N = dims->N;
    char* c_ptr = (char*) memory;

    qp->dims = *dims;

    qp->BAwt = (struct blasfeo_dmat*) c_ptr; c_ptr += N*sizeof(struct blasfeo_dmat);
    qp->RSQrq = (struct blasfeo_dmat*) c_ptr; c_ptr += (N+1)*sizeof(struct blasfeo_dmat);

    qp->Cx = (f64**) c_ptr; c_ptr += (N+1) * sizeof(f64*);
    qp->Cu = (f64**) c_ptr; c_ptr += N*sizeof(f64*);
    qp->Dx = (f64**) c_ptr; c_ptr += (N+1) * sizeof(f64*);
    qp->Du = (f64**) c_ptr; c_ptr += N * sizeof(f64*);
    qp->cl = (f64**) c_ptr; c_ptr += (N+1) * sizeof(f64*);
    qp->cu = (f64**) c_ptr; c_ptr += (N+1) * sizeof(f64*);
    qp->d = (f64**) c_ptr; c_ptr += (N+1) * sizeof(f64*);
    qp->lbu = (f64**) c_ptr; c_ptr += N * sizeof(f64*);
    qp->ubu = (f64**) c_ptr; c_ptr += N * sizeof(f64*);
    qp->lbx = (f64**) c_ptr; c_ptr += (N+1) * sizeof(f64*);
    qp->ubx = (f64**) c_ptr; c_ptr += (N+1) * sizeof(f64*);
    qp->idxbu = (u32**) c_ptr; c_ptr += N * sizeof(u32*);
    qp->idxbx = (u32**) c_ptr; c_ptr += (N + 1) * sizeof(u32*);

    qp->x0 = (f64*) c_ptr; c_ptr += dims->nx[0] * sizeof(f64);

    qp->Cx[0] = 0;
    for (u32 t = 1; t <= N; ++t) {
        qp->Cx[t] = (f64*) c_ptr;
        c_ptr += dims->ng[t] * dims->nx[t] * sizeof(f64);
    }
    for (u32 t = 0; t < N; ++t) {
        qp->Cu[t] = (f64*) c_ptr;
        c_ptr += dims->ng[t] * dims->nu[t] * sizeof(f64);
    }
    for (u32 t = 0; t <= N; ++t) {
        qp->Dx[t] = (f64*) c_ptr;
        c_ptr += dims->ne[t] * dims->nx[t] * sizeof(f64);
    }
    for (u32 t = 0; t < N; ++t) {
        qp->Du[t] = (f64*) c_ptr;
        c_ptr += dims->ne[t] * dims->nu[t] * sizeof(f64);
    }
    for (u32 t = 0; t <= N; ++t) {
        qp->cl[t] = (f64*) c_ptr;
        c_ptr += dims->ng[t] * sizeof(f64);
    }
    for (u32 t = 0; t <= N; ++t) {
        qp->cu[t] = (f64*) c_ptr;
        c_ptr += dims->ng[t] * sizeof(f64);
    }
    for (u32 t = 0; t <= N; ++t) {
        qp->d[t] = (f64*) c_ptr;
        c_ptr += dims->ne[t] * sizeof(f64);
    }
    for (u32 t = 0; t < N; ++t) {
        qp->lbu[t] = (f64*) c_ptr;
        c_ptr += dims->nbu[t] * sizeof(f64);
    }
    for (u32 t = 0; t < N; ++t) {
        qp->ubu[t] = (f64*) c_ptr;
        c_ptr += dims->nbu[t] * sizeof(f64);
    }
    qp->lbx[0] = 0;
    for (u32 t = 1; t <= N; ++t) {
        qp->lbx[t] = (f64*) c_ptr;
        c_ptr += (u32) dims->nbx[t] * sizeof(f64);
    }
    qp->ubx[0] = 0;
    for (u32 t = 1; t <= N; ++t) {
        qp->ubx[t] = (f64*) c_ptr;
        c_ptr += dims->nbx[t] * sizeof(f64);
    }
    for (u32 t = 0; t < N; ++t) {
        qp->idxbu[t] = (u32*) c_ptr;
        c_ptr += dims->nbu[t] * sizeof(u32);
    }
    qp->idxbx[0] = 0;
    for (u32 t = 1; t <= N; ++t) {
        qp->idxbx[t] = (u32*) c_ptr;
        c_ptr += (u32) dims->nbx[t] * sizeof(u32);
    }

    c_ptr = daocp_align_memory(c_ptr);
    for (u32 t = 0; t < N; ++t) {
        u32 rows = dims->nu[t] + dims->nx[t] + 1;
        blasfeo_create_dmat(rows, dims->nx[t + 1], qp->BAwt + t, c_ptr);
        c_ptr += blasfeo_memsize_dmat(rows, dims->nx[t + 1]);
    }
    for (u32 t = 0; t <= N; ++t) {
        u32 nv = dims->nu[t] + dims->nx[t];
        blasfeo_create_dmat(nv + 1, nv, qp->RSQrq + t, c_ptr);
        c_ptr += blasfeo_memsize_dmat(nv + 1, nv);
    }
}

u32 daocp_workspace_memsize(daocp_dims* dims) {
    u32 N = dims->N;
    u32 nx_tot, nu_tot, nb_tot, ng_tot, ne_tot, max_nx, max_nu;
    daocp_get_totals(
        dims, &nx_tot, &nu_tot, &nb_tot, &ng_tot, &ne_tot,
        &max_nx, &max_nu);
    u32 nin = nb_tot + ng_tot;
    u32 W_stride = DAOCP_MIN(nu_tot, nin) + 1;
    u32 max_nx_nu = DAOCP_MAX(max_nx, max_nu);
    u32 size = sizeof(daocp_workspace);

    // BLASFEO descriptors.
    size += 6 * N * sizeof(struct blasfeo_dmat);
    size += (3 * N + 1) * sizeof(struct blasfeo_dvec);

    // u, x, eta, six bound arrays, constraint types and active status.
    size += N * sizeof(f64*);
    size += (N + 1) * sizeof(f64*);
    size += N * sizeof(f64*);
    size += 4 * (N + 1) * sizeof(f64*);
    size += 2 * N * sizeof(f64*);
    size += (N + 1) * sizeof(daocp_constraint_type*);
    size += (N + 1) * sizeof(u32*);

    // Dense workspace data.
    size += (2 * nu_tot + nx_tot - dims->nx[0]) * sizeof(f64); // u, eta, x
    size += 2 * (nb_tot + ng_tot) * sizeof(f64);
    size += 3 * W_stride * sizeof(f64); // xi, p, dual_linear
    size += (W_stride + 1) * W_stride * sizeof(f64); // Ld
    size += 2 * W_stride * nu_tot * sizeof(f64); // Mu, Me
    size += ne_tot * max_nx * sizeof(f64); // H
    size += 2 * ne_tot * sizeof(f64); // h, tmp1
    size += (ne_tot + 1) * (max_nx + max_nu + 1) * sizeof(f64); // GEtmp
    size += max_nx * max_nx_nu * sizeof(f64); // ABtmp

    size += ng_tot * sizeof(daocp_constraint_type);
    size += nin * sizeof(u32); // active constraint status
    size += 3 * N * sizeof(u32); // cnu, rho, crho
    size += W_stride * sizeof(u32); // xi_sign
    size += W_stride * sizeof(daocp_constraint); // xi2con

    size += DAOCP_MEMORY_ALIGNMENT - 1;

    // BLASFEO matrix backing store.
    for (u32 t = 0; t < N; ++t) {
        size += blasfeo_memsize_dmat(dims->nx[t + 1U], dims->nx[t + 1U]);
        size += 2 * blasfeo_memsize_dmat(dims->nx[t], dims->nu[t]);
        size += 3 * blasfeo_memsize_dmat(dims->nu[t], dims->nu[t]);
    }
    size += blasfeo_memsize_dmat(max_nx_nu, max_nx); // tmp2

    // BLASFEO vector backing store.
    for (u32 t = 0; t < N; ++t) {
        size += blasfeo_memsize_dvec(dims->nu[t] + dims->nx[t]);
        size += 2 * blasfeo_memsize_dvec(dims->nu[t]);
    }
    size += blasfeo_memsize_dvec(dims->nx[N]);
    size += 2 * blasfeo_memsize_dvec(max_nx);

    return size;
}

static daocp_constraint_type daocp_constraint_type_at(
    daocp_dims* dims, daocp_qp* qp, u32 t, u32 row)
{
    if (t == 0) return DAOCP_ONLY_U;
    if (t == dims->N) return DAOCP_ONLY_X;

    u32 has_u = 0;
    u32 has_x = 0;

    if (t < dims->N) {
        f64* Cu = qp->Cu[t] + (size_t) row * dims->nu[t];
        for (u32 j = 0; j < dims->nu[t]; ++j)
            if (DAOCP_ABS(Cu[j]) > DAOCP_ZERO_TOL) {
                has_u = 1;
                break;
            }
    }

    if (t > 0) {
        f64* Cx = qp->Cx[t] + (size_t) row * dims->nx[t];
        for (u32 j = 0; j < dims->nx[t]; ++j)
            if (DAOCP_ABS(Cx[j]) > DAOCP_ZERO_TOL) {
                has_x = 1;
                break;
            }
    }

    if (has_u && has_x)
        return DAOCP_MIXED;
    if (has_u)
        return DAOCP_ONLY_U;
    return DAOCP_ONLY_X;
}

void daocp_workspace_memory_assign(daocp_dims* dims, daocp_qp* qp, void* memory)
{
    u32 N = dims->N;
    u32 nx_tot, nu_tot, nb_tot, ng_tot, ne_tot, max_nx, max_nu;
    daocp_get_totals(
        dims, &nx_tot, &nu_tot, &nb_tot, &ng_tot, &ne_tot,
        &max_nx, &max_nu);
    u32 nin = nb_tot + ng_tot;
    u32 W_stride = DAOCP_MIN(nu_tot, nin) + 1U;
    u32 max_nx_nu = DAOCP_MAX(max_nx, max_nu);
    char* c_ptr = (char*) memory;
    daocp_workspace* wrk = (daocp_workspace*) c_ptr;

    memset(wrk, 0, sizeof(*wrk));
    c_ptr += sizeof(*wrk);

    wrk->P = (struct blasfeo_dmat*) c_ptr;
    c_ptr += N * sizeof(struct blasfeo_dmat);
    wrk->Ku = (struct blasfeo_dmat*) c_ptr;
    c_ptr += N * sizeof(struct blasfeo_dmat);
    wrk->Ke = (struct blasfeo_dmat*) c_ptr;
    c_ptr += N * sizeof(struct blasfeo_dmat);
    wrk->Luu = (struct blasfeo_dmat*) c_ptr;
    c_ptr += N * sizeof(struct blasfeo_dmat);
    wrk->Lue = (struct blasfeo_dmat*) c_ptr;
    c_ptr += N * sizeof(struct blasfeo_dmat);
    wrk->Lee = (struct blasfeo_dmat*) c_ptr;
    c_ptr += N * sizeof(struct blasfeo_dmat);
    wrk->ux_lqr = (struct blasfeo_dvec*) c_ptr;
    c_ptr += (N + 1) * sizeof(struct blasfeo_dvec);
    wrk->eta_lqr = (struct blasfeo_dvec*) c_ptr;
    c_ptr += N * sizeof(struct blasfeo_dvec);
    wrk->b = (struct blasfeo_dvec*) c_ptr;
    c_ptr += N * sizeof(struct blasfeo_dvec);

    wrk->u = (f64**) c_ptr;
    c_ptr += N * sizeof(f64*);
    wrk->x = (f64**) c_ptr;
    c_ptr += (N+1) * sizeof(f64*);
    wrk->eta = (f64**) c_ptr;
    c_ptr += N * sizeof(f64*);
    wrk->lbu_wrk = (f64**) c_ptr;
    c_ptr += N * sizeof(f64*);
    wrk->ubu_wrk = (f64**) c_ptr;
    c_ptr += N * sizeof(f64*);
    wrk->lbx_wrk = (f64**) c_ptr;
    c_ptr += (N + 1) * sizeof(f64*);
    wrk->ubx_wrk = (f64**) c_ptr;
    c_ptr += (N + 1) * sizeof(f64*);
    wrk->lg_wrk = (f64**) c_ptr;
    c_ptr += (N + 1) * sizeof(f64*);
    wrk->ug_wrk = (f64**) c_ptr;
    c_ptr += (N + 1) * sizeof(f64*);
    wrk->contypes = (daocp_constraint_type**) c_ptr;
    c_ptr += (N + 1) * sizeof(daocp_constraint_type*);
    wrk->as.constraint_status = (u32**) c_ptr;
    c_ptr += (N + 1) * sizeof(u32*);

    f64* u_memory = (f64*) c_ptr;
    c_ptr += nu_tot * sizeof(f64);
    f64* eta_memory = (f64*) c_ptr;
    c_ptr += nu_tot * sizeof(f64);
    f64* x_memory = (f64*) c_ptr;
    c_ptr += (nx_tot-dims->nx[0]) * sizeof(f64);

    u32 offset = 0;
    for (u32 t = 0; t < N; ++t) {
        wrk->u[t] = u_memory + offset;
        wrk->eta[t] = eta_memory + offset;
        offset += dims->nu[t];
    }
    offset = 0;
    wrk->x[0] = 0;
    for (u32 t = 1; t <= N; ++t) {
        wrk->x[t] = x_memory + offset;
        offset += dims->nx[t];
    }
    memset(u_memory, 0, (size_t) nu_tot * sizeof(f64));
    memset(eta_memory, 0, (size_t) nu_tot * sizeof(f64));
    memset(x_memory, 0, (size_t) (nx_tot-dims->nx[0]) * sizeof(f64));

#define DAOCP_ASSIGN_F64_ROWS(field, count, start, end) \
    do { \
        for (u32 t = start; t <= end; ++t) { \
            wrk->field[t] = (f64*) c_ptr; \
            c_ptr += (size_t) (count)[t] * sizeof(f64); \
        } \
    } while (0)

    DAOCP_ASSIGN_F64_ROWS(lbu_wrk, dims->nbu, 0, N-1);
    DAOCP_ASSIGN_F64_ROWS(ubu_wrk, dims->nbu, 0, N-1);
    wrk->lbx_wrk[0] = 0; wrk->ubx_wrk[0] = 0;
    DAOCP_ASSIGN_F64_ROWS(lbx_wrk, dims->nbx, 1, N);
    DAOCP_ASSIGN_F64_ROWS(ubx_wrk, dims->nbx, 1, N);
    DAOCP_ASSIGN_F64_ROWS(lg_wrk, dims->ng, 0, N);
    DAOCP_ASSIGN_F64_ROWS(ug_wrk, dims->ng, 0, N);

#undef DAOCP_ASSIGN_F64_ROWS

    wrk->xi = (f64*) c_ptr;
    c_ptr += (size_t) W_stride * sizeof(f64);
    wrk->p = (f64*) c_ptr;
    c_ptr += (size_t) W_stride * sizeof(f64);
    wrk->dual_linear = (f64*) c_ptr;
    c_ptr += (size_t) W_stride * sizeof(f64);
    wrk->Ld = (f64*) c_ptr;
    c_ptr += (size_t) (W_stride + 1U) * W_stride * sizeof(f64);
    wrk->Mu = (f64*) c_ptr;
    c_ptr += (size_t) W_stride * nu_tot * sizeof(f64);
    wrk->Me = (f64*) c_ptr;
    c_ptr += (size_t) W_stride * nu_tot * sizeof(f64);
    wrk->H = (f64*) c_ptr;
    c_ptr += (size_t) ne_tot * max_nx * sizeof(f64);
    wrk->h = (f64*) c_ptr;
    c_ptr += (size_t) ne_tot * sizeof(f64);
    wrk->tmp1 = (f64*) c_ptr;
    c_ptr += (size_t) ne_tot * sizeof(f64);
    wrk->GEtmp = (f64*) c_ptr;
    c_ptr += (size_t) (ne_tot + 1U) * (max_nx + max_nu + 1U) * sizeof(f64);
    wrk->ABtmp = (f64*) c_ptr;
    c_ptr += (size_t) max_nx * max_nx_nu * sizeof(f64);

    for (u32 t = 0; t <= N; ++t) {
        wrk->contypes[t] = (daocp_constraint_type*) c_ptr;
        c_ptr += (size_t) dims->ng[t] * sizeof(daocp_constraint_type);
        for (u32 i = 0; i < dims->ng[t]; ++i)
            wrk->contypes[t][i] = daocp_constraint_type_at(dims, qp, t, i);
    }

    for (u32 t = 0; t <= N; ++t) {
        u32 stage_nin = dims->nbu[t] + dims->nbx[t]*(t>0 ? 1 : 0) + dims->ng[t];
        wrk->as.constraint_status[t] = (u32*) c_ptr;
        memset(c_ptr, 0, (size_t) stage_nin * sizeof(u32));
        c_ptr += (size_t) stage_nin * sizeof(u32);
    }

    wrk->cnu = (u32*) c_ptr;
    c_ptr += (size_t) N * sizeof(u32);
    wrk->rho = (u32*) c_ptr;
    c_ptr += (size_t) N * sizeof(u32);
    wrk->crho = (u32*) c_ptr;
    c_ptr += (size_t) N * sizeof(u32);
    wrk->xi_sign = (u32*) c_ptr;
    c_ptr += (size_t) W_stride * sizeof(u32);
    wrk->as.xi2con = (daocp_constraint*) c_ptr;
    c_ptr += (size_t) W_stride * sizeof(daocp_constraint);

    offset = 0;
    for (u32 t = 0; t < N; ++t) {
        wrk->cnu[t] = offset;
        offset += dims->nu[t];
    }
    memset(wrk->rho, 0, (size_t) N * sizeof(u32));
    memset(wrk->crho, 0, (size_t) N * sizeof(u32));

    c_ptr = daocp_align_memory(c_ptr);
    for (u32 t = 0; t < N; ++t) {
        blasfeo_create_dmat(
            dims->nx[t + 1U], dims->nx[t + 1U], wrk->P + t, c_ptr);
        c_ptr += blasfeo_memsize_dmat(dims->nx[t + 1U], dims->nx[t + 1U]);

        blasfeo_create_dmat(dims->nx[t], dims->nu[t], wrk->Ku + t, c_ptr);
        c_ptr += blasfeo_memsize_dmat(dims->nx[t], dims->nu[t]);
        blasfeo_create_dmat(dims->nx[t], dims->nu[t], wrk->Ke + t, c_ptr);
        c_ptr += blasfeo_memsize_dmat(dims->nx[t], dims->nu[t]);

        blasfeo_create_dmat(dims->nu[t], dims->nu[t], wrk->Luu + t, c_ptr);
        c_ptr += blasfeo_memsize_dmat(dims->nu[t], dims->nu[t]);
        blasfeo_create_dmat(dims->nu[t], dims->nu[t], wrk->Lue + t, c_ptr);
        c_ptr += blasfeo_memsize_dmat(dims->nu[t], dims->nu[t]);
        blasfeo_create_dmat(dims->nu[t], dims->nu[t], wrk->Lee + t, c_ptr);
        c_ptr += blasfeo_memsize_dmat(dims->nu[t], dims->nu[t]);
    }
    blasfeo_create_dmat(max_nx_nu, max_nx, &wrk->tmp2, c_ptr);
    c_ptr += blasfeo_memsize_dmat(max_nx_nu, max_nx);

    for (u32 t = 0; t < N; ++t) {
        blasfeo_create_dvec(
            dims->nu[t] + dims->nx[t], wrk->ux_lqr + t, c_ptr);
        c_ptr += blasfeo_memsize_dvec(dims->nu[t] + dims->nx[t]);
        blasfeo_create_dvec(dims->nu[t], wrk->eta_lqr + t, c_ptr);
        c_ptr += blasfeo_memsize_dvec(dims->nu[t]);
        blasfeo_create_dvec(dims->nu[t], wrk->b + t, c_ptr);
        c_ptr += blasfeo_memsize_dvec(dims->nu[t]);
    }
    blasfeo_create_dvec(dims->nx[N], wrk->ux_lqr + N, c_ptr);
    c_ptr += blasfeo_memsize_dvec(dims->nx[N]);
    blasfeo_create_dvec(max_nx, &wrk->costate0, c_ptr);
    c_ptr += blasfeo_memsize_dvec(max_nx);
    blasfeo_create_dvec(max_nx, &wrk->costate1, c_ptr);

    wrk->dims = &qp->dims;
    wrk->nx_tot = nx_tot;
    wrk->nu_tot = nu_tot;
    wrk->nb_tot = nb_tot;
    wrk->ng_tot = ng_tot;
    wrk->W_stride = W_stride;
    wrk->as.n_active = 0;
    wrk->as.max_t = 0;
    wrk->singular = 0;
}

u32 daocp_sol_memsize(daocp_dims* dims)
{
    u32 N = dims->N;
    size_t size = (size_t) (N + 1) * sizeof(struct blasfeo_dvec);

    size += DAOCP_MEMORY_ALIGNMENT - 1;
    for (u32 t = 0; t <= N; ++t)
        size += blasfeo_memsize_dvec(dims->nu[t] + dims->nx[t]);

    return (u32) size;
}

void daocp_sol_memory_assign(daocp_dims* dims, daocp_sol* sol, void* memory)
{
    u32 N = dims->N;
    char* c_ptr = (char*) memory;

    sol->ux = (struct blasfeo_dvec*) c_ptr;
    c_ptr += (N + 1) * sizeof(struct blasfeo_dvec);
    c_ptr = daocp_align_memory(c_ptr);

    for (u32 t = 0; t <= N; ++t) {
        u32 nv = dims->nu[t] + dims->nx[t];
        blasfeo_create_dvec(nv, sol->ux + t, c_ptr);
        c_ptr += blasfeo_memsize_dvec(nv);
    }
}
