import numpy as np

def rk4(f, x, u, dt):
    k1 = dt * f(x, u)
    k2 = dt * f(x + k1/2, u)
    k3 = dt * f(x + k2/2, u)
    k4 = dt * f(x + k3, u)
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6

def _stage(value, t, stage_ndim):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return np.asarray(value[t])
    value = np.asarray(value)
    return value if value.ndim == stage_ndim else value[t]


def check_primal_feasibility(res, x0, A, B, w,
                             lbx, ubx, lbu, ubu,
                             C, c_l, c_u, tol=1e-6):
    """Assert primal feasibility and return the largest constraint violation."""
    x = np.asarray(res.x)
    u = np.asarray(res.u)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(u)), (
        "primal solution contains non-finite values"
    )

    equality_violation = 0.0
    inequality_violation = 0.0
    previous_x = np.asarray(x0)
    for t in range(len(u)):
        disturbance = 0 if w is None else _stage(w, t, 1)
        dynamics_residual = (
            x[t] - _stage(A, t, 2) @ previous_x
            - _stage(B, t, 2) @ u[t] - disturbance
        )
        equality_violation = max(
            equality_violation,
            np.max(np.abs(dynamics_residual), initial=0.0),
        )

        if lbu is not None or ubu is not None:
            lower = -np.inf if lbu is None else _stage(lbu, t, 1)
            upper = np.inf if ubu is None else _stage(ubu, t, 1)
            upper_residual = u[t] - upper
            lower_residual = lower - u[t]
            inequality_violation = max(
                inequality_violation,
                np.max(upper_residual, initial=0.0),
                np.max(lower_residual, initial=0.0),
            )
        if lbx is not None or ubx is not None:
            lower = -np.inf if lbx is None else _stage(lbx, t, 1)
            upper = np.inf if ubx is None else _stage(ubx, t, 1)
            upper_residual = x[t] - upper
            lower_residual = lower - x[t]
            inequality_violation = max(
                inequality_violation,
                np.max(upper_residual, initial=0.0),
                np.max(lower_residual, initial=0.0),
            )
        previous_x = x[t]

    if C is not None:
        states = [np.asarray(x0), *x]
        for t, (C_u, C_x) in enumerate(C):
            C_u = None if C_u is None else np.asarray(C_u)
            C_x = None if C_x is None else np.asarray(C_x)
            rows = C_u.shape[0] if C_u is not None else C_x.shape[0]
            image = np.zeros(rows)
            if C_u is not None and C_u.shape[1]:
                image += C_u @ u[t]
            if C_x is not None:
                image += C_x @ states[t]
            lower = -np.inf if c_l is None else _stage(c_l, t, 1)
            upper = np.inf if c_u is None else _stage(c_u, t, 1)
            inequality_violation = max(
                inequality_violation,
                np.max(image - upper, initial=0.0),
                np.max(lower - image, initial=0.0),
            )

    max_violation = max(equality_violation, inequality_violation)
    assert np.isfinite(max_violation) and max_violation <= tol, (
        f"primal solution violates the constraints by {max_violation:.3e} "
        f"(tolerance: {tol:.3e})"
    )
    return float(max_violation)


def dare(A, B, Q, R, max_iter=1000):
    nu = B.shape[1]
    Lp = np.linalg.cholesky(Q)
    for k in range(max_iter):
        ALp = A.T @ Lp
        BLp = B.T @ Lp
        t22 = Q + ALp @ ALp.T
        t12 = BLp @ ALp.T
        t11 = R + BLp @ BLp.T
        L = np.linalg.cholesky(np.block([
            [t11, t12],
            [t12.T, t22]
        ]))
        Lp_new = L[nu:, nu:]
        res = np.linalg.norm(Lp_new - Lp)
        Lp = Lp_new
        if res < 1e-9: break
    if k == max_iter: print("!!! DARE reached maximum number of iterations.")
    return Lp_new @ Lp_new.T
