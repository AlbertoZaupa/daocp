import numpy as np


def _stage(value, t, stage_ndim):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return np.asarray(value[t])
    value = np.asarray(value)
    return value if value.ndim == stage_ndim else value[t]


def check_primal_feasibility(res, x0, A, B, w, D, C, du, dl, cu, cl,
                             Deq=None, Ceq=None, deq=None, ceq=None, tol=1e-6):
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
        dynamics_residual = (
            x[t] - _stage(A, t, 2) @ previous_x
            - _stage(B, t, 2) @ u[t] - _stage(w, t, 1)
        )
        equality_violation = max(
            equality_violation,
            np.max(np.abs(dynamics_residual), initial=0.0),
        )

        if Ceq is not None:
            residual = _stage(Ceq, t, 2) @ u[t] - _stage(ceq, t, 1)
            equality_violation = max(
                equality_violation, np.max(np.abs(residual), initial=0.0)
            )
        if Deq is not None:
            residual = _stage(Deq, t, 2) @ x[t] - _stage(deq, t, 1)
            equality_violation = max(
                equality_violation, np.max(np.abs(residual), initial=0.0)
            )
        if C is not None:
            image = _stage(C, t, 2) @ u[t]
            upper_residual = image - _stage(cu, t, 1)
            lower_residual = _stage(cl, t, 1) - image
            inequality_violation = max(
                inequality_violation,
                np.max(upper_residual, initial=0.0),
                np.max(lower_residual, initial=0.0),
            )
        if D is not None:
            image = _stage(D, t, 2) @ x[t]
            upper_residual = image - _stage(du, t, 1)
            lower_residual = _stage(dl, t, 1) - image
            inequality_violation = max(
                inequality_violation,
                np.max(upper_residual, initial=0.0),
                np.max(lower_residual, initial=0.0),
            )
        previous_x = x[t]

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
