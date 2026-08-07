import numpy as np


def check_primal_feasibility(res, x0, A, B, w, D, C, d, c, tol=1e-6):
    """Assert primal feasibility and return the largest constraint violation."""
    x = np.asarray(res.x)
    u = np.asarray(res.u)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(u)), (
        "primal solution contains non-finite values"
    )

    dynamics_residual = x[1:] - (x[:-1] @ A.T + u @ B.T + w)
    equality_violation = max(
        np.max(np.abs(x[0] - x0), initial=0.0),
        np.max(np.abs(dynamics_residual), initial=0.0),
    )

    input_residual = u @ C.T - c
    state_residual = x[1:] @ D.T - d
    inequality_violation = max(
        np.max(input_residual, initial=0.0),
        np.max(state_residual, initial=0.0),
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
