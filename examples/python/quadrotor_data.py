"""Regenerates examples/data/quadrotor.npz. Run from the top-level directory.

Records the QP subproblems that the real-time iteration scheme of the NMPC of
Sun et al. (see examples/python/quadrotor_model.py) solves while tracking one
of the elliptical references of its simulation study at 100 Hz. Every step
linearizes the model along the trajectory predicted by the previous solution
and simulates the quadrotor with the first optimized rotor thrusts.
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "lib"), str(root)]

from examples.python.quadrotor_model import *
from daocp import *

# Horizontal circle of the reference grid of Sec. VI-A. Flying it takes 9.39 N
# per rotor while the platform delivers 8.5 N, so it is one of the dynamically
# infeasible references of Table III.
a_max = 50.0        # m/s^2
v_max = 10.0        # m/s
ellipticity = 1
vertical = False
K = 250             # control steps to record, at 100 Hz
max_iter = 1000
over = int(round(Ts / control_dt))
eps = 1e-6

def jacobians(xb, ub):
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))
    for i in range(nx):
        d = np.zeros(nx)
        d[i] = eps
        A[:, i] = (h(xb + d, ub) - h(xb - d, ub)) / (2 * eps)
    for i in range(nu):
        d = np.zeros(nu)
        d[i] = eps
        B[:, i] = (h(xb, ub + d) - h(xb, ub - d)) / (2 * eps)
    return A, B

def stage_cost(q_ref):
    M = q_error_jacobian(q_ref)
    Q = np.zeros((nx, nx))
    Q[:3, :3] = np.diag(Qp)
    Q[3:6, 3:6] = np.diag(Qv)
    Q[6:10, 6:10] = M.T @ np.diag(Qq) @ M
    Q[10:, 10:] = np.diag(Qw)
    return Q

def record():
    duration = (K + over * N + 2) * control_dt
    x_ref, u_ref = ellipse_trajectory(control_dt, a_max, v_max, ellipticity,
                                      duration, vertical=vertical)
    print(f"reference: {v_max} m/s on a {v_max**2/a_max:.2f} m ellipse, "
          f"rotor thrust up to {u_ref.max():.2f} N (limit {u_max} N)")

    # The trajectory the model is linearized about starts as the rollout of the
    # reference thrusts and is updated with every solution.
    state = x_ref[0].copy()
    ubar = np.clip(u_ref[:over * N:over], u_min, u_max).copy()
    xbar = np.zeros((N + 1, nx))
    xbar[0] = state
    for t in range(N):
        xbar[t + 1] = h(xbar[t], ubar[t])

    stored = {key: [] for key in ('A', 'B', 'w', 'xbar', 'ubar', 'dx0')}
    iters = np.zeros(K)
    error = np.zeros(K)
    saturated = np.zeros(K)
    for k in range(K):
        xr = x_ref[k:k + over * (N + 1):over]
        ur = u_ref[k:k + over * N:over]
        A = np.zeros((N, nx, nx))
        B = np.zeros((N, nx, nu))
        w = np.zeros((N, nx))
        for t in range(N):
            A[t], B[t] = jacobians(xbar[t], ubar[t])
            w[t] = h(xbar[t], ubar[t]) - xbar[t + 1]
        Ql = [stage_cost(xr[t, 6:10]) for t in range(N + 1)]
        q = np.array([Ql[t] @ (xbar[t] - xr[t]) for t in range(N + 1)])
        r = Qu * (ubar - ur)
        lbx = np.full((N, nx), -np.inf)
        ubx = np.full((N, nx), np.inf)
        lbx[:, 10:] = -w_max - xbar[1:, 10:]
        ubx[:, 10:] = w_max - xbar[1:, 10:]
        dx0 = state - xbar[0]

        solver = OCPsolver(A, B, w, Ql, np.diag(Qu), None, q, r,
                            lbx, ubx, u_min - ubar, u_max - ubar,
                            None, None, None,
                            dx0, N, nx, nu, max_iter)
        res = solver.solve()
        assert res.info.status == "SOLVED", (k, res.info.status)
        iters[k] = res.info.iters
        error[k] = np.linalg.norm(state[:3] - xr[0, :3])
        for key, value in (('A', A), ('B', B), ('w', w), ('xbar', xbar.copy()),
                           ('ubar', ubar.copy()), ('dx0', dx0)):
            stored[key].append(value)

        xbar = np.vstack([state, xbar[1:] + np.asarray(res.x)])
        ubar = ubar + np.asarray(res.u)
        saturated[k] = np.sum((ubar > u_max - 1e-9) | (ubar < u_min + 1e-9))
        state = simulate(state, ubar[0])

    print(f"iterations: mean {iters.mean():.1f}, max {iters.max():.0f}")
    print(f"saturated rotor thrusts: {saturated.mean():.1f} of {N*nu} per solution")
    print(f"position tracking RMSE: {np.sqrt(np.mean(error**2)):.3f} m")
    np.savez_compressed('examples/data/quadrotor.npz',
                        x_ref=x_ref, u_ref=u_ref, t0=np.int64(np.argmax(iters)),
                        **{key: np.asarray(value) for key, value in stored.items()})
    print("saved examples/data/quadrotor.npz")

if __name__ == '__main__':
    record()
