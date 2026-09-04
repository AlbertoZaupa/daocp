import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "lib"), str(root)]

from examples.python.utils import *
from examples.python.quadrotor_model import *
from daocp import *

# Subproblems of the real-time iteration scheme of the quadrotor NMPC of Sun et
# al., "A Comparative Study of Nonlinear MPC and Differential-Flatness-Based
# Control for Quadrotor Agile Flight" (T-RO 2022), recorded over 2.5 s of flight
# at 100 Hz. The reference is one of the ellipses of its simulation study: a 2 m
# circle flown at 10 m/s (a_max = 50 m/s^2, ellipticity 1). Tracking it would
# take 9.39 N per rotor while the platform delivers 8.5 N, so the reference is
# dynamically infeasible and the thrust bounds are active over most of the
# horizon. Each step stores the trajectory the model was linearized about, and
# the cost of eq. (10) is rebuilt around it here.
data = np.load('examples/data/quadrotor.npz')
A = data['A']
B = data['B']
w = data['w']
xbar = data['xbar']
ubar = data['ubar']
dx0 = data['dx0']
x_ref = data['x_ref']
u_ref = data['u_ref']
t0 = int(data['t0'])
K = len(A)
over = int(round(Ts / control_dt))   # reference samples per optimization node
max_iter = 1000
nreps = 200

def stage_cost(q_ref):
    """State weight of eq. (11), acting on the quaternion error of eq. (12)."""
    M = q_error_jacobian(q_ref)
    Q = np.zeros((nx, nx))
    Q[:3, :3] = np.diag(Qp)
    Q[3:6, 3:6] = np.diag(Qv)
    Q[6:10, 6:10] = M.T @ np.diag(Qq) @ M
    Q[10:, 10:] = np.diag(Qw)
    return Q

def stage_bounds(k):
    """Rotor thrust and body rate limits, relative to the linearization point."""
    lbx = np.full((N, nx), -np.inf)
    ubx = np.full((N, nx), np.inf)
    lbx[:, 10:] = -w_max - xbar[k, 1:, 10:]
    ubx[:, 10:] = w_max - xbar[k, 1:, 10:]
    return lbx, ubx, u_min - ubar[k], u_max - ubar[k]

def subproblem(k):
    xr = x_ref[k:k + over * (N + 1):over]
    ur = u_ref[k:k + over * N:over]
    Ql = [stage_cost(xr[t, 6:10]) for t in range(N + 1)]
    q = np.array([Ql[t] @ (xbar[k, t] - xr[t]) for t in range(N + 1)])
    r = Qu * (ubar[k] - ur)
    lbx, ubx, lbu, ubu = stage_bounds(k)
    return OCPsolver(A[k], B[k], w[k], Ql, np.diag(Qu), None, q, r,
                        lbx, ubx, lbu, ubu,
                        None, None, None,
                        dx0[k], N, nx, nu, max_iter, greedy=False)

def solve_ocp():
    solver = subproblem(t0)
    print(f"Setup time: {solver.setup_time:.3e} us")
    res = solver.solve()
    assert res.info.status == "SOLVED", res.info.status
    lbx, ubx, lbu, ubu = stage_bounds(t0)
    primal_violation = check_primal_feasibility(
        res, dx0[t0], A[t0], B[t0], w[t0], lbx, ubx, lbu, ubu,
        None, None, None
    )
    print(f"Maximum primal feasibility violation: {primal_violation:.3e}")
    print(f"Iterations until convergence: {res.info.iter}")
    print(f"Solve time: {res.info.solve_time:.3f} us")

def time_ocp():
    total = 0
    for i in range(nreps):
        solver = subproblem(t0)
        res = solver.solve()
        total += res.info.solve_time
    print(f"Average solve time: {(total/nreps):3f} us")

def solve_mpc():
    error = np.zeros(K)
    iters = np.zeros(K)
    for k in range(K):
        solver = subproblem(k)
        res = solver.solve()
        assert res.info.status == "SOLVED", res.info.status
        iters[k] = res.info.iters
        error[k] = np.linalg.norm(xbar[k, 0, :3] + dx0[k][:3] - x_ref[k, :3])
        saturated = np.sum(np.asarray(res.u) + ubar[k] > u_max - 1e-9)
        print(f"Timestep {k}. Iters: {res.info.iters}. Solve time: {res.info.solve_time:.3f} us.")
        print(f"Position error: {error[k]:.3f} m. Rotors at full thrust: {saturated} of {N*nu}.")
        print()
    print(f"Position tracking RMSE: {np.sqrt(np.mean(error**2)):.3f} m")
    print(f"Iterations: mean {iters.mean():.1f}, max {iters.max():.0f}")

if __name__ == '__main__':
    solve_ocp()
