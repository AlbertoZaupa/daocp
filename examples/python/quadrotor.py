import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "lib"), str(root)]

from examples.python.utils import *
from examples.python.quadrotor_model import *
from daocp import *

# Tracking MPC for the quadrotor of Sun et al., "A Comparative Study of
# Nonlinear MPC and Differential-Flatness-Based Control for Quadrotor Agile
# Flight" (T-RO 2022): 13 states, 4 rotor thrusts bounded by what the platform
# delivers, N = 20 nodes over a 1 s horizon, solved at 100 Hz with the cost of
# Table I. The reference is one of the dynamically feasible circles of its
# simulation study, flown at 10 m/s. Such a circle is a steady state of the
# quadrotor: model, cost and bounds are all invariant under rotation about the
# vertical axis, so in coordinates that rotate with the reference the problem is
# time invariant. The model is linearized about it once, and the controller then
# solves that same QP at every step with only its initial state updated.
data = np.load('examples/data/quadrotor.npz')
A = data['A']
B = data['B']
w = data['w']
x_bar = data['x_bar']       # the reference, at zero phase
u_bar = data['u_bar']
rate = float(data['rate'])  # the phase of the reference advances at this rate
x0 = data['x0']
drag = 2.0                  # the simulated vehicle has twice the drag of the
                            # model, the `+100% Drag` case of Table IV
K = 400                     # steps of the simulated flight
max_iter = 1000
nreps = 200

# State weight of eq. (11), acting on the quaternion error of eq. (12), and the
# rotor thrust and body rate limits, all relative to the reference.
M = q_error_jacobian(x_bar[6:10])
Q = np.zeros((nx, nx))
Q[:3, :3] = np.diag(Qp)
Q[3:6, 3:6] = np.diag(Qv)
Q[6:10, 6:10] = M.T @ np.diag(Qq) @ M
Q[10:, 10:] = np.diag(Qw)
R = np.diag(Qu)
lbx = np.full(nx, -np.inf)
ubx = np.full(nx, np.inf)
lbx[10:] = -w_max - x_bar[10:]
ubx[10:] = w_max - x_bar[10:]
lbu = u_min - u_bar
ubu = u_max - u_bar

def deviation(state, phase):
    """State error, rotated back to the frame the reference is at rest in."""
    z = yaw_rotate(state, -phase)
    if z[6:10] @ x_bar[6:10] < 0:
        z[6:10] = -z[6:10]      # a quaternion and its negative are one attitude
    return z - x_bar

def make_solver(dx0):
    return OCPsolver(A, B, w, Q, R, None, None, None, lbx, ubx, lbu, ubu,
                        None, None, None,
                        dx0, N, nx, nu, max_iter)

def solve_ocp():
    # The first step of the flight, where the quadrotor still hovers at the
    # center of the circle and the rotors saturate to catch the reference.
    dx0 = deviation(x0, 0.0)
    solver = make_solver(dx0)
    print(f"Setup time: {solver.setup_time:.3e} us")
    res = solver.solve()
    assert res.info.status == "SOLVED", res.info.status
    primal_violation = check_primal_feasibility(
        res, dx0, A, B, w, lbx, ubx, lbu, ubu,
        None, None, None
    )
    print(f"Maximum primal feasibility violation: {primal_violation:.3e}")
    print(f"Iterations until convergence: {res.info.iter}")
    print(f"Solve time: {res.info.solve_time:.3f} us")

def time_ocp():
    dx0 = deviation(x0, 0.0)
    total = 0
    for i in range(nreps):
        solver = make_solver(dx0)
        res = solver.solve()
        total += res.info.solve_time
    print(f"Average solve time: {(total/nreps):3f} us")

def solve_mpc():
    # One solver for the whole flight: every step only updates the state it
    # starts from, and keeps the working set of the previous solution.
    solver = make_solver(deviation(x0, 0.0))
    state = x0.copy()
    error = np.zeros(K)
    iters = np.zeros(K)
    times = np.zeros(K)
    for k in range(K):
        phase = rate * k * control_dt
        solver.update(deviation(state, phase))
        res = solver.solve()
        assert res.info.status == "SOLVED", res.info.status
        error[k] = np.linalg.norm(state[:3] - yaw_rotate(x_bar, phase)[:3])
        iters[k] = res.info.iters
        times[k] = res.info.solve_time
        print(f"Timestep {k}. Iters: {res.info.iters}. Solve time: {res.info.solve_time:.3f} us.")
        print(f"Position error: {error[k]:.3f} m")
        # The simulated vehicle carries the drag the model does not know about.
        state = simulate(state, u_bar + res.u[0], drag=drag)
        print()
    print(f"Position tracking RMSE over the last second: "
          f"{np.sqrt(np.mean(error[-100:]**2)):.3f} m")
    print(f"Iterations: mean {iters.mean():.1f}, max {iters.max():.0f}")
    print(f"Solve time: mean {times.mean():.3f} us, max {times.max():.3f} us")

if __name__ == '__main__':
    solve_ocp()
