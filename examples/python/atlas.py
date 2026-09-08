import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "lib"), str(root)]

from examples.python.utils import *
from daocp import *

nx = 58
nu = 29
N = 30
data = np.load('examples/data/atlas.npz')
A = data['A']
B = data['B']
Q = np.diag(data['Q'])
R = np.diag(data['R'])
P = dare(A, B, Q, R)
x_ref = data['x_ref']
u_ref = data['u_ref']
x0 = data['x0']
ub = data['d_u'] - u_ref
lb = -data['d_u'] - u_ref
lbu = lb
ubu = ub
dx0 = x0 - x_ref
lN = lambda a, stages=N: [a.copy() for i in range(stages)]
max_iter = 1000
Ql = lN(Q, N + 1)
Ql[-1] = P.copy()
nreps = 200

def solve_ocp():
    solver = OCPsolver(A, B, None, Ql, R, None,
                        None, None, None, None, lbu, ubu,
                        None, None, None,
                        dx0, N, nx, nu, max_iter)
    print(f"Setup time: {(solver.setup_time/1000):.3e} ms")
    res = solver.solve()
    assert res.info.status == "SOLVED", res.info.status
    primal_violation = check_primal_feasibility(
        res, dx0, A, B, None, None, None, lbu, ubu,
        None, None, None
    )
    print(f"Maximum primal feasibility violation: {primal_violation:.3e}")
    print(f"Iterations until convergence: {res.info.iter}")
    print(f"Solve time: {res.info.solve_time:.3f} us")

def time_ocp():
    total = 0
    for i in range(nreps):
        solver = OCPsolver(A, B, None, Ql, R, None,
                        None, None, None, None, lbu, ubu,
                        None, None, None,
                        dx0, N, nx, nu, max_iter)
        res = solver.solve()
        total += res.info.solve_time
    print(f"Average solve time: {(total/nreps):3f} us")

def solve_mpc():
    nruns = 20
    nsteps = 300
    timing = np.zeros(nsteps)
    for run in range(nruns):
        solver = OCPsolver(A, B, None, Ql, R, None,
                        None, None, None, None, lbu, ubu,
                        None, None, None,
                        dx0, N, nx, nu, max_iter)
        state = dx0.copy()
        for t in range(nsteps):
            res = solver.solve()
            assert res.info.status == "SOLVED", res.info.status
            if nruns == 1:
                print(f"Timestep {t}. Iters: {res.info.iters}. Solve time: {res.info.solve_time:.3f} us.")
            timing[t] += res.info.solve_time
            state = A @ state + B @ np.clip(res.u[0], lb, ub)
            solver.update(state)
            
    timing /= nruns
    print(f"Average solve time: {np.mean(timing):.4f} us.")
    print(f"Maximum solve time: {np.max(timing):.4f} us.")

if __name__ == '__main__':
    solve_mpc()
