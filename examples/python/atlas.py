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
w = np.zeros(nx)
Q = np.diag(data['Q'])
S = np.zeros((nu, nx))
R = np.diag(data['R'])
P = dare(A, B, Q, R)
x_ref = data['x_ref']
u_ref = data['u_ref']
x0 = data['x0']
ub = data['d_u'] - u_ref
lb = -data['d_u'] - u_ref
idxbu = np.arange(nu)
lbu = lb
ubu = ub
dx0 = x0 - x_ref
q = np.zeros(nx)
r = np.zeros(nu)
lN = lambda a : [a.copy() for i in range(N)]
max_iter = 1000
Ql = lN(Q)
Ql[-1] = P.copy()
nreps = 200

def solve_ocp():
    solver = OCPsolver(A, B, w, Ql, R, S,
                        q, r, None, idxbu, None, None, lbu, ubu,
                        None, None, None, None, None, None,
                        None, None, None, None,
                        dx0, N, nx, nu, max_iter)
    res = solver.solve()
    assert res.info.status == "SOLVED", res.info.status
    primal_violation = check_primal_feasibility(
        res, dx0, A, B, w, None, idxbu, None, None, lbu, ubu,
        None, None, None, None, None, None,
        None, None, None, None
    )
    print(f"Maximum primal feasibility violation: {primal_violation:.3e}")
    print(f"Iterations until convergence: {res.info.iter}")
    print(f"Solve time: {res.info.solve_time:.3f} us")

def time_ocp():
    total = 0
    for i in range(nreps):
        solver = OCPsolver(A, B, w, Ql, R, S,
                        q, r, None, idxbu, None, None, lbu, ubu,
                        None, None, None, None, None, None,
                        None, None, None, None,
                        dx0, N, nx, nu, max_iter)
        res = solver.solve()
        total += res.info.solve_time
    print(f"Average solve time: {(total/nreps):3f} us")

def solve_mpc():
    solver = OCPsolver(A, B, w, Ql, R, S,
                        q, r, None, idxbu, None, None, lbu, ubu,
                        None, None, None, None, None, None,
                        None, None, None, None,
                        dx0, N, nx, nu, max_iter)
    state = dx0.copy()
    for t in range(300):
        res = solver.solve()
        assert res.info.status == "SOLVED", res.info.status
        print(f"Timestep {t}. Iters: {res.info.iters}. Solve time: {res.info.solve_time:.3f} us.")
        state = A @ state + B @ np.clip(res.u[0], lb, ub) + w
        solver.update(state)
        print()

    print(f"\nFinal state:\n{state}")

if __name__ == '__main__':
    solve_ocp()
