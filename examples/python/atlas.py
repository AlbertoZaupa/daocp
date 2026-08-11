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
C = np.vstack([np.eye(nu), -np.eye(nu)])
ub = data['d_u'] - u_ref
lb = -data['d_u'] - u_ref
c = np.hstack([ub, -lb])
dx0 = x0 - x_ref
q = np.zeros(nx)
r = np.zeros(nu)
lN = lambda a : [a.copy() for i in range(N)]
max_iter = 1000
Ql = lN(Q)
Ql[-1] = P.copy()

def solve_ocp():
    solver = OCPsolver(A, B, w, Ql, R, S,
                        q, r, None, C, None, c,
                        None, None, None, None,
                        dx0, N, nx, nu, max_iter)
    res = solver.solve()
    assert res.info.status == "SOLVED", res.info.status
    primal_violation = check_primal_feasibility(
        res, dx0, A, B, w, None, C, None, c, None, None, None, None
    )
    print(f"Maximum primal feasibility violation: {primal_violation:.3e}")
    print(f"Iterations until convergence: {res.info.iter}")
    print(f"Solve time: {res.info.solve_time:.3f} us")

def solve_mpc():
    solver = OCPsolver(A, B, w, Ql, R, S,
                        q, r, None, C, None, c,
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
