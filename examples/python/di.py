import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "lib"), str(root)]

from examples.python.utils import *
from daocp import *

nx = 2
nu = 1
A = np.array([[1, 1], [0, 1]])
B = np.array([[0], [1]])
w = np.zeros(nx)
Q = np.eye(nx)
R = np.eye(nu)
P = dare(A, B, Q, R)
S = np.zeros((nu, nx))
q = np.zeros(nx)
r = np.zeros(nu)
C = np.array([[1], [-1]])
D = np.vstack([np.eye(nx), -np.eye(nx)])
c = np.ones(2*nu)
d = np.array([35, 30, 35, 30])
x0 = np.array([10, 5])
max_iter = 1000
N = 20
lN = lambda a: [a.copy() for i in range(N)]
Ql = lN(Q)
Ql[-1] = P.copy()

def solve_ocp():
    solver = OCPsolver(lN(A), lN(B), lN(w), Ql, lN(R), lN(S),
                        lN(q), lN(r), lN(D), lN(C), lN(d), lN(c),
                        x0, N, nx, nu, max_iter)
    res = solver.solve()
    assert res.info.status == "SOLVED"
    primal_violation = check_primal_feasibility(
        res, x0, A, B, w, D, C, d, c
    )
    print(f"Optimal control over horizon {N}:")
    for t in range(N):
        print(f"u[{t}] = {res.u[t]}")

    print(f"Maximum primal feasibility violation: {primal_violation:.3e}")
    print(f"Iterations until convergence: {res.info.iter}")
    print(f"Solve time: {res.info.solve_time:.3f} us")

def solve_mpc():
    solver = OCPsolver(lN(A), lN(B), lN(w), Ql, lN(R), lN(S),
                        lN(q), lN(r), lN(D), lN(C), lN(d), lN(c),
                        x0, N, nx, nu, max_iter)
    state = x0.copy()
    for t in range(N):
        res = solver.solve()
        assert res.info.status == "SOLVED"
        print(f"Timestep {t}. Iters: {res.info.iters}. Solve time: {res.info.solve_time:.3f} us.")
        print(f"u[{t}] = {res.u[0]}")
        state = A @ state + B @ np.clip(res.u[0], -1, 1) + w
        solver.update(state)
        print()

    print(f"\nFinal state:\n{state}")
    
if __name__ == '__main__':
    #solve_ocp()
    solve_mpc()
