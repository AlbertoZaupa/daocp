import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "lib"), str(root)]

from examples.python.utils import *
from daocp import *

nx = 52
nu = 32
N = 15
data = np.load('examples/data/quadruped.npz')
dX = data['dX']
tau_low = data['l'][-20:]
tau_upp = data['u'][-20:]
A = data['Ad']
B = data['Bd']
w = np.zeros(nx)
Q = data['Q']
S = np.zeros((nu, nx))
R = data['R']
P = dare(A, B, Q, R)
Dx = data["C_pf"][:, 32:]
Cu = np.block([
    [np.eye(20), np.zeros((20, 12))],
    [data["C_nf"][:, :32]]
])
lbu = np.hstack([data["l"][-20:], data["l_nf"]])
ubu = np.hstack([data["u"][-20:], np.inf*np.ones(20)])
C = np.vstack([Cu, -Cu])
D = np.vstack([Dx, -Dx])
c = np.hstack([ubu, -lbu])
d = np.zeros(24)
dx0 = data["dX"][:52]
q = np.zeros(nx)
r = np.zeros(nu)
lN = lambda a : [a.copy() for i in range(N)]
max_iter = 1000
Ql = lN(Q)
Ql[-1] = P.copy()

def solve_ocp():
    solver = OCPsolver(lN(A), lN(B), lN(w), Ql, lN(R), lN(S),
                        lN(q), lN(r), lN(D), lN(C), lN(d), lN(c),
                        dx0, N, nx, nu, max_iter)
    res = solver.solve()
    assert res.info.status == "SOLVED", res.info.status
    primal_violation = check_primal_feasibility(
        res, dx0, A, B, w, D, C, d, c
    )
    print(f"Maximum primal feasibility violation: {primal_violation:.3e}")
    print(f"Iterations until convergence: {res.info.iter}")
    print(f"Solve time: {res.info.solve_time:.3f} us")

if __name__ == '__main__':
    solve_ocp()
