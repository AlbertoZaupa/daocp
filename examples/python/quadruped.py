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
Deq = data["C_pf"][:, 32:]
deq = np.zeros(12)
Cu = np.block([
    [np.eye(20), np.zeros((20, 12))],
    [data["C_nf"][:, :32]]
])
lbu = np.hstack([data["l"][-20:], data["l_nf"]])
C = Cu
cl = lbu
cu = np.hstack([data["u"][-20:], np.full(len(data["l_nf"]), np.inf)])
dx0 = data["dX"][:52]
q = np.zeros(nx)
r = np.zeros(nu)
lN = lambda a : [a.copy() for i in range(N)]
max_iter = 1000
Ql = lN(Q)
Ql[-1] = P.copy()
nreps = 200

def solve_ocp():
    solver = OCPsolver(A, B, w, Ql, R, S,
                        q, r, None, C, None, None, cu, cl,
                        Deq, None, deq, None,
                        dx0, N, nx, nu, max_iter)
    res = solver.solve()
    assert res.info.status == "SOLVED", res.info.status
    primal_violation = check_primal_feasibility(
        res, dx0, A, B, w, None, C, None, None, cu, cl,
        Deq, None, deq, None
    )
    print(f"Maximum primal feasibility violation: {primal_violation:.3e}")
    print(f"Iterations until convergence: {res.info.iter}")
    print(f"Solve time: {res.info.solve_time:.3f} us")

def time_ocp():
    total = 0
    for i in range(nreps):
        solver = OCPsolver(A, B, w, Ql, R, S,
                        q, r, None, C, None, None, cu, cl,
                        Deq, None, deq, None,
                        dx0, N, nx, nu, max_iter)
        res = solver.solve()
        total += res.info.solve_time    
    print(f"Average solve time: {(total/nreps):3f} us.")

if __name__ == '__main__':
    solve_ocp()
