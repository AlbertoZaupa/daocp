# Copyright (c) 2026 Alberto Zaupa
# SPDX-License-Identifier: MIT
# See LICENSE in the project root for license information.

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
Q = data['Q']
R = data['R']
P = dare(A, B, Q, R)
Cx_eq = data["C_pf"][:, 32:]
c_eq = np.zeros(12)
lbu = np.full(nu, -np.inf)
ubu = np.full(nu, np.inf)
lbu[:20] = tau_low
ubu[:20] = tau_upp
Cu_ineq = data["C_nf"][:, :32]
c_l_ineq = data["l_nf"]
c_u_ineq = np.full(len(c_l_ineq), np.inf)
dx0 = data["dX"][:52]
lN = lambda a, stages=N: [a.copy() for i in range(stages)]
max_iter = 1000
Ql = lN(Q, N + 1)
Ql[-1] = P.copy()
nreps = 200

C = []
c_l = []
c_u = []
for t in range(N + 1):
    stage_Cu = []
    stage_Cx = []
    stage_cl = []
    stage_cu = []
    if t < N:
        stage_Cu.append(Cu_ineq)
        stage_Cx.append(np.zeros((len(c_l_ineq), nx)))
        stage_cl.append(c_l_ineq)
        stage_cu.append(c_u_ineq)
    if t > 0:
        stage_Cu.append(np.zeros((len(c_eq), nu if t < N else 0)))
        stage_Cx.append(Cx_eq)
        stage_cl.append(c_eq)
        stage_cu.append(c_eq)
    C.append([np.vstack(stage_Cu), np.vstack(stage_Cx)])
    c_l.append(np.hstack(stage_cl))
    c_u.append(np.hstack(stage_cu))

def solve_ocp():
    solver = OCPsolver(A, B, None, Ql, R, None,
                        None, None, None, None, lbu, ubu,
                        C, c_l, c_u,
                        dx0, N, nx, nu, max_iter)
    print(f"Setup time: {(solver.setup_time/1000):.3e} ms")
    res = solver.solve()
    assert res.info.status == "SOLVED", res.info.status
    primal_violation = check_primal_feasibility(
        res, dx0, A, B, None, None, None, lbu, ubu,
        C, c_l, c_u
    )
    print(f"Maximum primal feasibility violation: {primal_violation:.3e}")
    print(f"Iterations until convergence: {res.info.iter}")
    print(f"Solve time: {res.info.solve_time:.3f} us")

def time_ocp():
    total = 0
    for i in range(nreps):
        solver = OCPsolver(A, B, None, Ql, R, None,
                        None, None, None, None, lbu, ubu,
                        C, c_l, c_u,
                        dx0, N, nx, nu, max_iter)
        res = solver.solve()
        total += res.info.solve_time    
    print(f"Average solve time: {(total/nreps):3f} us.")

def solve_mpc():
    solver = OCPsolver(A, B, None, Ql, R, None,
                        None, None, None, None, lbu, ubu,
                        C, c_l, c_u,
                        dx0, N, nx, nu, max_iter)
    for t in range(300):
        state = dX[t*nx:(t+1)*nx]
        solver.update(state)
        res = solver.solve()
        assert res.info.status == "SOLVED", res.info.status
        print(f"Timestep {t}. Iters: {res.info.iters}. Solve time: {res.info.solve_time:.3f} us.")
        print()

if __name__ == '__main__':
    time_ocp()
