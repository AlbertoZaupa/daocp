# Copyright (c) 2026 Alberto Zaupa
# SPDX-License-Identifier: MIT
# See LICENSE in the project root for license information.

# QUADRUPED CONTROL PROBLEM
# Problem data from https://github.com/RoboticExplorationLab/ReLUQP.jl

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "lib"), str(root)]
from examples.python.utils import *
from daocp import *

nx = 52                                 # Number of states
nu = 32                                 # Number of controls (torques + normals)
N = 15                                  # Horizon
data = np.load('examples/data/quadruped.npz')
dX = data['dX']                         # State sequence (here we solve a sequence of fixed QPs)
tau_low = data['l'][-20:]               # torque lower bounds
tau_upp = data['u'][-20:]               # torque upper bounds
A = data['Ad']                          
B = data['Bd']
Q = data['Q']
R = data['R']
P = dare(A, B, Q, R)
Cx_eq = data["C_pf"][:, 32:]            # Pinned foot constraints (state-equalities)
c_eq = np.zeros(12)     
lbu = np.full(nu, -np.inf)
ubu = np.full(nu, np.inf)
lbu[:20] = tau_low                      # Torque lower bounds (normals only appear in mixed constraints)
ubu[:20] = tau_upp                      # Torque upper bounds
Cu_ineq = data["C_nf"][:, :32]          # Normal forces constraint matrix
c_l_ineq = data["l_nf"]                 # Lower bounds for normal forces
c_u_ineq = np.full(len(c_l_ineq), np.inf)
dx0 = data["dX"][:52]                   # Initial condition

# Q matrices are [Q, Q, ..., P]
lN = lambda a, stages=N: [a.copy() for i in range(stages)]
Ql = lN(Q, N + 1)
Ql[-1] = P.copy()

# Constructing mixed constraints (normal force + pinned foot)
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
    """
    Solve a single OCP
    """
    solver = OCPsolver(
        A,      # A matrix. Can also be list [A0, A1, ..., A_{N-1}]
        B,      # B matrix. Can also be list [B0, B1, ..., B_{N-1}]
        None,   # w[k] affine terms. None => w[k] = 0
        Ql,     # Q matrix. Here terminal cost == stage cost. Can also be a list
        R,      # R matrix. Can also be list [R0, R1, ..., R_{N-1}]
        None,   # S matrix (mixing quadratic term). None => S[k] = 0
        None,   # q[k] state-linear terms. None => q[k] = 0
        None,   # r[k] input-linear terms. None => r[k] = 0
        None,   # state lower-bounds. None => lbx[k] = -INF
        None,   # state upper-bounds. None => ubx[k] = INF
        lbu,    # input lower-bounds. If None, lbu[k] = -INF
        ubu,    # input upper bounds. If None, ubu[k] = INF
        C,      # C[k] = [Cu[k] Cx[k]].
        c_l,    # cl[k] lower bounds for mixed constraints
        c_u,    # cu[k] upper bounds for mixed constraints
        dx0,    # intial state (set to zero if unknown at setup)
        N,      # horizon length
        greedy=True    # constraint selection strategy. Greedy=True good for simple prob.
    )

    # Setup time (excluding problem data validation)
    print(f"Setup time: {(solver.setup_time/1000):.3e} ms")

    # Solve and display stats
    res = solver.solve()
    assert res.info.status == "SOLVED", res.info.status
    print(f"Iterations until convergence: {res.info.iter}")
    print(f"Solve time: {res.info.solve_time:.3f} us")

def time_ocp():
    """
    Time the OCP
    """
    total = 0
    nreps = 200
    for i in range(nreps):
        solver = OCPsolver(A=A, B=B, Q=Ql, R=R, lbu=lbu, ubu=ubu,
                            C=C, c_l=c_l, c_u=c_u, x0=dx0, N=N)
        res = solver.solve()
        total += res.info.solve_time    
    print(f"Average solve time: {(total/nreps):3f} us.")

def solve_mpc():
    """
    Receding horizon MPC controller
    """
    solver = OCPsolver(A=A, B=B, Q=Ql, R=R, lbu=lbu, ubu=ubu,
                        C=C, c_l=c_l, c_u=c_u, N=N)
    for t in range(300):
        # Fetch state from fixed sequence
        x = dX[t*nx:(t+1)*nx]

        # Update problem data
        solver.update(x)
        res = solver.solve()
        assert res.info.status == "SOLVED", res.info.status
        print(f"Timestep {t}. Iters: {res.info.iters}. Solve time: {res.info.solve_time:.3f} us.")
        print()

if __name__ == '__main__':
    solve_ocp()
