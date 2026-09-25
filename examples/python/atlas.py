# Copyright (c) 2026 Alberto Zaupa
# SPDX-License-Identifier: MIT
# See LICENSE in the project root for license information.

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "lib"), str(root)]
from examples.python.utils import *
from daocp import *

# STABILIZATION OF A HUMANOID ROBOT (linearized dynamics)
# Problem data from https://github.com/RoboticExplorationLab/ReLUQP.jl

nx = 58                                    # number of states 
nu = 29                                    # number of controls
N = 30                                     # horizon length
data = np.load('examples/data/atlas.npz')
A = data['A']                              
B = data['B']                                     
Q = np.diag(data['Q'])
R = np.diag(data['R'])
P = dare(A, B, Q, R)
x_ref = data['x_ref']                      # reference equilibrium
u_ref = data['u_ref']                      # corresponding input
x0 = data['x0']                            # initial state

# dx, du reformulation: dx = x - x_ref , du = u - u_ref
ub = data['d_u'] - u_ref                   # input upper bounds
lb = -data['d_u'] - u_ref                  # input lower bounds
lbu = lb            
ubu = ub
dx0 = x0 - x_ref                           # initial state for controller

# Q matrices are [Q, Q, ..., P]
lN = lambda a, stages=N: [a.copy() for i in range(stages)]
Ql = lN(Q, N + 1)
Ql[-1] = P.copy()

def solve_ocp():
    """
    Solve a single OCP problem
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
        None,   # C[k] = [Cu[k] Cx[k]]. If None, no mixed constraints
        None,   # cl[k] lower bounds for mixed constraints
        None,   # cu[k] upper bounds for mixed constraints
        dx0,    # intial state (set to zero if unknown at setup)
        N,      # horizon length
        nx,     # number of states
        nu,     # number of controls
        greedy=False    # constraint selection strategy. Greedy=True good for simple prob.
    )

    # Setup time (excluding problem validation)
    print(f"Setup time: {(solver.setup_time/1000):.3e} ms")

    # Solve and show stats
    res = solver.solve()
    assert res.info.status == "SOLVED", res.info.status
    print(f"Iterations until convergence: {res.info.iter}")
    print(f"Solve time: {res.info.solve_time:.3f} us")

def time_ocp():
    """
    Time single OCP solve
    """
    total = 0
    nreps = 200
    for i in range(nreps):
        solver = OCPsolver(A, B, None, Ql, R, None,
                        None, None, None, None, lbu, ubu,
                        None, None, None,
                        dx0, N, nx, nu, greedy=True)
        res = solver.solve()
        total += res.info.solve_time
    print(f"Average solve time: {(total/nreps):3f} us")

def solve_mpc():
    """
    Receding horizon MPC controller
    """
    nruns = 20      # We run the closed loop simulation multiple times to average stats
    nsteps = 300    # Simulation steps. dt=10ms => 3s simulation time.
    timing = np.zeros(nsteps)

    for run in range(nruns):
        # Instantiate solver
        solver = OCPsolver(A, B, None, Ql, R, None,
                        None, None, None, None, lbu, ubu,
                        None, None, None,
                        dx0, N, nx, nu, greedy=False)
        x = dx0.copy()
        for t in range(nsteps):
            res = solver.solve()
            assert res.info.status == "SOLVED", res.info.status

            # Print per-simulation stats if this is not a timing sweep
            if nruns == 1:
                print(f"Timestep {t}. Iters: {res.info.iters}. Solve time: {res.info.solve_time:.3f} us.")
            timing[t] += res.info.solve_time
            # Propagate linearized dynamics
            x = A @ x + B @ np.clip(res.u[0], lb, ub)
            # Update problem data
            solver.update(x)
            
    timing /= nruns
    print(f"Average solve time: {np.mean(timing):.4f} us.")
    print(f"Maximum solve time: {np.max(timing):.4f} us.")

if __name__ == '__main__':
    solve_mpc()
