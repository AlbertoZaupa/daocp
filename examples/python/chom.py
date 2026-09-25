# Copyright (c) 2026 Alberto Zaupa
# SPDX-License-Identifier: MIT
# See LICENSE in the project root for license information.

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "lib"), str(root)]
from examples.python.chom_model import *    # Chain of Masses model loaded from here
from examples.python.utils import *
import daocp

# CHAIN OF 1-D MASSES.
# First six states are positions, others are velocities.
# We can apply a force to 3 pairs of masses.

N = 30                                      # Horizon length
R = np.eye(nu)                              # R matrix
Q = np.zeros((nx, nx))                      # Q matrix
lbx = np.hstack([                           # state lower bounds
    -4*np.ones(6), 
    -np.inf*np.ones(6)]
)
ubx = -lbx                                  # state upper bounds (symmetric)
lbu = -0.5*np.ones(nu)                      # input lower bounds
ubu = -lbu                                  # input upper bounds (symmetric)

# We add a terminal equality constraint: x[N] = xT.
# This can be expressed as a state-bound with equal upper and lower components.
# The solver will detect that this is really an equality constraint, and deal with
# it efficiently.
lbx = [lbx.copy() for t in range(N-1)]      
ubx = [ubx.copy() for t in range(N-1)]
lbx.append(xT)
ubx.append(xT)

def solve_ocp():
    """
        Solve a single OCP.
    """
    # Instantiate the solver
    solver = daocp.OCPsolver(
        A,      # A matrix. Can also be list [A0, A1, ..., A_{N-1}]
        B,      # B matrix. Can also be list [B0, B1, ..., B_{N-1}]
        None,   # w[k] affine terms. None => w[k] = 0
        Q,      # Q matrix. Here terminal cost == stage cost. Can also be a list
        R,      # R matrix. Can also be list [R0, R1, ..., R_{N-1}]
        None,   # S matrix (mixing quadratic term). None => S[k] = 0
        None,   # q[k] state-linear terms. None => q[k] = 0
        None,   # r[k] input-linear terms. None => r[k] = 0
        lbx,    # state lower-bounds. If None, lbx[k] = -INF
        ubx,    # state upper-bounds. If None, ubx[k] = INF
        lbu,    # input lower-bounds. If None, lbu[k] = -INF
        ubu,    # input upper bounds. If None, ubu[k] = INF
        None,   # C[k] = [Cu[k] Cx[k]]. If None, no mixed constraints
        None,   # cl[k] lower bounds for mixed constraints
        None,   # cu[k] upper bounds for mixed constraints
        x0,     # intial state (set to zero if unknown at setup)
        N,      # horizon length
        nx,     # number of states
        nu,     # number of controls
        greedy=False    # constraint selection strategy. Greedy=True good for simple prob.
    )

    # Print setup time (excludes problem data validation)
    print(f"Setup time: {solver.setup_time:.4f} us.")

    # Solve the problem.
    result = solver.solve()
    print(f"Solve time: {result.info.solve_time:.4f} us.")
    print(f"Iterations: {result.info.iter}")
    print(f"Solver status: {result.info.status}")

if __name__ == '__main__':
    solve_ocp()