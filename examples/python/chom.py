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
            A=A, B=B, Q=Q, R=R, lbx=lbx, ubx=ubx, lbu=lbu, ubu=ubu,
            N=N, x0=x0
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