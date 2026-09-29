# Copyright (c) 2026 Alberto Zaupa
# SPDX-License-Identifier: MIT
# See LICENSE in the project root for license information.


# DOUBLE INTEGRATOR EXAMPLE

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "lib"), str(root)]
from examples.python.utils import *
from daocp import *

# DEFINING PROBLEM DATA
nx = 2                              # state dimension
nu = 1                              # control dimension
A = np.array([[1, 1], [0, 1]])      # A matrix
B = np.array([[0], [1]])            # B matrix
Q = np.eye(nx)                      # stage cost matrix for the state
R = np.eye(nu)                      # stage cost matrix for the control
P = dare(A, B, Q, R)                # terminal cost matrix
lbu = -np.ones(nu)                  # input lower bounds
ubu = np.ones(nu)                   # input upper bounds
lbx = np.array([-35, -30])          # state lower bounds
ubx = np.array([35, 30])            # state upper bounds
Zbx = np.array([100, np.inf])       # quadratic penalties for slacks associated to 
                                    # state bounds
zbx = np.array([100, 0.0])          # linear penalties for slacks associated to 
                                    # state bounds
x0 = np.array([20, 5])              # initial condition (infeasible)
N = 20                              # horizon length

# The solver can accept list of matrices to define time-varying data.
# Here we create Q_t = [Q, Q, ..., P]
lN = lambda a, stages=N: [a.copy() for i in range(stages)]
Ql = lN(Q, N + 1)
Ql[-1] = P.copy()

max_iter = 1000                     # maximum solver iterations

def solve_ocp(verbose=False):
    """
        Here we solve a single OCP for the double integrator.
    """
    # Instantiate the solver
    solver = OCPsolver(
            A=A, B=B, Q=Q, R=R, lbu=lbu, ubu=ubu, lbx=lbx, ubx=ubx,
            Zbx=Zbx, zbx=zbx, nx=nx, nu=nu, N=N, x0=x0
        )

    # Display the setup time (which excludes validation of problem data)
    print(f"Setup time: {solver.setup_time:.3e} us")

    # Solve
    res = solver.solve()
    assert res.info.status == "SOLVED"

    # Display optimal control input
    if verbose:
        print(f"Optimal control over horizon {N}:")
        for t in range(N):
            print(f"u[{t}] = {res.u[t]}")
        print()
        print(f"Final state:")
        print(f"x[{N}] = {res.x[-1]}")

    # Solve time and solver iterations
    print(f"Iterations until convergence: {res.info.iter}")
    print(f"Solve time: {res.info.solve_time:.3f} us")

if __name__ == '__main__':
    solve_ocp(True)
