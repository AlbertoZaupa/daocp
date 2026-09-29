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
x0 = np.array([10, 5])              # initial condition (close to feasibility boundary)
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
        A,    # either a list [A0, A1, ...] or a single matrix for the whole horizon
        B,    # either a list [B0, B1, ...] or a single matrix for the whole horizon
        None, # w[k] affine term. None => w[k] = 0 
        Ql,   # either a list [Q0, Q1, ...] or a single matrix for the whole horizon
        R,    # either a list [R0, R1, ...] or a single matrix for the whole horizon
        None, # S[k] mixing quadratic terms. None => S[k] = 0
        None, # q[k] state-linear terms. None => q[k] = 0
        None, # r[k] input-linear terms. None => r[k] = 0
        lbx,  # state lower bounds. None => lbx[k] = -INF  
        ubx,  # state upper bounds. None => ubx[k] = INF
        lbu,  # input lower bounds. None => lbu[k] = -INF
        ubu,  # input upper bounds. None => ubu[k] = INF
        None, # C[k] = [Cu[k] Cx[k]] constraint matrices. None => no mixed constraints
        None, # cl[k] mixed constraints lower bounds. None if and only if C = None.
        None, # cu[k] mixed constraints upper bounds. None if and only if C = None.
        x0,   # Initial state (just set to zero if unknown at solver setup).
        N,    # Horizon length
        nx,   # number of state variables
        nu,   # number of input variables
        max_iter,      # maximum number of iterations
        greedy=False   # constraint selection heuristic. Simple problem? => greedy=True
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

    # Solve time and solver iterations
    print(f"Iterations until convergence: {res.info.iter}")
    print(f"Solve time: {res.info.solve_time:.3f} us")

def time_ocp():
    """
        Time the OCP solution
    """
    nreps = 1000
    total = 0
    for i in range(nreps):
        solver = OCPsolver(A, B, None, Ql, R, None,
                        None, None, lbx, ubx, lbu, ubu,
                        None, None, None,
                        x0, N, nx, nu, max_iter)
        res = solver.solve()
        total += res.info.solve_time
    print(f"Average solve time: {(total/nreps):3f} us.")

def solve_mpc(verbose=False):
    """
        Receding horizon setting: at every sample time we update the
        solver's x0 value.
    """
    solver = OCPsolver(A, B, None, Ql, R, None,
                        None, None, lbx, ubx, lbu, ubu,
                        None, None, None,
                        x0, N, nx, nu, max_iter)
    x = x0.copy()

    # Simulate N steps.
    for t in range(N):
        res = solver.solve()
        assert res.info.status == "SOLVED"
        print(f"Timestep {t}. Iters: {res.info.iters}. Solve time: {res.info.solve_time:.3f} us.")

        if verbose:
            print(f"u[{t}] = {res.u[0]}")
        
        x = A @ x + B @ np.clip(res.u[0], -1, 1)
        # Update solver's initial condition
        solver.update(x)
        print()

if __name__ == '__main__':
    time_ocp()
