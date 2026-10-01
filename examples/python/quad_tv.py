# Copyright (c) 2026 Alberto Zaupa
# SPDX-License-Identifier: MIT
# See LICENSE in the project root for license information.

# TIME-VARYING REFERENCE FOLLOWING PROBLEM FOR A QUADROTOR

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "lib"), str(root)]
from examples.python.utils import *
from examples.python.quad_model import *    # quadrotor model loaded from here
from daocp import *

N = 40                      # Horizon length
T = 10.0                    # Simulation duration in seconds
umin = np.zeros(nu)         # Minimum thrust for motors
umax = u_hover * 1.8        # Maximum thrust for motors
Q = np.diag([100,           # Diagonal state cost matrix
             1, 
             100, 
             1, 
             200, 
             1,   
             10, 
             1, 
             10, 
             1, 
             10, 
             1])
R = np.eye(nu) * 0.1        # Diagonal input cost matrix
P = dare(A, B, Q, R)        # Terminal cost matrix
x0 = np.zeros(nx)           # Initial condition
ubu = umax - u_hover        # Reformulate as u[k] = du[k] + u_hovering
                            # where u_hovering compensates gravity
lbu = umin - u_hover

# List of Q matrices: [Q, Q, ..., P]
lN = lambda a, stages=N: [a.copy() for i in range(stages)]
Ql = lN(Q, N + 1)
Ql[-1] = P.copy()

# Generate reference state trajectory
x_ref = generate_ref(Ts, T, N)
# RK4 integrator of nonlinear dynamics
h = lambda x,u: rk4(f, x, u, Ts)

def solve_mpc():
    # Instantiate the solver
    solver = OCPsolver(
        A,      # A matrix. Can be list [A0, A1, ..., A_{N-1}]
        B,      # B matrix. Can be list [B0, B1, ..., B_{N-1}] 
        None,   # w[k] affine terms. None => w[k] = 0
        Ql,     # [Q0, Q1, ..., QN]. Can be a single matrix => Q[k] = Q for all k
        R,      # R matrix. Can be list [R0, R1, ..., R_{N-1}]
        None,   # S matrix, quadratic mixing. None => S[k] = 0
        None,   # q[k] state-linear terms. None => q[k] = 0
        None,   # r[k] input-linear terms. None => r[k] = 0
        None,   # lbx[k] state-lower bounds. None => lbx[k] = -INF
        None,   # ubx[k] state-upper bounds. None => ubx[k] = INF
        lbu,    # lbu[k] input-lower bounds. If None, lbu[k] = -INF
        ubu,    # ubu[k] input-upper bounds. If None, ubu[k] = INF
        None,   # C[k] = [Cu[k] Cx[k]] mixed constraint matrix. None => no mixed constr.
        None,   # cl[k] mixed constraints lower bounds. 
        None,   # cu[k] mixed constraints upper bounds.
        x0,     # initial state. (Can be updated, so arbitrary)
        N,      # horizon length
        greedy=True    # constraint selection strategy. Greedy=True great for simple probs.
    )

    # Run simulation
    x = x0.copy()
    for t in range(int(T/Ts)):
        # Retrieve state-linear term from reference
        q = [- Ql[i] @ x_ref[:, t + i] for i in range(N+1)]
        # Update x0 and q for the solver
        solver.update(x, q)
        res = solver.solve()
        assert res.info.status == "SOLVED", res.info.status
        print(f"Timestep {t}. Iters: {res.info.iters}. Solve time: {res.info.solve_time:.3f} us.")

        # Propagate nonlinar dynamics
        x = h(x, np.clip(res.u[0] + u_hover, lbu, ubu))
        print()

if __name__ == '__main__':
    solve_mpc()
