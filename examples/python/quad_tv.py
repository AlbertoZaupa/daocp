import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "lib"), str(root)]

from examples.python.utils import *
from examples.python.quad_model import *
from daocp import *

N = 40
T = 10.0
umin = np.zeros(nu)
umax = u_hover * 1.8
Q = np.diag([100, 1, 100, 1, 200, 1,   10, 1, 10, 1, 10, 1])
R = np.eye(nu) * 0.1
P = dare(A, B, Q, R)
x0 = np.zeros(nx)
ubu = umax - u_hover
lbu = umin - u_hover
lN = lambda a, stages=N: [a.copy() for i in range(stages)]
max_iter = 1000
Ql = lN(Q, N + 1)
Ql[-1] = P.copy()
nreps = 200
x_ref = generate_ref(Ts, T, N)
h = lambda x,u: rk4(f, x, u, Ts)

def solve_mpc():
    solver = OCPsolver(A, B, None, Ql, R, None,
                        None, None, None, None, lbu, ubu,
                        None, None, None,
                        x0, N, nx, nu, max_iter)
    state = x0.copy()
    for t in range(int(T/Ts)):
        q = [- Ql[i] @ x_ref[:, t + i] for i in range(N+1)]
        solver.update(state, q)
        res = solver.solve()
        assert res.info.status == "SOLVED", res.info.status
        print(f"Timestep {t}. Iters: {res.info.iters}. Solve time: {res.info.solve_time:.3f} us.")
        state = h(state, np.clip(res.u[0] + u_hover, lbu, ubu))
        solver.update(state)
        print()

if __name__ == '__main__':
    solve_mpc()
