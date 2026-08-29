import sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "lib"), str(root)]
from examples.python.chom_model import *
from examples.python.utils import *
import daocp

N = 30
R = np.eye(nu)
Q = np.zeros((nx, nx))
lbx = np.hstack([-4*np.ones(6), -np.inf*np.ones(6)])
ubx = -lbx
lbu = -0.5*np.ones(nu)
ubu = -lbu
lbx = [lbx.copy() for t in range(N-1)]
ubx = [ubx.copy() for t in range(N-1)]
lbx.append(xT)
ubx.append(xT)

def solve_ocp():
    solver = daocp.OCPsolver(A, B, None, Q, R, None, None, None,
                            lbx, ubx, lbu, ubu, None, None, None,
                            x0, N, nx, nu)
    print(f"Setup time: {solver.setup_time:.4f} us.")

    result = solver.solve()
    print(f"Solve time: {result.info.solve_time:.4f} us.")
    print(f"Iterations: {result.info.iter}")
    print(f"Solver status: {result.info.status}")

if __name__ == '__main__':
    solve_ocp()