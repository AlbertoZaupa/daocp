"""Regenerates examples/data/quadrotor.npz. Run from the top-level directory.

The reference of the NMPC of Sun et al. (see examples/python/quadrotor_model.py)
is one of the dynamically feasible circles of its simulation study. Such a
circle is a steady state of the quadrotor: the reference at time t is the one at
time 0 rotated about the vertical axis, and the model, the cost of eq. (11) and
the bounds are all invariant under that rotation. Written in coordinates that
rotate with the reference, the tracking problem is therefore time invariant, and
the controller solves the same QP at every step with only its initial state
updated. This script linearizes the rotating dynamics about the reference and
stores that QP.
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "lib"), str(root)]

from examples.python.quadrotor_model import *

# Circle of the reference grid of Sec. VI-A, flown clockwise at 10 m/s with
# 40 m/s^2 of acceleration. It needs 7.56 N of the 8.5 N a rotor delivers, so it
# is one of the dynamically feasible references of Table III.
a_max = 40.0        # m/s^2
v_max = 10.0        # m/s
eps = 1e-6

def record():
    x_bar, u_bar, rate = circular_reference(a_max, v_max)
    print(f"reference: {v_max} m/s on a {v_max**2/a_max:.2f} m circle, "
          f"{u_bar.max():.2f} N per rotor (limit {u_max} N)")

    # One node of the prediction, in coordinates rotating with the reference.
    F = lambda z, u: yaw_rotate(h(z, u), -rate * Ts)
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))
    for i in range(nx):
        d = np.zeros(nx)
        d[i] = eps
        A[:, i] = (F(x_bar + d, u_bar) - F(x_bar - d, u_bar)) / (2 * eps)
    for i in range(nu):
        d = np.zeros(nu)
        d[i] = eps
        B[:, i] = (F(x_bar, u_bar + d) - F(x_bar, u_bar - d)) / (2 * eps)
    w = F(x_bar, u_bar) - x_bar
    print(f"the reference is a steady state to {np.max(np.abs(w)):.2e}")

    # The flight starts from hover at the center of the circle, so the quadrotor
    # has to enter the maneuver before it can track it.
    x0 = np.zeros(nx)
    x0[2] = x_bar[2]
    x0[6] = 1.0

    np.savez('examples/data/quadrotor.npz', A=A, B=B, w=w, x_bar=x_bar,
             u_bar=u_bar, rate=np.float64(rate), x0=x0)
    print("saved examples/data/quadrotor.npz")

if __name__ == '__main__':
    record()
