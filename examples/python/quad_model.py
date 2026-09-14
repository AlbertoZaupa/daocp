# Copyright (c) 2026 Alberto Zaupa
# SPDX-License-Identifier: MIT
# See LICENSE in the project root for license information.

import numpy as np
from examples.python.utils import rk4

nx = 12
nu = 4

g = 9.81
mass = 1.5
Jx_ = Jy_ = 0.03
Jz_ = 0.06
l = 0.225
c = 0.015
Ts = 0.02

A_ = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
B_ = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1/mass, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1/Jx_, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1/Jy_, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1/Jz_]])
M = np.array([[1, 1, 1, 1],
                  [0, l, 0, -l],
                  [-l, 0, l, 0],
                  [-c, c, -c, c]])
B_ct = B_ @ M
A = np.linalg.inv(np.eye(nx) - Ts * A_)
B = Ts * A @ B_ct
u_hover = np.array([g*mass/4, g*mass/4, g*mass/4, g*mass/4])

def f(x, u):
    T = np.sum(u)
    tau_phi = l * (u[1] - u[3])
    tau_th = l * (u[2] - u[0])
    tau_psi = c * (-u[0] - u[2] + u[1] + u[3])

    phi = x[6]
    sphi = np.sin(phi)
    cphi = np.cos(phi)
    th = x[8]
    sth = np.sin(th)
    cth = np.cos(th)
    psi = x[10]
    spsi = np.sin(psi)
    cpsi = np.cos(psi)
    dphi = x[7]
    dth = x[9]
    dpsi = x[11]
    dx = x[1]
    dy = x[3]
    dz = x[5]
        
    ddx = T * (spsi*sphi + cpsi*sth*cphi) / mass
    ddy = T * (spsi*sth*cphi - cpsi*sphi) / mass
    ddz = T * cth * cphi / mass - g
    ddphi = (Jy_ - Jz_) / Jx_ * dth * dpsi + tau_phi / Jx_
    ddth = (Jz_ - Jx_) / Jy_ * dpsi * dphi + tau_th / Jy_
    ddpsi = (Jx_ - Jy_) / Jz_ * dphi * dth + tau_psi / Jz_

    return np.array([dx, ddx, dy, ddy, dz, ddz, dphi, ddphi, dth, ddth, dpsi, ddpsi])

h = lambda x, u: rk4(f, x, u + u_hover, Ts)

def generate_ref(dt, T_sim, N):
    # Generate Reference Trajectory (Figure 8)
    time_steps = int(T_sim / dt)
    ref_traj_full = np.zeros((12, time_steps + N + 1))

    for k in range(time_steps + N + 1):
        t = k * dt
        # Fast Figure-8
        ref_traj_full[0, k] = 2.0 * np.sin(1.0 * t)       # x
        ref_traj_full[2, k] = 2.0 * np.sin(2.0 * t) / 2.0 # y
        ref_traj_full[4, k] = 1.0 + 0.5 * np.sin(0.5 * t) # z (varying altitude)
    
    return ref_traj_full