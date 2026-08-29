import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

nu = 3
nx = 12
A21 = -2.0*np.eye(6)
for i in range(5):
    A21[i, i+1] = 1.0
    A21[i+1, i] = 1.0
B2 = np.zeros((6, 3))
B2[0, 0] = B2[2, 1] = B2[3, 2] = 1.0
B2[1, 0] = B2[4, 1] = B2[5, 2] = -1.0
Ac = np.block([
    [np.zeros((6,6)), np.eye(6)],
    [A21, np.zeros((6,6))]
])
Bc = np.vstack([np.zeros((6,3)), B2])
Cc = np.hstack([np.eye(6), np.zeros((6,6))])
Dc = np.zeros((6,3))
sys = ctrl.StateSpace(Ac, Bc, Cc, Dc)
Ts = 0.5
dtsys = ctrl.c2d(sys, Ts)
A = dtsys.A
B = dtsys.B
#x0 = np.hstack([np.zeros(6), 2.0*np.ones(6)])
x0 = np.hstack([2*np.ones(6), np.zeros(6)])
xT = np.zeros(12)