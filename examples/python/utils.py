import numpy as np

def dare(A, B, Q, R, max_iter=1000):
    nu = B.shape[1]
    Lp = np.linalg.cholesky(Q)
    for k in range(max_iter):
        ALp = A.T @ Lp
        BLp = B.T @ Lp
        t22 = Q + ALp @ ALp.T
        t12 = BLp @ ALp.T
        t11 = R + BLp @ BLp.T
        L = np.linalg.cholesky(np.block([
            [t11, t12],
            [t12.T, t22]
        ]))
        Lp_new = L[nu:, nu:]
        res = np.linalg.norm(Lp_new - Lp)
        Lp = Lp_new
        if res < 1e-9: break
    if k == max_iter: print("!!! DARE reached maximum number of iterations.")
    return Lp_new @ Lp_new.T