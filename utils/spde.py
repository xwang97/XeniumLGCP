import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import inv as sparse_inv
from scipy.spatial.distance import cdist
from scipy.special import kv, gamma

def spde_precision_with_mass(coords, tri, kappa=0.2):
    """
    Build Q = kappa^4 * C + 2*kappa^2 * G + G C^{-1} G
    using CPU-based finite elements.
    """
    n = coords.shape[0]
    C = lil_matrix((n, n))
    G = lil_matrix((n, n))

    for t in tri:
        V = coords[t]
        area = 0.5 * abs(
            (V[1,0] - V[0,0]) * (V[2,1] - V[0,1]) -
            (V[2,0] - V[0,0]) * (V[1,1] - V[0,1])
        )
        # B matrix for gradients
        B = np.ones((3, 3))
        B[:, 1:] = V
        Binv = np.linalg.inv(B)
        grad = Binv[1:, :]   # 2x3

        # Local mass and stiffness
        C_loc = (area/12.0) * (np.ones((3,3)) + np.eye(3))
        G_loc = area * (grad.T @ grad)

        for i in range(3):
            ii = t[i]
            for j in range(3):
                jj = t[j]
                C[ii, jj] += C_loc[i, j]
                G[ii, jj] += G_loc[i, j]

    C = C.tocsr()
    G = G.tocsr()

    # C^{-1} (mass matrix inverse)
    Cinv = sparse_inv(C)

    # SPDE precision
    Q = (kappa**4) * C + 2*(kappa**2)*G + G @ Cinv @ G
    return Q, C

def matern_covariance(coords, sigma2=0.22, kappa=0.2, nu=0.5):
    d = cdist(coords, coords)
    d = np.maximum(d, 1e-9)
    factor = sigma2 / (2 ** (nu - 1) * gamma(nu))
    K = factor * (kappa * d)**nu * kv(nu, kappa*d)
    K[np.isnan(K)] = sigma2
    return K