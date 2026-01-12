import cupy as cp
import cupyx.scipy.sparse as cpx_sp

def stable_wexp(eta, w):
    """
    Compute: w * exp(eta) using max-trick for stability.
    """
    m = cp.max(eta)
    e = cp.exp(eta - m)
    return w * e, cp.exp(m)

def stable_sum(vec, scale):
    return float(scale * cp.sum(vec))

def stable_XT_wexp(X, ve, scale):
    return scale * (X.T @ ve)

def stable_quad_AT_W_A(A, ve, scale):
    """
    Compute A^T diag(ve) A efficiently.
    Works for cuSparse A.
    """
    M = A.T @ (cpx_sp.diags(ve) @ A)
    if hasattr(M, "toarray"):
        M = M.toarray()
    return scale * M

def pivoted_cholesky_inverse(H_dense, rank=15):
    """
    Compute a low-rank approximation of H^{-1} using pivoted Cholesky.
    H_dense is n×n on GPU (cupy.ndarray).
    Output:
        D: diagonal vector (n,)
        U: low-rank factor (n×rank)
    """
    n = H_dense.shape[0]
    diagH = cp.diag(H_dense)
    D = 1.0 / (diagH + 1e-8)          # initial diagonal approx
    U = cp.zeros((n, rank), dtype=H_dense.dtype)

    # Simplified pivoted-Cholesky suitable for inverse approx
    for k in range(rank):
        pivot = int(cp.argmax(D))
        dk = D[pivot]
        if dk < 1e-12:
            break

        u = cp.zeros(n, dtype=H_dense.dtype)
        u[pivot] = cp.sqrt(dk)

        U[:, k] = u
        D = D - u*u
        D = cp.maximum(D, 1e-12)

    return D, U[:, :k+1]