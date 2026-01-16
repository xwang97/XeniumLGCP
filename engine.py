import time
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpx_sp
from tqdm import tqdm

# Import the GPU stable operations from the utils folder
from .utils.stable_ops import (
    stable_wexp, stable_sum, stable_XT_wexp, 
    stable_quad_AT_W_A, pivoted_cholesky_inverse
)

class VoGCAM_SPDE_GPU:
    def __init__(
        self,
        tau_prior=(2.0, 0.5),
        lambda_beta=2.0,
        tau_floor=0.1,
        base_ridge=1e-4,
        max_iter=100,
        tol=1e-4,
        verbose=True,
        rank=20
    ):
        self.tau_prior = tau_prior
        self.lambda_beta = lambda_beta
        self.tau_floor = tau_floor
        self.base_ridge = base_ridge
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.rank = rank

    def cg_solve(self, A, b, max_iter=200, tol=1e-6):
        x = cp.zeros_like(b)
        r = b - A @ x
        p = r.copy()
        rs_old = cp.dot(r, r)

        for _ in range(max_iter):
            Ap = A @ p
            alpha = rs_old / (cp.dot(p, Ap) + 1e-12)
            x += alpha * p
            r -= alpha * Ap
            rs_new = cp.dot(r, r)
            if cp.sqrt(rs_new) < tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def compute_eta_var(self, A_tilde, D, U):
        eta_var = A_tilde.power(2).dot(D)
        for k in range(U.shape[1]):
            uk = U[:, k]
            Auk = A_tilde @ uk
            eta_var += Auk * Auk
        return eta_var

    def trace_QSigma(self, Q_dense, D, U):
        tr1 = cp.sum(Q_dense.diagonal() * D)
        tr2 = 0.0
        for k in range(U.shape[1]):
            uk = U[:, k]
            Quk = Q_dense @ uk
            tr2 += float(uk @ Quk)
        return tr1 + tr2

    def logdet_Sigma(self, D, U):
        logdetD = cp.sum(cp.log(D))
        if U.shape[1] == 0:
            return float(logdetD)
        Dinv = 1.0 / (D + 1e-12)
        UD = U.T * Dinv  
        B = UD @ U        
        signB, logdetB = cp.linalg.slogdet(cp.eye(B.shape[0]) + B)
        return float(logdetD + logdetB)

    def _elbo_full(self, beta, mu, D, U, Qb, A_tilde, X_tilde, A, X, w, tau):
        eta_var = self.compute_eta_var(A_tilde, D, U)
        eta = X_tilde @ beta + A_tilde @ mu + 0.5 * eta_var
        ve, sc = stable_wexp(eta, w)

        like = float(cp.sum(X @ beta) + cp.sum(A @ mu) - stable_sum(ve, sc))

        Q = (tau**2) * Qb + 1e-8 * cpx_sp.eye(Qb.shape[0])
        Q_dense = Q.toarray()
        muQmu = float(mu @ (Q_dense @ mu))

        trQSigma = self.trace_QSigma(Q_dense, D, U)
        prior_mu = -0.5 * (muQmu + trQSigma)

        logdetSigma = self.logdet_Sigma(D, U)
        signQ, logdetQ = np.linalg.slogdet(cp.asnumpy(Q_dense))
        entropy = 0.5 * (logdetSigma - logdetQ)

        a, b = self.tau_prior
        tau_prior = (a + 1) * cp.log(tau**2) + b * (tau**2)
        beta_pen = -0.5 * self.lambda_beta * float(beta[1:] @ beta[1:])

        return like + prior_mu + entropy + beta_pen - float(tau_prior)

    def fit(self, A_tilde, A, X_tilde, X, w, Q_base, beta_init):
        n_nodes = Q_base.shape[0]
        p = X_tilde.shape[1]

        A_tilde = cpx_sp.csr_matrix(A_tilde)
        A = cpx_sp.csr_matrix(A)
        X_tilde = cp.asarray(X_tilde)
        X = cp.asarray(X)
        w = cp.asarray(w)
        Qb = cpx_sp.csr_matrix(Q_base)

        self.beta_ = cp.asarray(beta_init)
        self.mu_ = cp.zeros(n_nodes)
        D = cp.ones(n_nodes) * 1e-2
        U = cp.zeros((n_nodes, self.rank))
        self.D_, self.U_ = D, U
        self.tau_ = cp.array(1.0)
        self.elbo_history_ = []

        pbar = tqdm(range(self.max_iter), disable=not self.verbose, desc="LGCP Fitting")
        for it in pbar:
            # 1. mu-step
            eta_var = self.compute_eta_var(A_tilde, D, U)
            eta = X_tilde @ self.beta_ + A_tilde @ self.mu_ + 0.5 * eta_var
            ve, sc = stable_wexp(eta, w)

            Q = (self.tau_**2) * Qb + 1e-8 * cpx_sp.eye(n_nodes)
            Hmu_dense = (Q + stable_quad_AT_W_A(A_tilde, ve, sc)).get()
            
            grad_mu = (A.T @ cp.ones(A.shape[0])) - (A_tilde.T @ (sc * ve)) - (Q @ self.mu_)
            grad_mu = grad_mu.get()

            mu_new = cp.asarray(self.cg_solve(cp.asarray(Hmu_dense), cp.asarray(grad_mu)))
            self.mu_ += 0.1 * (mu_new - self.mu_)

            # 2. Sigma-step (Pivoted Cholesky)
            H = Q + stable_quad_AT_W_A(A_tilde, ve, sc)
            D_new, U_new = pivoted_cholesky_inverse(cp.asarray(H), rank=self.rank)
            self.D_ = 0.95 * self.D_ + 0.05 * D_new
            self.U_ = 0.95 * self.U_ + 0.05 * U_new
            D, U = self.D_, self.U_

            # 3. Beta-step
            eta_var = self.compute_eta_var(A_tilde, D, U)
            eta = X_tilde @ self.beta_ + A_tilde @ self.mu_ + 0.5 * eta_var
            ve, sc = stable_wexp(eta, w)

            Hbeta_dense = stable_quad_AT_W_A(X_tilde, ve, sc).get()
            grad_beta = (X.T @ cp.ones(A.shape[0])) - stable_XT_wexp(X_tilde, ve, sc)
            grad_beta[1:] -= self.lambda_beta * self.beta_[1:]

            delta_beta = cp.linalg.solve(cp.asarray(Hbeta_dense) + 1e-6 * cp.eye(p), grad_beta)
            self.beta_ += 0.2 * delta_beta

            # 4. Tau-step
            Q_dense = Qb.toarray()
            trQSigma = self.trace_QSigma(Q_dense, D, U)
            muQmu = float(self.mu_ @ ((self.tau_**2) * (Q_dense @ self.mu_)))
            a, b = self.tau_prior
            tau_new = cp.sqrt(max((n_nodes + 2*a - 2), 1e-10) / max((trQSigma + muQmu + 2*b), 1e-10))
            tau_new = cp.maximum(tau_new, self.tau_floor)
            self.tau_ = 0.9 * self.tau_ + 0.1 * tau_new

            # Track ELBO
            elbo = self._elbo_full(self.beta_, self.mu_, D, U, Qb, A_tilde, X_tilde, A, X, w, self.tau_)
            self.elbo_history_.append(elbo)
            
            # if self.verbose:
            #     print(f"[Iter {it}] ELBO={elbo:.3e} tau={float(self.tau_):.3f}")

        return self

    def expected_intensity_on_grid(self, A_tilde, X_tilde):
        A_tilde = cpx_sp.csr_matrix(A_tilde)
        X_tilde = cp.asarray(X_tilde)
        D, U = self.D_, self.U_
        eta_var = self.compute_eta_var(A_tilde, D, U)
        eta = X_tilde @ self.beta_ + A_tilde @ self.mu_ + 0.5 * eta_var
        ve, sc = stable_wexp(eta, cp.ones_like(eta))
        return sc * ve