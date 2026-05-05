"""
Stochastic Lanczos Quadrature (SLQ) estimators for spectral functions.

Goal: estimate Tr[f(A)] for Hermitian PSD A in O(k * N^2) instead of the
O(N^3) eigendecomposition cost.

Reference
---------
Ubaru, Chen, Saad (2017),
"Fast estimation of tr(f(A)) via stochastic Lanczos quadrature"
SIAM J. Matrix Anal. Appl.

Algorithm
---------
1. Draw m independent random probe vectors v_i (complex Rademacher).
2. For each v_i, run k-step Lanczos on A starting from v_i. The recurrence
        A Q_k = Q_k T_k + beta_k q_{k+1} e_k^T
   yields the (real symmetric) tridiagonal T_k (k x k).
3. Diagonalize the small T_k = U Theta U^T  (cheap, O(k^3) but k is tiny).
4. Estimate v_i^H f(A) v_i = ||v_i||^2 * sum_j tau_j f(theta_j)
   where tau_j = (U[0, j])^2.
5. Tr[f(A)] approximated by  (1/m) * sum_i v_i^H f(A) v_i.
   (Complex Rademacher entries from {1, -1, i, -i} have variance 1, giving
   E[v^H M v] = Tr[M] for any Hermitian M.)

Implementation note
-------------------
SLQ is dominated by k matvecs of A on N-dim vectors per probe, i.e.
O(k * N^2). For N <= a few thousand, NumPy + Accelerate (BLAS) on M-series
silicon already saturates the matrix-vector bandwidth and has zero
dispatch overhead per step, while every MLX call pays a synchronisation
cost. We therefore implement the inner Lanczos loop in NumPy. The public
API still accepts ``mx.array`` inputs (we convert once); this preserves
compatibility with the rest of mlx-qre.

If you want a pure-MLX hot path in the future, the right move is to
batch all m probes into block Lanczos and JIT the recurrence with
``mx.compile`` once GPU eigh / block tridiagonalisation lands.

Public API
----------
    lanczos_tridiag(A, v, k)          low-level Lanczos (numpy)
    stochastic_lanczos_logtr(A, ...)  estimate Tr[A ln A]  (= -S(rho))
    von_neumann_entropy_lanczos       S(rho) = -Tr[rho ln rho]
    stochastic_lanczos_cross_logtr    Tr[rho ln sigma] (Hutchinson + SLQ)
    quantum_relative_entropy_lanczos  D(rho||sigma) via SLQ

Cross-term Tr[rho ln sigma]
---------------------------
Tr[rho ln sigma] cannot be written as Tr[g(M)] for a single matrix M with
a scalar g, so single-shot Lanczos quadrature (which lives on the spectrum
of one matrix) does not apply directly. We use Hutchinson + Lanczos:
    Tr[rho ln sigma] = E[v^H rho ln(sigma) v]
                     = E[(rho v)^H ln(sigma) v]   (rho Hermitian)
For each probe v we form u = rho @ v (one O(N^2) matvec) and approximate
the matvec ln(sigma) v via Lanczos:
    ln(sigma) v approx ||v|| * Q_k ln(T_k) e_1
where (Q_k, T_k) come from a k-step Lanczos run on sigma starting from v.
We then take the inner product u^H (ln(sigma) v) directly. This is one
Lanczos run per probe (cheaper than the polarisation alternative which
needs 4) and empirically gives the same ~5% accuracy at k=20, m=10.

Tr[rho ln sigma] is real for Hermitian PSD rho, sigma; the imaginary
part of the sample average is estimator noise and is dropped.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from typing import Optional, Tuple, Union

ArrayLike = Union[mx.array, np.ndarray]

_EPS = 1e-30


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _to_numpy(A: ArrayLike) -> np.ndarray:
    """Bring A to a NumPy array on the host. mx.array -> np.array via mx.eval."""
    if isinstance(A, np.ndarray):
        return A
    if isinstance(A, mx.array):
        mx.eval(A)
        return np.array(A)
    return np.asarray(A)


def _make_probe_vectors(
    N: int,
    m: int,
    seed: Optional[int] = None,
    kind: str = "rademacher",
    complex_dtype: bool = True,
    dtype: type = np.complex128,
) -> np.ndarray:
    """Generate m probe vectors of dimension N.

    complex Rademacher: i.i.d. uniform on {1, -1, i, -i}. Each entry has
    unit modulus and zero mean -> Hutchinson estimator with E[v^H M v] = Tr[M].

    Returns array of shape (m, N) with the requested complex dtype.
    """
    rng = np.random.default_rng(seed)
    if kind == "rademacher":
        if complex_dtype:
            choices = np.array([1.0 + 0j, -1.0 + 0j, 0 + 1j, 0 - 1j], dtype=dtype)
            idx = rng.integers(0, 4, size=(m, N))
            V = choices[idx]
        else:
            V = rng.choice(np.array([1.0, -1.0]), size=(m, N)).astype(np.float64)
    elif kind == "gaussian":
        if complex_dtype:
            V = (rng.standard_normal((m, N)) + 1j * rng.standard_normal((m, N))) / np.sqrt(2)
            V = V.astype(dtype)
        else:
            V = rng.standard_normal((m, N)).astype(np.float64)
    else:
        raise ValueError("kind must be 'rademacher' or 'gaussian'")
    return V


# ---------------------------------------------------------------------------
#  Low level Lanczos (numpy hot path)
# ---------------------------------------------------------------------------

def lanczos_tridiag(
    A: ArrayLike,
    v0: ArrayLike,
    k: int,
    reorth: bool = True,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    k-step Lanczos tridiagonalization of Hermitian A starting from v0.

    Builds an orthonormal basis Q = [q_1, ..., q_kp] (kp <= k) such that
    A Q_kp = Q_kp T_kp + beta_kp q_{kp+1} e_{kp}^T,
    with T_kp the real symmetric tridiagonal (alpha on diag, beta off-diag).

    Parameters
    ----------
    A : array, shape (N, N)
        Hermitian matrix (numpy or mx.array; we convert to numpy).
    v0 : array, shape (N,)
        Starting vector (need not be unit norm; we normalize internally).
    k : int
        Maximum number of Lanczos steps.
    reorth : bool
        If True, perform full re-orthogonalization (Modified Gram-Schmidt)
        on every iteration. Costs an extra O(k * N) per step but kills
        ghost-eigenvalue / loss-of-orthogonality artefacts at modest k.
    tol : float
        If beta drops below tol the iteration terminates early (invariant
        subspace reached).

    Returns
    -------
    alpha : np.ndarray, shape (kp,)
        Real diagonal of T.
    beta : np.ndarray, shape (kp - 1,)
        Real off-diagonal of T.
    kp : int
        Effective number of Lanczos steps actually performed (<= k).
    """
    A_np = _to_numpy(A)
    v_np = _to_numpy(v0)
    return _lanczos_tridiag_np(A_np, v_np, k, reorth=reorth, tol=tol)


def _lanczos_tridiag_np(
    A: np.ndarray,
    v0: np.ndarray,
    k: int,
    reorth: bool = True,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Numpy hot path for Lanczos tridiagonalization."""
    N = v0.shape[0]
    if k <= 0:
        raise ValueError("k must be >= 1")
    if k > N:
        k = N

    # Promote dtype so complex Hermitian A and probe vector mix cleanly
    if np.iscomplexobj(A) and not np.iscomplexobj(v0):
        v0 = v0.astype(A.dtype)
    elif np.iscomplexobj(v0) and not np.iscomplexobj(A):
        A = A.astype(v0.dtype)
    work_dtype = np.result_type(A.dtype, v0.dtype, np.complex128)
    A = A.astype(work_dtype, copy=False)
    v0 = v0.astype(work_dtype, copy=False)

    nv = float(np.linalg.norm(v0))
    if nv < tol:
        raise ValueError("Starting vector has near-zero norm")
    q = v0 / nv

    # Pre-allocate basis matrix Q of shape (k, N) row-major for fast MGS
    Q = np.zeros((k, N), dtype=work_dtype)
    Q[0] = q
    alphas = np.empty(k, dtype=np.float64)
    betas = np.empty(k - 1 if k > 1 else 0, dtype=np.float64)

    beta_prev = 0.0
    q_prev = np.zeros_like(q)
    kp = k

    for j in range(k):
        # w = A q_j - beta_{j-1} q_{j-1}
        w = A @ q
        if j > 0:
            w = w - beta_prev * q_prev

        # alpha_j = <q_j, w>  (real for Hermitian A)
        a_complex = np.vdot(q, w)
        alpha = float(np.real(a_complex))
        alphas[j] = alpha

        w = w - alpha * q

        if reorth and j > 0:
            # Modified Gram-Schmidt against ALL previous basis vectors.
            # Two passes are the textbook safe choice; in our k <= ~50
            # regime one pass is enough, two adds negligible cost.
            for _ in range(2):
                # Coefficients <q_old, w> for q_old in Q[:j+1]
                coefs = Q[: j + 1].conj() @ w  # shape (j+1,)
                w = w - coefs @ Q[: j + 1]

        beta = float(np.linalg.norm(w))

        if j < k - 1:
            if beta < tol:
                # Invariant subspace: terminate
                kp = j + 1
                break
            betas[j] = beta
            q_prev = q
            q = w / beta
            Q[j + 1] = q
            beta_prev = beta
        # else: final step, no q_{k+1} needed

    if kp < k:
        alphas = alphas[:kp]
        betas = betas[: max(kp - 1, 0)]
    return alphas, betas, kp


# ---------------------------------------------------------------------------
#  Quadrature: tridiagonal -> nodes & weights
# ---------------------------------------------------------------------------

def _tridiag_eigh(alpha: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigendecompose real symmetric tridiagonal T (kp x kp).

    Returns (theta, tau) where theta are the eigenvalues (Ritz values, the
    quadrature nodes) and tau_j = U[0, j]^2 are the squared first
    components of the eigenvectors (the quadrature weights).
    """
    kp = alpha.shape[0]
    if kp == 1:
        return alpha.copy(), np.array([1.0])
    T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    theta, U = np.linalg.eigh(T)
    tau = U[0, :] ** 2
    return theta, tau


def _quadratic_form_lanczos_np(
    A: np.ndarray,
    v: np.ndarray,
    k: int,
    f,
    eps: float = _EPS,
    reorth: bool = True,
) -> float:
    """Estimate v^H f(A) v via k-step Lanczos quadrature (numpy hot path)."""
    nv2 = float(np.real(np.vdot(v, v)))
    if nv2 < 1e-30:
        return 0.0
    alpha, beta, _ = _lanczos_tridiag_np(A, v, k, reorth=reorth)
    theta, tau = _tridiag_eigh(alpha, beta)
    theta_safe = np.maximum(theta, eps)
    f_vals = f(theta_safe)
    quad = float(np.sum(tau * f_vals))
    return nv2 * quad


# ---------------------------------------------------------------------------
#  Tr[A ln A]  via SLQ
# ---------------------------------------------------------------------------

def stochastic_lanczos_logtr(
    A: ArrayLike,
    k: int = 20,
    m: int = 10,
    seed: Optional[int] = None,
    eps: float = _EPS,
    reorth: bool = True,
) -> float:
    """
    Estimate Tr[A ln A] for Hermitian PSD A via Stochastic Lanczos Quadrature.

    Equivalent to summing  lambda_i * ln(lambda_i)  over all eigenvalues
    of A, but at cost O(m * k * N^2) instead of O(N^3).

    Note: For a density matrix rho (PSD, Tr=1) this returns
        Tr[rho ln rho] = -S(rho)   (negative von Neumann entropy).

    Parameters
    ----------
    A : mx.array or np.ndarray, shape (N, N)
        Hermitian PSD matrix.
    k : int
        Lanczos depth (typical 15-40 for entropy-class problems).
    m : int
        Number of probe vectors.
    seed : int, optional
        RNG seed.
    eps : float
        Floor on eigenvalues.
    reorth : bool
        Full re-orth in Lanczos.

    Returns
    -------
    float
        Estimate of Tr[A ln A].
    """
    A_np = _to_numpy(A)
    N = A_np.shape[-1]
    f = lambda x: x * np.log(np.maximum(x, eps))

    V = _make_probe_vectors(N, m, seed=seed, kind="rademacher", complex_dtype=True)
    total = 0.0
    for i in range(m):
        total += _quadratic_form_lanczos_np(A_np, V[i], k, f, eps=eps, reorth=reorth)
    return total / m


def von_neumann_entropy_lanczos(
    rho: ArrayLike,
    k: int = 20,
    m: int = 10,
    seed: Optional[int] = None,
    eps: float = _EPS,
    reorth: bool = True,
) -> float:
    """
    Von Neumann entropy S(rho) = -Tr[rho ln rho] via Stochastic Lanczos.

    Returns a Python float (real, non-negative up to estimator noise).
    """
    return -stochastic_lanczos_logtr(
        rho, k=k, m=m, seed=seed, eps=eps, reorth=reorth
    )


# ---------------------------------------------------------------------------
#  Cross-term Tr[rho ln sigma]  via Hutchinson + Lanczos quadrature
# ---------------------------------------------------------------------------

def _lanczos_apply_f_np(
    A: np.ndarray,
    v: np.ndarray,
    k: int,
    f,
    eps: float = _EPS,
    reorth: bool = True,
    tol: float = 1e-12,
) -> np.ndarray:
    """Compute f(A) v approximately via Lanczos.

    Builds the Lanczos basis Q_kp and tridiagonal T_kp, then approximates
        f(A) v approx ||v|| * Q_kp f(T_kp) e_1
    For f = log this is the standard Lanczos rational approximation; for
    f = x log x it gives the corresponding polynomial-on-Krylov estimate.

    Returns the approximate vector f(A) v of shape (N,).
    """
    N = v.shape[0]
    if k <= 0:
        raise ValueError("k must be >= 1")
    if k > N:
        k = N
    work_dtype = np.result_type(A.dtype, v.dtype, np.complex128)
    A = A.astype(work_dtype, copy=False)
    v = v.astype(work_dtype, copy=False)

    nv = float(np.linalg.norm(v))
    if nv < tol:
        return np.zeros(N, dtype=work_dtype)
    q = v / nv

    Q = np.zeros((k, N), dtype=work_dtype)
    Q[0] = q
    alphas = np.empty(k, dtype=np.float64)
    betas = np.empty(k - 1 if k > 1 else 0, dtype=np.float64)

    beta_prev = 0.0
    q_prev = np.zeros_like(q)
    kp = k

    for j in range(k):
        w = A @ q
        if j > 0:
            w = w - beta_prev * q_prev
        a_complex = np.vdot(q, w)
        alpha = float(np.real(a_complex))
        alphas[j] = alpha
        w = w - alpha * q

        if reorth and j > 0:
            for _ in range(2):
                coefs = Q[: j + 1].conj() @ w
                w = w - coefs @ Q[: j + 1]

        beta = float(np.linalg.norm(w))
        if j < k - 1:
            if beta < tol:
                kp = j + 1
                break
            betas[j] = beta
            q_prev = q
            q = w / beta
            Q[j + 1] = q
            beta_prev = beta

    if kp < k:
        alphas = alphas[:kp]
        betas = betas[: max(kp - 1, 0)]
        Q = Q[:kp]

    if kp == 1:
        T = np.array([[alphas[0]]])
    else:
        T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
    theta, U = np.linalg.eigh(T)
    f_theta = f(np.maximum(theta, eps))
    # f(T) e_1 = U diag(f_theta) U^T e_1 = U (f_theta * U[0, :])
    f_T_e1 = U @ (f_theta * U[0, :])
    return nv * (Q.T @ f_T_e1)


def stochastic_lanczos_cross_logtr(
    rho: ArrayLike,
    sigma: ArrayLike,
    k: int = 20,
    m: int = 10,
    seed: Optional[int] = None,
    eps: float = _EPS,
    reorth: bool = True,
) -> float:
    """
    Estimate Tr[rho ln sigma] via Hutchinson + Lanczos quadrature on sigma.

    For each probe v we compute u = rho @ v (one O(N^2) matvec) and then
    estimate Re(u^H ln(sigma) v) by polarization + two k-step Lanczos
    runs on sigma.

    Returns a Python float.
    """
    rho_np = _to_numpy(rho)
    sigma_np = _to_numpy(sigma)
    N = rho_np.shape[-1]
    V = _make_probe_vectors(N, m, seed=seed, kind="rademacher", complex_dtype=True)
    total = 0.0 + 0.0j
    for i in range(m):
        v = V[i]
        u = rho_np @ v  # O(N^2)
        # Lanczos approximation of ln(sigma) v then inner product with u.
        ln_sig_v = _lanczos_apply_f_np(sigma_np, v, k, np.log, eps=eps, reorth=reorth)
        total += np.vdot(u, ln_sig_v)
    # Tr[rho ln sigma] is real for Hermitian PSD operators; the imaginary
    # part of the sample mean is estimator noise.
    return float(np.real(total / m))


def quantum_relative_entropy_lanczos(
    rho: ArrayLike,
    sigma: ArrayLike,
    k: int = 20,
    m: int = 10,
    seed: Optional[int] = None,
    eps: float = _EPS,
    reorth: bool = True,
) -> float:
    """
    Estimate D(rho || sigma) = Tr[rho (ln rho - ln sigma)] via SLQ.

    The two terms are estimated independently:
        Tr[rho ln rho]    <- stochastic_lanczos_logtr(rho)
        Tr[rho ln sigma]  <- stochastic_lanczos_cross_logtr(rho, sigma)

    Cost: O((2 m) * k * N^2) -- m Lanczos runs on rho plus m on sigma
    (one Lanczos-apply per probe).

    Parameters
    ----------
    rho, sigma : mx.array or np.ndarray, shape (N, N)
        Hermitian PSD matrices, Tr = 1. rho's support must lie inside
        sigma's support; otherwise the cross-term diverges.
    k : int
        Lanczos depth per run.
    m : int
        Number of Hutchinson probes.
    seed : int, optional
        RNG seed for reproducibility.
    eps : float
        Floor for ln.
    reorth : bool
        Full re-orth in Lanczos (recommended True).

    Returns
    -------
    float
        Estimate of D(rho || sigma).
    """
    s1 = seed
    s2 = None if seed is None else seed + 10_007
    term_self = stochastic_lanczos_logtr(rho, k=k, m=m, seed=s1, eps=eps, reorth=reorth)
    term_cross = stochastic_lanczos_cross_logtr(rho, sigma, k=k, m=m, seed=s2, eps=eps, reorth=reorth)
    return float(term_self - term_cross)
