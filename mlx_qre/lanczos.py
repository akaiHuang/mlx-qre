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

Implementation note (pure MLX hot path)
---------------------------------------
This implementation runs entirely on MLX with no NumPy fallback in the
hot path. Two key optimisations make MLX competitive with (and at large
N faster than) the previous Accelerate-backed NumPy hot path:

1. **Block all m probes together as a single (N, m) matrix.** Each
   Lanczos step then needs a single matmul ``A @ V`` (shape (N, N) @
   (N, m) -> (N, m)) instead of m independent matvecs. This both
   amortises GPU dispatch overhead and lets MLX tile the m probes
   across compute units, easily 5-10x faster than m separate matvecs at
   N >= 500.

2. **Lazy evaluation: do not call ``mx.eval`` inside the inner loop.**
   The full Lanczos recurrence is built as a single MLX graph (via
   ``mx.compile`` for the per-step kernel) and only materialised once
   at quadrature time. This avoids per-step GPU command-buffer flushes.

The only non-MLX op left is the small (k, k) tridiagonal eigh used to
extract Ritz nodes/weights. We keep that on the MLX CPU stream
(``mx.linalg.eigh(..., stream=mx.cpu)``); it is real symmetric
tridiagonal of size k <= 30, so CPU is the right place anyway.

Public API
----------
    lanczos_tridiag(A, v, k)          low-level Lanczos (mlx)
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
from typing import Callable, Optional, Tuple, Union

ArrayLike = Union[mx.array, np.ndarray]

_EPS = 1e-30


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _to_mlx(A: ArrayLike, dtype: mx.Dtype = mx.complex64) -> mx.array:
    """Bring A into an mx.array with the requested dtype (default complex64)."""
    if isinstance(A, mx.array):
        if A.dtype != dtype:
            A = A.astype(dtype)
        return A
    if isinstance(A, np.ndarray):
        if np.iscomplexobj(A):
            return mx.array(A.astype(np.complex64)).astype(dtype)
        return mx.array(A).astype(dtype)
    arr = np.asarray(A)
    if np.iscomplexobj(arr):
        return mx.array(arr.astype(np.complex64)).astype(dtype)
    return mx.array(arr).astype(dtype)


def _make_probe_vectors_mlx(
    N: int,
    m: int,
    seed: Optional[int] = None,
    kind: str = "rademacher",
    dtype: mx.Dtype = mx.complex64,
) -> mx.array:
    """Generate m probe vectors of dimension N as an MLX (N, m) complex matrix.

    complex Rademacher: i.i.d. uniform on {1, -1, i, -i}. Each entry has
    unit modulus and zero mean -> Hutchinson estimator with E[v^H M v] = Tr[M].

    We use NumPy here because (a) we only do this once per call (no hot
    path), and (b) MLX's complex random generators are limited (no native
    complex Rademacher / categorical-from-complex-set). The cost is one
    O(N * m) host-side op; everything downstream is on MLX.
    """
    rng = np.random.default_rng(seed)
    if kind == "rademacher":
        # Sample 0..3 -> {1, -1, i, -i}
        idx = rng.integers(0, 4, size=(N, m))
        choices = np.array([1.0 + 0j, -1.0 + 0j, 0 + 1j, 0 - 1j], dtype=np.complex64)
        V = choices[idx]
    elif kind == "gaussian":
        V = (rng.standard_normal((N, m)) + 1j * rng.standard_normal((N, m))).astype(np.complex64) / np.sqrt(2.0).astype(np.float32)
    else:
        raise ValueError("kind must be 'rademacher' or 'gaussian'")
    return mx.array(V).astype(dtype)


# ---------------------------------------------------------------------------
#  Block Lanczos hot path (pure MLX)
# ---------------------------------------------------------------------------
#
# The two compiled kernels below are the hot path of every SLQ run.
# Splitting the per-step work into (a) the three-term recurrence + alpha,
# and (b) the beta computation + alive-mask + q_new normalisation lets
# both halves take fixed-shape inputs and benefit from `mx.compile`'s
# graph caching. The full re-orthogonalisation step (against a growing
# basis) sits between them and runs in plain MLX.

@mx.compile
def _lanczos_step_recurrence(
    A: mx.array, q: mx.array, q_prev: mx.array, beta_prev: mx.array
) -> Tuple[mx.array, mx.array]:
    """Compute alpha_j and the un-orthogonalised w = (A - alpha I) q - beta_{j-1} q_{j-1}.

    Inputs are all fixed-shape per call (we never change N, m, k inside a
    Lanczos run), so `mx.compile` can cache the kernel.
    """
    w = A @ q - beta_prev[None, :] * q_prev
    a_complex = mx.sum(mx.conj(q) * w, axis=0)
    alpha = mx.real(a_complex)
    w = w - alpha.astype(q.dtype)[None, :] * q
    return alpha, w


@mx.compile
def _lanczos_step_finalize(
    w: mx.array, alive: mx.array
) -> Tuple[mx.array, mx.array, mx.array]:
    """Given the orthogonalised residual w, return (beta, q_new, new_alive).

    Handles per-probe early termination: a probe whose beta drops below
    1e-12 is marked dead, its q_new is forced to zero, and alpha/beta in
    later steps will be masked to zero so the small (k, k) tridiag has a
    clean block-diagonal structure. The 1e-6 floor on beta_safe avoids
    MLX's complex64 0/very-small-float NaN.
    """
    beta = mx.sqrt(mx.real(mx.sum(mx.conj(w) * w, axis=0)))
    beta = beta * alive
    new_alive = alive * (beta > mx.array(np.float32(1e-12))).astype(mx.float32)
    # Float floor: 1.0 on dead probes (so we divide w=0 by 1.0 -> 0)
    # and the true beta (>=1e-12) on live probes; an extra global 1e-6
    # floor protects against MLX's complex-division NaN at tiny denominators.
    beta_safe = mx.maximum(beta, mx.array(np.float32(1.0)) * (1.0 - new_alive))
    beta_safe = mx.maximum(beta_safe, mx.array(np.float32(1e-6)))
    q_new = w / beta_safe.astype(w.dtype)[None, :]
    q_new = q_new * new_alive.astype(q_new.dtype)[None, :]
    return beta, q_new, new_alive


def _block_lanczos_mlx(
    A: mx.array,
    V0: mx.array,
    k: int,
    reorth: bool = True,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Run k-step Lanczos on Hermitian A starting from each column of V0.

    All m probes are processed in lockstep as columns of an (N, m) matrix.
    Each step is a single block matmul ``A @ Q`` (shape (N, m)) which is
    why this is fast on MLX even at moderate N: GPU dispatch is amortised
    across the m probes.

    The per-probe tridiagonal matrices T_k(b) have their own (alpha, beta)
    sequences, returned as MLX arrays of shape (k, m) for alpha and
    (k-1, m) for beta. The full Lanczos basis is returned with shape
    (k, N, m) so the caller can compute ``Q^T f(T) e1`` later.

    Returns
    -------
    alphas : mx.array, shape (k, m), real (float32)
        Diagonal of T for each probe.
    betas : mx.array, shape (k - 1, m), real (float32)
        Off-diagonal of T for each probe (always >= 0).
    Q : mx.array, shape (k, N, m), complex
        Orthonormal Lanczos basis per probe (Q[i, :, b] = q_i for probe b).
    norms : mx.array, shape (m,), real (float32)
        Initial norms ||V0[:, b]||, needed to scale the quadrature.
    """
    N, m = V0.shape
    if k <= 0:
        raise ValueError("k must be >= 1")
    if k > N:
        k = N

    # Normalise each column.  norms shape (m,)
    norms = mx.sqrt(mx.real(mx.sum(mx.conj(V0) * V0, axis=0)))
    # Avoid divide-by-zero; if a probe is null, replace by 1 (we'll zero out later)
    norms_safe = mx.maximum(norms, mx.array(1e-30))
    Q0 = V0 / norms_safe[None, :]

    # Pre-allocate as Python lists -- MLX builds a single graph via the
    # closures, so the loop unrolls into one big lazy computation. We
    # avoid in-place writes (MLX is functional) and rely on stack at the
    # end.
    Q_hist = [Q0]                        # list of (N, m) complex
    alpha_hist = []                      # list of (m,) float
    beta_hist = []                       # list of (m,) float

    q_prev = mx.zeros((N, m), dtype=Q0.dtype)
    q = Q0
    beta_prev = mx.zeros((m,), dtype=mx.float32)
    # Per-probe "alive" flag: 1.0 while the probe still produces a
    # non-trivial Krylov direction, 0.0 once beta has dropped below the
    # invariance tol (or the initial vector was already null). Once a
    # probe is dead we keep its q frozen at 0 and write alpha=0, beta=0
    # for remaining steps; the small (k, k) tridiag eigh then lives on a
    # block-diagonal structure (head = converged subspace, tail = zero
    # block) and the quadrature picks up only the meaningful Ritz pairs
    # from the head. This is essential for matrices with degenerate
    # spectra (e.g. rho = I/N) where Lanczos converges in one step.
    alive = (norms > 1e-30).astype(mx.float32)             # (m,)

    for j in range(k):
        # Three-term recurrence + alpha (compiled, fixed shape)
        alpha, w = _lanczos_step_recurrence(A, q, q_prev, beta_prev)
        alpha = alpha * alive
        alpha_hist.append(alpha)

        # Modified Gram-Schmidt against ALL previous basis vectors per
        # probe. The cost is O(j * N * m) per step; reorth=True is the
        # textbook recipe for stable Lanczos. The shape of Q_stack
        # changes each step, so this part stays uncompiled.
        if reorth and j > 0:
            Q_stack = mx.stack(Q_hist, axis=0)             # (j+1, N, m)
            for _pass in range(2):
                coefs = mx.sum(mx.conj(Q_stack) * w[None, :, :], axis=1)   # (j+1, m)
                correction = mx.sum(coefs[:, None, :] * Q_stack, axis=0)
                w = w - correction

        # Beta + termination + q_new normalisation (compiled, fixed shape)
        beta, q_new, new_alive = _lanczos_step_finalize(w, alive)

        if j < k - 1:
            beta_hist.append(beta)
            q_prev = q
            q = q_new
            Q_hist.append(q)
            beta_prev = beta
            alive = new_alive
        # final step: no q_{k+1} needed; beta_k is not used in T_k

    alphas = mx.stack(alpha_hist, axis=0)                  # (k, m)
    betas = mx.stack(beta_hist, axis=0) if beta_hist else mx.zeros((0, m), dtype=mx.float32)
    Q = mx.stack(Q_hist, axis=0)                           # (k, N, m)
    return alphas, betas, Q, norms


def lanczos_tridiag(
    A: ArrayLike,
    v0: ArrayLike,
    k: int,
    reorth: bool = True,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """k-step Lanczos tridiagonalization of Hermitian A starting from v0.

    Builds an orthonormal basis Q = [q_1, ..., q_kp] (kp <= k) such that
    A Q_kp = Q_kp T_kp + beta_kp q_{kp+1} e_{kp}^T,
    with T_kp the real symmetric tridiagonal (alpha on diag, beta off-diag).

    This thin wrapper drives a single-probe block Lanczos (m=1) on MLX
    and returns numpy arrays for legacy callers / tests. Used for
    correctness checks; the SLQ entry points below call the block path
    directly.

    Parameters
    ----------
    A : array, shape (N, N)
        Hermitian matrix (numpy or mx.array).
    v0 : array, shape (N,)
        Starting vector (need not be unit norm).
    k : int
        Maximum number of Lanczos steps.
    reorth : bool
        If True, do full re-orthogonalisation (MGS) every step.
    tol : float
        Threshold for early termination on small beta.

    Returns
    -------
    alpha : np.ndarray, shape (kp,)
        Real diagonal of T.
    beta : np.ndarray, shape (kp - 1,)
        Real off-diagonal of T.
    kp : int
        Effective number of Lanczos steps actually performed.
    """
    A_mx = _to_mlx(A, mx.complex64)
    if isinstance(v0, mx.array):
        v_mx = v0.astype(mx.complex64)
    else:
        v_arr = np.asarray(v0)
        if not np.iscomplexobj(v_arr):
            v_arr = v_arr.astype(np.complex64)
        else:
            v_arr = v_arr.astype(np.complex64)
        v_mx = mx.array(v_arr)
    if v_mx.ndim != 1:
        raise ValueError("v0 must be 1-D")

    V0 = v_mx[:, None]                                     # (N, 1)
    alphas, betas, _Q, _norms = _block_lanczos_mlx(A_mx, V0, k, reorth=reorth)
    mx.eval(alphas, betas)
    a_np = np.array(alphas[:, 0]).astype(np.float64)
    if betas.shape[0] == 0:
        b_np = np.zeros((0,), dtype=np.float64)
    else:
        b_np = np.array(betas[:, 0]).astype(np.float64)

    # Detect early termination by inspecting beta < tol; we don't actually
    # truncate the MLX recursion (k stays fixed) -- but for backward API
    # compatibility we trim if there is a hard zero in the middle.
    kp = a_np.shape[0]
    if b_np.size > 0:
        small = np.where(b_np < tol)[0]
        if small.size > 0:
            kp = int(small[0]) + 1
            a_np = a_np[:kp]
            b_np = b_np[: max(kp - 1, 0)]
    return a_np, b_np, kp


# ---------------------------------------------------------------------------
#  Quadrature: tridiagonal -> nodes & weights
# ---------------------------------------------------------------------------

def _tridiag_eigh_mlx(alpha: mx.array, beta: mx.array) -> Tuple[mx.array, mx.array]:
    """Eigendecompose real symmetric tridiagonal T (k x k) for each probe.

    alpha shape (k, m), beta shape (k-1, m).

    Returns:
        theta : (k, m) Ritz values
        tau   : (k, m) squared first components U[0, :]^2
    """
    k, m = alpha.shape

    # Build dense T(k, k) for each probe.  We fold the m axis into a batch.
    # T[b, i, j] -> alpha[i, b] on diag, beta[i, b] on (i, i+1) and (i+1, i).
    if k == 1:
        theta = alpha                                       # (1, m)
        tau = mx.ones((1, m), dtype=alpha.dtype)
        return theta, tau

    # alpha_b shape (m, k); beta_b shape (m, k-1)
    alpha_b = mx.transpose(alpha, (1, 0))
    beta_b = mx.transpose(beta, (1, 0))

    # Construct (m, k, k) tridiagonals by scatter via index arrays.
    # Easiest: build with broadcasting + masks.
    diag_eye = mx.eye(k, dtype=alpha.dtype)                # (k, k)
    # diag part
    T = alpha_b[:, :, None] * diag_eye[None, :, :]         # (m, k, k)
    # off-diagonal masks
    upper = mx.eye(k, k=1, dtype=alpha.dtype)              # superdiag mask
    lower = mx.eye(k, k=-1, dtype=alpha.dtype)             # subdiag mask
    # beta_b shape (m, k-1) -> needs to live on offsets 1 and -1
    # Pad beta to length k for easy broadcasting (the last entry is unused).
    beta_pad = mx.concatenate([beta_b, mx.zeros((m, 1), dtype=beta_b.dtype)], axis=1)  # (m, k)
    # superdiag: T[b, i, i+1] = beta[b, i]; pattern positions (0,1),(1,2),...
    # Multiply (m, k) coefficient broadcast against the upper mask.
    # The mask `upper` has 1 at (i, i+1); we need beta[b, i] there.
    # That equals beta_pad[:, :, None] * upper[None, :, :] iff we broadcast on the row axis.
    T = T + beta_pad[:, :, None] * upper[None, :, :]
    # subdiag mirror: T[b, i+1, i] = beta[b, i]; mask `lower` is 1 at (i, i-1) for i>=1,
    # i.e. (i, j) with j = i - 1. So beta_pad shifted by one on the row axis:
    # beta_at_lower[b, i, j] = beta[b, j] when (i, j) is on lower; broadcast by column axis.
    # We can use beta_pad along the column axis and the mask:
    T = T + beta_pad[:, None, :] * lower[None, :, :]

    # Real symmetric eigh on a small k x k batch.  MLX eigh requires a CPU
    # stream and supports complex64 only for hermitian matrices, but real
    # symmetric works on float32.  Cast to float32 for the small problem.
    T_real = mx.real(T).astype(mx.float32)
    theta_b, U_b = mx.linalg.eigh(T_real, stream=mx.cpu)   # (m, k), (m, k, k)

    # tau[b, j] = U_b[b, 0, j]^2
    U0 = U_b[:, 0, :]                                       # (m, k)
    tau_b = U0 * U0                                         # (m, k)

    # Transpose back to (k, m)
    theta = mx.transpose(theta_b, (1, 0))                  # (k, m)
    tau = mx.transpose(tau_b, (1, 0))                      # (k, m)
    return theta, tau


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
    """Estimate Tr[A ln A] for Hermitian PSD A via Stochastic Lanczos Quadrature.

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
    A_mx = _to_mlx(A, mx.complex64)
    N = int(A_mx.shape[-1])

    V = _make_probe_vectors_mlx(N, m, seed=seed, kind="rademacher")   # (N, m)
    alphas, betas, _Q, norms = _block_lanczos_mlx(A_mx, V, k, reorth=reorth)
    theta, tau = _tridiag_eigh_mlx(alphas, betas)                     # (k, m)

    # f(theta) = theta * ln(max(theta, eps))
    eps_arr = mx.array(np.float32(eps))
    theta_safe = mx.maximum(theta, eps_arr)
    f_vals = theta_safe * mx.log(theta_safe)                          # (k, m)

    # Per-probe quadratic form: ||v_i||^2 * sum_j tau_j f(theta_j)
    quad = mx.sum(tau * f_vals, axis=0)                               # (m,)
    weighted = (norms * norms) * quad                                  # (m,)
    est = mx.mean(weighted)
    mx.eval(est)
    return float(est.item())


def von_neumann_entropy_lanczos(
    rho: ArrayLike,
    k: int = 20,
    m: int = 10,
    seed: Optional[int] = None,
    eps: float = _EPS,
    reorth: bool = True,
) -> float:
    """Von Neumann entropy S(rho) = -Tr[rho ln rho] via Stochastic Lanczos.

    Returns a Python float (real, non-negative up to estimator noise).
    """
    return -stochastic_lanczos_logtr(
        rho, k=k, m=m, seed=seed, eps=eps, reorth=reorth
    )


# ---------------------------------------------------------------------------
#  Cross-term Tr[rho ln sigma]  via Hutchinson + Lanczos quadrature
# ---------------------------------------------------------------------------

def _block_lanczos_apply_log_mlx(
    A: mx.array,
    V0: mx.array,
    k: int,
    eps: float = _EPS,
    reorth: bool = True,
) -> mx.array:
    """Block Lanczos approximation of ln(A) @ V0, columnwise.

    For each column v of V0, returns the Lanczos approximation
        ln(A) v approx ||v|| * Q_k ln(T_k) e_1
    Output has the same shape as V0.
    """
    N, m = V0.shape
    alphas, betas, Q, norms = _block_lanczos_mlx(A, V0, k, reorth=reorth)

    # Tridiagonal eigendecomp per probe
    theta, tau = _tridiag_eigh_mlx(alphas, betas)          # (k, m)

    # We need full U per probe, not just U[0, :]. Re-do the eigh to grab U.
    # Cheap (k x k) so we just rebuild and call once.
    k_, m_ = alphas.shape
    if k_ == 1:
        # T = (alpha,), ln(T) e_1 = ln(max(alpha, eps))
        eps_arr = mx.array(np.float32(eps))
        f_vec = mx.log(mx.maximum(alphas[0], eps_arr))     # (m,)
        # f(T) e_1 = (f_vec,) -> shape (1, m)
        f_T_e1 = f_vec[None, :]                            # (1, m)
    else:
        # Reuse the tridiag construction
        alpha_b = mx.transpose(alphas, (1, 0))             # (m, k)
        beta_b = mx.transpose(betas, (1, 0))               # (m, k-1)
        diag_eye = mx.eye(k_, dtype=alphas.dtype)
        T = alpha_b[:, :, None] * diag_eye[None, :, :]
        upper = mx.eye(k_, k=1, dtype=alphas.dtype)
        lower = mx.eye(k_, k=-1, dtype=alphas.dtype)
        beta_pad = mx.concatenate([beta_b, mx.zeros((m_, 1), dtype=beta_b.dtype)], axis=1)
        T = T + beta_pad[:, :, None] * upper[None, :, :]
        T = T + beta_pad[:, None, :] * lower[None, :, :]
        T_real = mx.real(T).astype(mx.float32)
        theta_b, U_b = mx.linalg.eigh(T_real, stream=mx.cpu)   # (m, k), (m, k, k)

        # f(T) e_1 = U diag(f(theta)) U^T e_1 = U (f(theta) * U[0, :])
        eps_arr = mx.array(np.float32(eps))
        f_theta = mx.log(mx.maximum(theta_b, eps_arr))     # (m, k)
        # weighted_U_row = f_theta * U_b[:, 0, :]            # (m, k)
        U0 = U_b[:, 0, :]                                   # (m, k)
        weighted = f_theta * U0                             # (m, k)
        # f(T) e_1 [b, :] = U_b[b, :, :] @ weighted[b, :]
        # = sum_j U_b[b, i, j] * weighted[b, j]
        f_T_e1_b = mx.sum(U_b * weighted[:, None, :], axis=2)   # (m, k)
        # Transpose to (k, m)
        f_T_e1 = mx.transpose(f_T_e1_b, (1, 0))

    # ln(A) v approx ||v|| * Q ( f(T) e_1 ).
    # Q has shape (k, N, m); for each probe b:
    #   Q[:, :, b] @ f_T_e1[:, b]  -> shape (N,)
    # = sum_i Q[i, n, b] * f_T_e1[i, b]    along i
    Q_cast = Q.astype(V0.dtype)
    f_T_e1_c = f_T_e1.astype(V0.dtype)
    out = mx.sum(Q_cast * f_T_e1_c[:, None, :], axis=0)    # (N, m)
    out = out * norms.astype(out.dtype)[None, :]            # scale per probe
    return out


def stochastic_lanczos_cross_logtr(
    rho: ArrayLike,
    sigma: ArrayLike,
    k: int = 20,
    m: int = 10,
    seed: Optional[int] = None,
    eps: float = _EPS,
    reorth: bool = True,
) -> float:
    """Estimate Tr[rho ln sigma] via Hutchinson + Lanczos quadrature on sigma.

    For each probe v we compute u = rho @ v (one O(N^2) matvec) and then
    estimate Re(u^H ln(sigma) v) via a k-step Lanczos run on sigma.

    Returns a Python float.
    """
    rho_mx = _to_mlx(rho, mx.complex64)
    sigma_mx = _to_mlx(sigma, mx.complex64)
    N = int(rho_mx.shape[-1])

    V = _make_probe_vectors_mlx(N, m, seed=seed, kind="rademacher")   # (N, m)

    # Block-apply ln(sigma) to all probes in lockstep
    ln_sig_V = _block_lanczos_apply_log_mlx(sigma_mx, V, k, eps=eps, reorth=reorth)

    # Block matvec rho @ V (single matmul)
    U = rho_mx @ V                                          # (N, m)

    # Per-probe inner product u^H (ln_sig_v):  sum_n conj(u[n, b]) * ln_sig_V[n, b]
    inner = mx.sum(mx.conj(U) * ln_sig_V, axis=0)           # (m,) complex
    est = mx.mean(mx.real(inner))
    mx.eval(est)
    return float(est.item())


def quantum_relative_entropy_lanczos(
    rho: ArrayLike,
    sigma: ArrayLike,
    k: int = 20,
    m: int = 10,
    seed: Optional[int] = None,
    eps: float = _EPS,
    reorth: bool = True,
) -> float:
    """Estimate D(rho || sigma) = Tr[rho (ln rho - ln sigma)] via SLQ.

    The two terms are estimated independently:
        Tr[rho ln rho]    <- stochastic_lanczos_logtr(rho)
        Tr[rho ln sigma]  <- stochastic_lanczos_cross_logtr(rho, sigma)

    Cost: O((2 m) * k * N^2) -- m Lanczos runs on rho plus m on sigma
    (one Lanczos-apply per probe), all batched as block (N, m) matmuls.

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
