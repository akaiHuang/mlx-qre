"""
Core Quantum Relative Entropy computation on Apple Silicon GPU via MLX.

Implements: Sigma = D(rho || sigma) = Tr[rho (ln rho - ln sigma)]

The computation uses eigendecomposition:
    rho = U diag(lambda) U^dag
    ln rho = U diag(ln lambda) U^dag

This is the mathematical backbone for:
    - Gravitational entropy production (Paper 2-3)
    - Petz recovery bound verification (Paper 1)
    - Channel capacity analysis
"""

import mlx.core as mx
import numpy as np
from typing import Optional, Union

# Numerical floor for eigenvalues to avoid ln(0)
_EPS = 1e-30


def _ensure_complex(A: mx.array) -> mx.array:
    """Promote to complex64 if real-valued."""
    if A.dtype in (mx.float32, mx.float16, mx.bfloat16):
        return A.astype(mx.complex64)
    return A


def _eye_complex(n: int) -> mx.array:
    """Create complex64 identity matrix (workaround for MLX GPU scatter)."""
    return mx.eye(n).astype(mx.complex64)


def matrix_log(A: mx.array, eps: float = _EPS) -> mx.array:
    """
    Matrix logarithm via eigendecomposition on GPU.

    For Hermitian A = U diag(lambda) U^dag:
        ln A = U diag(ln lambda) U^dag

    Parameters
    ----------
    A : mx.array
        Hermitian matrix of shape (..., N, N).
    eps : float
        Floor for eigenvalues to prevent ln(0). Default 1e-30.

    Returns
    -------
    mx.array
        Matrix logarithm ln(A), same shape as A.
    """
    A = _ensure_complex(A)
    # MLX eigendecomposition for Hermitian matrices (CPU stream required)
    eigenvalues, eigenvectors = mx.linalg.eigh(A, stream=mx.cpu)
    # Floor small/negative eigenvalues (numerical noise)
    eigenvalues_safe = mx.maximum(eigenvalues, eps)
    log_eigenvalues = mx.log(eigenvalues_safe)
    # Reconstruct: ln A = U diag(ln lambda) U^dag
    # (U @ diag(f))_{ij} = U_{ij} * f_j, then multiply by U^dag
    U = eigenvectors
    U_dag = mx.conj(mx.swapaxes(U, -2, -1))
    # Broadcast: U * f along columns => (..., N, N) * (..., 1, N)
    scaled = U * log_eigenvalues[..., None, :]
    return scaled @ U_dag


def quantum_relative_entropy(
    rho: mx.array,
    sigma: mx.array,
    eps: float = _EPS,
    check_inputs: bool = False,
) -> mx.array:
    """
    Quantum relative entropy: D(rho || sigma) = Tr[rho (ln rho - ln sigma)].

    GPU-accelerated via MLX eigendecomposition on Apple Silicon.

    Parameters
    ----------
    rho : mx.array
        Density matrix (or batch), shape (..., N, N). Must be positive
        semidefinite with Tr = 1.
    sigma : mx.array
        Reference density matrix (or batch), shape (..., N, N). Must be
        positive semidefinite with Tr = 1. Support of rho must be contained
        in support of sigma (otherwise D = +inf).
    eps : float
        Eigenvalue floor to handle numerical zeros. Default 1e-30.
    check_inputs : bool
        If True, verify that inputs are valid density matrices.

    Returns
    -------
    mx.array
        Scalar (or batch of scalars) D(rho || sigma).

    Notes
    -----
    - When sigma has zero eigenvalues where rho has nonzero weight, the result
      is formally +inf. We return a large finite value instead.
    - For pure states rho = |psi><psi|, D(rho || sigma) = -ln <psi|sigma|psi>.
    - Batched: pass shapes (B, N, N) to compute B pairs in parallel on GPU.

    Examples
    --------
    >>> import mlx.core as mx
    >>> rho = mx.array([[0.8, 0.1], [0.1, 0.2]])
    >>> sigma = mx.array([[0.5, 0.0], [0.0, 0.5]])
    >>> D = quantum_relative_entropy(rho, sigma)
    """
    rho = _ensure_complex(rho)
    sigma = _ensure_complex(sigma)

    if check_inputs:
        if not is_density_matrix(rho):
            raise ValueError("rho is not a valid density matrix")
        if not is_density_matrix(sigma):
            raise ValueError("sigma is not a valid density matrix")

    log_rho = matrix_log(rho, eps=eps)
    log_sigma = matrix_log(sigma, eps=eps)

    # D = Tr[rho (ln rho - ln sigma)]
    diff = log_rho - log_sigma
    product = rho @ diff

    # Tr = sum of diagonal elements
    trace = _batch_trace(product)
    # D(rho||sigma) is real for valid density matrices
    return mx.real(trace)


def _batch_trace(A: mx.array) -> mx.array:
    """Trace over last two dimensions, supporting batched arrays."""
    diag = mx.diagonal(A, axis1=-2, axis2=-1)
    return mx.sum(diag, axis=-1)


def is_density_matrix(
    rho: mx.array, atol: float = 1e-5
) -> bool:
    """
    Check if rho is a valid density matrix.

    Criteria:
    - Hermitian: rho = rho^dag
    - Positive semidefinite: all eigenvalues >= 0
    - Unit trace: Tr(rho) = 1

    Parameters
    ----------
    rho : mx.array
        Matrix to check, shape (..., N, N).
    atol : float
        Absolute tolerance for checks.

    Returns
    -------
    bool
        True if rho satisfies all density matrix conditions.
    """
    rho = _ensure_complex(rho)
    # Take the last two dimensions for the check
    # Hermiticity: rho = rho^dag
    rho_dag = mx.conj(mx.swapaxes(rho, -2, -1))
    hermitian_err = mx.max(mx.abs(rho - rho_dag))
    mx.eval(hermitian_err)
    if hermitian_err.item() > atol:
        return False

    # Positive semidefinite (CPU stream required for eigvalsh)
    eigenvalues = mx.linalg.eigvalsh(rho, stream=mx.cpu)
    min_eig = mx.min(eigenvalues)
    mx.eval(min_eig)
    if min_eig.item() < -atol:
        return False

    # Unit trace
    trace = _batch_trace(rho)
    trace_err = mx.max(mx.abs(trace - 1.0))
    mx.eval(trace_err)
    if trace_err.item() > atol:
        return False

    return True


def random_density_matrix(
    n: int,
    batch_size: Optional[int] = None,
    pure: bool = False,
    key: Optional[mx.array] = None,
) -> mx.array:
    """
    Generate random density matrix/matrices (Haar-random) on GPU.

    Uses the method: rho = A A^dag / Tr(A A^dag) where A is a random
    complex Gaussian matrix. For pure states, A is a column vector.

    Parameters
    ----------
    n : int
        Dimension of the Hilbert space.
    batch_size : int, optional
        If provided, generate a batch of density matrices.
    pure : bool
        If True, generate pure states |psi><psi|.
    key : mx.array, optional
        Random key (unused in current MLX, reserved for future).

    Returns
    -------
    mx.array
        Density matrix of shape (N, N) or (batch_size, N, N).
    """
    if pure:
        shape = (batch_size, n, 1) if batch_size else (n, 1)
        psi_real = mx.random.normal(shape)
        psi_imag = mx.random.normal(shape)
        psi = psi_real + 1j * psi_imag
        psi = psi / mx.sqrt(mx.real(mx.sum(mx.conj(psi) * psi, axis=-2, keepdims=True)))
        rho = psi @ mx.conj(mx.swapaxes(psi, -2, -1))
    else:
        shape = (batch_size, n, n) if batch_size else (n, n)
        A_real = mx.random.normal(shape)
        A_imag = mx.random.normal(shape)
        A = A_real + 1j * A_imag
        A_dag = mx.conj(mx.swapaxes(A, -2, -1))
        rho = A @ A_dag
        trace = _batch_trace(rho)
        # Normalize: rho / Tr(rho)
        if batch_size:
            rho = rho / trace[:, None, None]
        else:
            rho = rho / trace

    mx.eval(rho)
    return rho


def von_neumann_entropy(rho: mx.array, eps: float = _EPS) -> mx.array:
    """
    Von Neumann entropy: S(rho) = -Tr[rho ln rho].

    Parameters
    ----------
    rho : mx.array
        Density matrix, shape (..., N, N).
    eps : float
        Eigenvalue floor.

    Returns
    -------
    mx.array
        Scalar (or batch) entropy value.
    """
    rho = _ensure_complex(rho)
    eigenvalues = mx.linalg.eigvalsh(rho, stream=mx.cpu)
    eigenvalues_safe = mx.maximum(mx.real(eigenvalues), eps)
    # S = -sum lambda_i ln(lambda_i)
    return -mx.sum(eigenvalues_safe * mx.log(eigenvalues_safe), axis=-1)


def relative_entropy_pure_state(
    psi: mx.array, sigma: mx.array, eps: float = _EPS
) -> mx.array:
    """
    Efficient QRE for pure state: D(|psi><psi| || sigma) = -ln <psi|sigma|psi>.

    This avoids eigendecomposition of rho (which is rank-1).

    Parameters
    ----------
    psi : mx.array
        State vector, shape (..., N) or (..., N, 1).
    sigma : mx.array
        Reference density matrix, shape (..., N, N).
    eps : float
        Floor for the expectation value.

    Returns
    -------
    mx.array
        D(|psi><psi| || sigma).
    """
    psi = _ensure_complex(psi)
    sigma = _ensure_complex(sigma)
    # Ensure psi is a column vector
    if psi.ndim == 1 or psi.shape[-1] != 1:
        psi = psi[..., :, None]  # (..., N, 1)
    psi_dag = mx.conj(mx.swapaxes(psi, -2, -1))  # (..., 1, N)
    # <psi|sigma|psi>
    expectation = psi_dag @ sigma @ psi  # (..., 1, 1)
    expectation = mx.real(expectation).squeeze(-1).squeeze(-1)
    expectation = mx.maximum(expectation, eps)
    return -mx.log(expectation)
