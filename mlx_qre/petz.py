"""
Petz recovery map and fidelity bounds on Apple Silicon GPU.

The Petz recovery map is the optimal "undo" operation for a quantum channel.
Given a channel N and reference state sigma:

    R_sigma(.) = sigma^{1/2} N^dag[ N(sigma)^{-1/2} (.) N(sigma)^{-1/2} ] sigma^{1/2}

The key bound (Paper 1):
    F(rho, R o N(rho)) >= exp(-Sigma/2)

where Sigma = D(rho || sigma) - D(N(rho) || N(sigma)) is the entropy drop
through the channel. This quantifies the "cost of retrodiction" in our
framework.

When Sigma = 0 (zero-entropy environment), the Petz map is exact:
    R o N(rho) = rho    (perfect retrodiction, no time arrow)
"""

import mlx.core as mx
from typing import List, Optional, Tuple
from mlx_qre.qre import (
    quantum_relative_entropy,
    _ensure_complex,
    _batch_trace,
    _EPS,
)
from mlx_qre.channels import apply_channel


def _matrix_power(A: mx.array, power: float, eps: float = _EPS) -> mx.array:
    """
    Compute A^power for Hermitian positive semidefinite A via eigendecomposition.

    A^p = U diag(lambda^p) U^dag
    """
    A = _ensure_complex(A)
    eigenvalues, eigenvectors = mx.linalg.eigh(A, stream=mx.cpu)
    eigenvalues_safe = mx.maximum(mx.real(eigenvalues), eps)
    powered = mx.power(eigenvalues_safe, power)
    # Reconstruct: U diag(f) U^dag
    U = eigenvectors
    U_dag = mx.conj(mx.swapaxes(U, -2, -1))
    scaled = U * powered[..., None, :]
    return scaled @ U_dag


def _matrix_sqrt(A: mx.array, eps: float = _EPS) -> mx.array:
    """Matrix square root for Hermitian PSD matrix."""
    return _matrix_power(A, 0.5, eps=eps)


def _matrix_sqrt_inv(A: mx.array, eps: float = _EPS) -> mx.array:
    """Matrix inverse square root for Hermitian PSD matrix."""
    return _matrix_power(A, -0.5, eps=eps)


def fidelity(rho: mx.array, sigma: mx.array, eps: float = _EPS) -> mx.array:
    """
    Quantum fidelity: F(rho, sigma) = [Tr sqrt(sqrt(rho) sigma sqrt(rho))]^2.

    For our framework: F >= exp(-Sigma/2) is the Petz bound.

    Parameters
    ----------
    rho : mx.array
        Density matrix, shape (..., N, N).
    sigma : mx.array
        Density matrix, shape (..., N, N).
    eps : float
        Eigenvalue floor.

    Returns
    -------
    mx.array
        Fidelity F(rho, sigma) in [0, 1].
    """
    rho = _ensure_complex(rho)
    sigma = _ensure_complex(sigma)

    sqrt_rho = _matrix_sqrt(rho, eps=eps)
    # M = sqrt(rho) @ sigma @ sqrt(rho)
    M = sqrt_rho @ sigma @ sqrt_rho
    # eigenvalues of M (CPU stream required)
    eigenvalues = mx.linalg.eigvalsh(M, stream=mx.cpu)
    eigenvalues_safe = mx.maximum(mx.real(eigenvalues), eps)
    sqrt_eigenvalues = mx.sqrt(eigenvalues_safe)
    trace_val = mx.sum(sqrt_eigenvalues, axis=-1)
    return mx.real(trace_val ** 2)


def petz_recovery_map(
    kraus_operators: List[mx.array],
    sigma: mx.array,
    eps: float = _EPS,
) -> List[mx.array]:
    """
    Construct the Petz recovery map for channel N and reference state sigma.

    R_sigma(X) = sigma^{1/2} N^dag[ N(sigma)^{-1/2} X N(sigma)^{-1/2} ] sigma^{1/2}

    In operator form, the Petz recovery Kraus operators are:
        R_i = sigma^{1/2} K_i^dag N(sigma)^{-1/2}

    Parameters
    ----------
    kraus_operators : list of mx.array
        Kraus operators {K_i} of the forward channel N.
    sigma : mx.array
        Reference state, shape (N, N).
    eps : float
        Eigenvalue floor for inversions.

    Returns
    -------
    list of mx.array
        Kraus operators for the Petz recovery map R.
    """
    sigma = _ensure_complex(sigma)

    # sigma^{1/2}
    sigma_sqrt = _matrix_sqrt(sigma, eps=eps)

    # N(sigma) = sum_i K_i sigma K_i^dag
    sigma_out = apply_channel(kraus_operators, sigma)

    # N(sigma)^{-1/2}
    sigma_out_sqrt_inv = _matrix_sqrt_inv(sigma_out, eps=eps)

    # R_i = sigma^{1/2} K_i^dag N(sigma)^{-1/2}
    recovery_kraus = []
    for K in kraus_operators:
        K = _ensure_complex(K)
        K_dag = mx.conj(K.T)
        R_i = sigma_sqrt @ K_dag @ sigma_out_sqrt_inv
        recovery_kraus.append(R_i)

    return recovery_kraus


def petz_recovery_fidelity(
    rho: mx.array,
    sigma: mx.array,
    kraus_operators: List[mx.array],
    eps: float = _EPS,
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Compute the Petz recovery fidelity and verify the bound.

    Returns F(rho, R o N(rho)) and checks F >= exp(-Sigma/2).

    Parameters
    ----------
    rho : mx.array
        Input state, shape (N, N).
    sigma : mx.array
        Reference state, shape (N, N).
    kraus_operators : list of mx.array
        Channel Kraus operators.
    eps : float
        Eigenvalue floor.

    Returns
    -------
    tuple of (F, Sigma, bound)
        F : fidelity F(rho, R o N(rho))
        Sigma : entropy production D(rho||sigma) - D(N(rho)||N(sigma))
        bound : exp(-Sigma/2), the Petz lower bound on F
    """
    rho = _ensure_complex(rho)
    sigma = _ensure_complex(sigma)

    # Forward channel
    rho_out = apply_channel(kraus_operators, rho)
    sigma_out = apply_channel(kraus_operators, sigma)

    # Entropy production: Sigma = D(rho||sigma) - D(N(rho)||N(sigma))
    D_in = quantum_relative_entropy(rho, sigma, eps=eps)
    D_out = quantum_relative_entropy(rho_out, sigma_out, eps=eps)
    Sigma = D_in - D_out
    mx.eval(Sigma)

    # Petz recovery
    recovery_kraus = petz_recovery_map(kraus_operators, sigma, eps=eps)

    # Apply recovery: R(N(rho))
    rho_recovered = apply_channel(recovery_kraus, rho_out)

    # Fidelity F(rho, R(N(rho)))
    F = fidelity(rho, rho_recovered, eps=eps)
    mx.eval(F)

    # Bound: exp(-Sigma/2)
    bound = mx.exp(-Sigma / 2.0)
    mx.eval(bound)

    return F, Sigma, bound


def verify_petz_bound(
    rho: mx.array,
    sigma: mx.array,
    kraus_operators: List[mx.array],
    eps: float = _EPS,
    verbose: bool = True,
) -> bool:
    """
    Verify F(rho, R o N(rho)) >= exp(-Sigma/2).

    This is the central inequality of Paper 1.

    Parameters
    ----------
    rho : mx.array
        Input state.
    sigma : mx.array
        Reference state.
    kraus_operators : list of mx.array
        Channel Kraus operators.
    eps : float
        Eigenvalue floor.
    verbose : bool
        If True, print results.

    Returns
    -------
    bool
        True if the Petz bound is satisfied.
    """
    F, Sigma, bound = petz_recovery_fidelity(rho, sigma, kraus_operators, eps=eps)

    F_val = F.item()
    Sigma_val = Sigma.item()
    bound_val = bound.item()
    satisfied = F_val >= bound_val - 1e-6  # small tolerance for numerics

    if verbose:
        print(f"Petz Recovery Bound Verification")
        print(f"================================")
        print(f"  Entropy production Sigma = {Sigma_val:.6f}")
        print(f"  Recovery fidelity  F     = {F_val:.6f}")
        print(f"  Petz bound  exp(-S/2)    = {bound_val:.6f}")
        print(f"  F >= exp(-S/2)?          {'YES' if satisfied else 'NO'}")
        print(f"  Gap: F - bound           = {F_val - bound_val:.6e}")
        if Sigma_val < 1e-8:
            print(f"  --> Near-zero entropy: perfect retrodiction (no time arrow)")

    return satisfied


def retrodiction_quality(
    rho: mx.array,
    sigma: mx.array,
    kraus_operators: List[mx.array],
    eps: float = _EPS,
) -> mx.array:
    """
    Retrodiction quality: tau = 1 - F(rho, R o N(rho)).

    tau = 0: perfect retrodiction (closed system, no time arrow)
    tau = 1: complete information loss (irreversible, strong time arrow)

    This is the central quantity in our framework connecting
    quantum information to spacetime structure.

    Parameters
    ----------
    rho : mx.array
        Input state.
    sigma : mx.array
        Reference state.
    kraus_operators : list of mx.array
        Channel Kraus operators.
    eps : float
        Eigenvalue floor.

    Returns
    -------
    mx.array
        tau = 1 - F, the retrodiction deficit.
    """
    F, _, _ = petz_recovery_fidelity(rho, sigma, kraus_operators, eps=eps)
    return 1.0 - F
