"""
Quantum channel operations and entropy production on Apple Silicon GPU.

Implements:
    - Channel application via Kraus operators: N(rho) = sum_i K_i rho K_i^dag
    - Entropy production: Sigma = D(N(rho) || N(sigma))
    - Thermal attenuator channel (gravitational channel eta = 1/Q^2)
    - Standard channels: depolarizing, dephasing, amplitude damping

The entropy production Sigma through a quantum channel is the central
quantity in the retrocausality framework:
    Sigma_grav = D(N_grav(rho) || N_grav(sigma_thermal))
"""

import mlx.core as mx
import numpy as np
from typing import List, Optional, Tuple
from mlx_qre.qre import quantum_relative_entropy, _ensure_complex, _batch_trace, _eye_complex

_EPS = 1e-30


def apply_channel(
    kraus_operators: List[mx.array],
    rho: mx.array,
) -> mx.array:
    """
    Apply quantum channel via Kraus representation.

    N(rho) = sum_i K_i rho K_i^dag

    Parameters
    ----------
    kraus_operators : list of mx.array
        Kraus operators {K_i}, each of shape (d_out, d_in).
        Must satisfy sum_i K_i^dag K_i = I (trace-preserving).
    rho : mx.array
        Input density matrix, shape (..., d_in, d_in).

    Returns
    -------
    mx.array
        Output density matrix N(rho), shape (..., d_out, d_out).
    """
    rho = _ensure_complex(rho)
    result = None

    for i, K in enumerate(kraus_operators):
        K = _ensure_complex(K)
        K_dag = mx.conj(K.T)
        term = K @ rho @ K_dag
        if i == 0:
            result = term
        else:
            result = result + term

    return result


def channel_entropy_production(
    kraus_operators: List[mx.array],
    rho: mx.array,
    sigma: Optional[mx.array] = None,
    eps: float = _EPS,
) -> mx.array:
    """
    Entropy production through a quantum channel.

    Sigma = D(N(rho) || N(sigma))

    This is the core quantity: for the gravitational channel,
    Sigma_grav = D(N_eta(rho) || N_eta(sigma_thermal)).

    By the data processing inequality, Sigma <= D(rho || sigma),
    with equality iff the Petz recovery map is exact.

    Parameters
    ----------
    kraus_operators : list of mx.array
        Kraus operators defining the channel N.
    rho : mx.array
        Input state, shape (..., N, N).
    sigma : mx.array, optional
        Reference state. If None, uses maximally mixed state I/N.
    eps : float
        Eigenvalue floor.

    Returns
    -------
    mx.array
        Sigma = D(N(rho) || N(sigma)).
    """
    N = rho.shape[-1]
    if sigma is None:
        sigma = _eye_complex(N) / N
        if rho.ndim > 2:
            sigma = mx.broadcast_to(sigma, rho.shape)

    rho_out = apply_channel(kraus_operators, rho)
    sigma_out = apply_channel(kraus_operators, sigma)

    return quantum_relative_entropy(rho_out, sigma_out, eps=eps)


def _make_matrix_from_entries(n: int, entries: list) -> mx.array:
    """
    Create complex64 matrix from list of (row, col, value) entries.

    Workaround for MLX GPU scatter not supporting complex64.
    Constructs via NumPy then converts.
    """
    mat = np.zeros((n, n), dtype=np.complex64)
    for r, c, v in entries:
        mat[r, c] = v
    return mx.array(mat)


def thermal_attenuator(eta: float, n_dim: int = 2) -> List[mx.array]:
    """
    Thermal attenuator (beam splitter) channel with transmissivity eta.

    For a qubit (n_dim=2), the Kraus operators are:
        K_0 = [[1, 0], [0, sqrt(eta)]]
        K_1 = [[0, sqrt(1-eta)], [0, 0]]

    In our gravitational context: eta = 1/Q^2 = -g_00 (for static metrics).
    The channel represents information loss due to gravitational redshift.

    Parameters
    ----------
    eta : float
        Transmissivity, 0 <= eta <= 1.
        eta = 1: identity (no loss, flat spacetime)
        eta = 0: complete erasure (horizon)
    n_dim : int
        Hilbert space dimension. Default 2 (qubit).

    Returns
    -------
    list of mx.array
        Kraus operators for the thermal attenuator.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"Transmissivity eta must be in [0, 1], got {eta}")

    if n_dim == 2:
        sqrt_eta = float(np.sqrt(eta))
        sqrt_1_eta = float(np.sqrt(1.0 - eta))

        K0 = mx.array([
            [1.0 + 0j, 0.0 + 0j],
            [0.0 + 0j, sqrt_eta + 0j],
        ], dtype=mx.complex64)

        K1 = mx.array([
            [0.0 + 0j, sqrt_1_eta + 0j],
            [0.0 + 0j, 0.0 + 0j],
        ], dtype=mx.complex64)

        return [K0, K1]
    else:
        # Generalized: K_0 attenuates higher levels by sqrt(eta)^level
        K0_entries = [(i, i, eta ** (i / 2.0) + 0j) for i in range(n_dim)]
        K0 = _make_matrix_from_entries(n_dim, K0_entries)

        kraus_list = [K0]
        for k in range(1, n_dim):
            coeff = (1.0 - eta ** k) ** 0.5
            Kk = _make_matrix_from_entries(n_dim, [(k - 1, k, coeff + 0j)])
            kraus_list.append(Kk)

        return kraus_list


def depolarizing_channel(p: float, n_dim: int = 2) -> List[mx.array]:
    """
    Depolarizing channel: N(rho) = (1-p) rho + p I/d.

    Parameters
    ----------
    p : float
        Depolarizing parameter, 0 <= p <= 1.
    n_dim : int
        Hilbert space dimension.

    Returns
    -------
    list of mx.array
        Kraus operators.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")

    d = n_dim
    if d == 2:
        I2 = _eye_complex(2)
        sx = mx.array([[0, 1], [1, 0]], dtype=mx.complex64)
        sy = mx.array([[0, -1j], [1j, 0]], dtype=mx.complex64)
        sz = mx.array([[1, 0], [0, -1]], dtype=mx.complex64)

        c0 = (1.0 - 3.0 * p / 4.0) ** 0.5
        c1 = (p / 4.0) ** 0.5

        return [c0 * I2, c1 * sx, c1 * sy, c1 * sz]
    else:
        K0_coeff = (1.0 - p + p / d**2) ** 0.5
        Kij_coeff = (p / d**2) ** 0.5

        kraus_list = [K0_coeff * _eye_complex(d)]
        for m in range(d):
            for n in range(d):
                if m == 0 and n == 0:
                    continue
                K = _make_matrix_from_entries(d, [(m, n, Kij_coeff + 0j)])
                kraus_list.append(K)

        return kraus_list


def dephasing_channel(gamma: float, n_dim: int = 2) -> List[mx.array]:
    """
    Dephasing channel: kills off-diagonal elements by factor (1-gamma).

    For qubit:
        K_0 = sqrt(1 - gamma/2) I
        K_1 = sqrt(gamma/2) sigma_z

    Parameters
    ----------
    gamma : float
        Dephasing strength, 0 <= gamma <= 1.
    n_dim : int
        Hilbert space dimension.

    Returns
    -------
    list of mx.array
        Kraus operators.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    if n_dim == 2:
        I2 = _eye_complex(2)
        sz = mx.array([[1, 0], [0, -1]], dtype=mx.complex64)
        c0 = (1.0 - gamma / 2.0) ** 0.5
        c1 = (gamma / 2.0) ** 0.5
        return [c0 * I2, c1 * sz]
    else:
        K0_coeff = (1.0 - gamma) ** 0.5
        K_diag_coeff = (gamma / n_dim) ** 0.5
        kraus_list = [K0_coeff * _eye_complex(n_dim)]
        for k in range(n_dim):
            K = _make_matrix_from_entries(n_dim, [(k, k, K_diag_coeff + 0j)])
            kraus_list.append(K)
        return kraus_list


def verify_trace_preserving(kraus_operators: List[mx.array], atol: float = 1e-5) -> bool:
    """
    Verify sum_i K_i^dag K_i = I (trace-preserving condition).

    Parameters
    ----------
    kraus_operators : list of mx.array
        Kraus operators to verify.
    atol : float
        Tolerance.

    Returns
    -------
    bool
        True if trace-preserving within tolerance.
    """
    d = kraus_operators[0].shape[-1]
    total = mx.zeros((d, d), dtype=mx.complex64)
    for K in kraus_operators:
        K = _ensure_complex(K)
        total = total + mx.conj(K.T) @ K
    identity = _eye_complex(d)
    err = mx.max(mx.abs(total - identity))
    mx.eval(err)
    return err.item() < atol
