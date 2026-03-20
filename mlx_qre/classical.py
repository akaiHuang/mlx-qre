"""
Classical information-theoretic divergences on Apple Silicon GPU via MLX.

Provides GPU-accelerated KL divergence and related measures for
classical probability distributions.
"""

import mlx.core as mx
from typing import Optional

_EPS = 1e-30


def kl_divergence(
    p: mx.array,
    q: mx.array,
    eps: float = _EPS,
    axis: int = -1,
) -> mx.array:
    """
    Kullback-Leibler divergence: D_KL(p || q) = sum_i p_i ln(p_i / q_i).

    Fully vectorized on GPU. The classical limit of quantum relative entropy
    when rho and sigma are diagonal.

    Parameters
    ----------
    p : mx.array
        Probability distribution(s), shape (..., K).
    q : mx.array
        Reference distribution(s), shape (..., K).
    eps : float
        Floor to avoid ln(0). Default 1e-30.
    axis : int
        Axis along which to sum. Default -1.

    Returns
    -------
    mx.array
        D_KL(p || q). Scalar or batch of scalars.

    Examples
    --------
    >>> p = mx.array([0.3, 0.7])
    >>> q = mx.array([0.5, 0.5])
    >>> kl_divergence(p, q)
    """
    p_safe = mx.maximum(p, eps)
    q_safe = mx.maximum(q, eps)
    return mx.sum(p_safe * mx.log(p_safe / q_safe), axis=axis)


def jensen_shannon_divergence(
    p: mx.array,
    q: mx.array,
    eps: float = _EPS,
    axis: int = -1,
) -> mx.array:
    """
    Jensen-Shannon divergence: JSD(p || q) = [D_KL(p||m) + D_KL(q||m)] / 2
    where m = (p + q) / 2.

    Symmetric and bounded: 0 <= JSD <= ln(2).

    Parameters
    ----------
    p : mx.array
        Probability distribution(s).
    q : mx.array
        Probability distribution(s).
    eps : float
        Floor for numerical stability.
    axis : int
        Summation axis.

    Returns
    -------
    mx.array
        JSD(p || q).
    """
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m, eps=eps, axis=axis)
                  + kl_divergence(q, m, eps=eps, axis=axis))


def renyi_divergence(
    p: mx.array,
    q: mx.array,
    alpha: float,
    eps: float = _EPS,
    axis: int = -1,
) -> mx.array:
    """
    Renyi divergence of order alpha:
        D_alpha(p || q) = (1/(alpha-1)) ln sum_i p_i^alpha q_i^(1-alpha)

    Converges to KL divergence as alpha -> 1.

    Parameters
    ----------
    p : mx.array
        Probability distribution(s).
    q : mx.array
        Reference distribution(s).
    alpha : float
        Order parameter, alpha > 0 and alpha != 1.
    eps : float
        Numerical floor.
    axis : int
        Summation axis.

    Returns
    -------
    mx.array
        D_alpha(p || q).
    """
    if abs(alpha - 1.0) < 1e-10:
        return kl_divergence(p, q, eps=eps, axis=axis)
    p_safe = mx.maximum(p, eps)
    q_safe = mx.maximum(q, eps)
    integrand = mx.power(p_safe, alpha) * mx.power(q_safe, 1.0 - alpha)
    return mx.log(mx.sum(integrand, axis=axis)) / (alpha - 1.0)


def cross_entropy(
    p: mx.array,
    q: mx.array,
    eps: float = _EPS,
    axis: int = -1,
) -> mx.array:
    """
    Cross entropy: H(p, q) = -sum_i p_i ln q_i = H(p) + D_KL(p || q).

    Parameters
    ----------
    p : mx.array
        True distribution.
    q : mx.array
        Model distribution.
    eps : float
        Numerical floor.
    axis : int
        Summation axis.

    Returns
    -------
    mx.array
        H(p, q).
    """
    q_safe = mx.maximum(q, eps)
    p_safe = mx.maximum(p, eps)
    return -mx.sum(p_safe * mx.log(q_safe), axis=axis)


def shannon_entropy(
    p: mx.array,
    eps: float = _EPS,
    axis: int = -1,
) -> mx.array:
    """
    Shannon entropy: H(p) = -sum_i p_i ln p_i.

    Parameters
    ----------
    p : mx.array
        Probability distribution(s).
    eps : float
        Numerical floor.
    axis : int
        Summation axis.

    Returns
    -------
    mx.array
        H(p).
    """
    p_safe = mx.maximum(p, eps)
    return -mx.sum(p_safe * mx.log(p_safe), axis=axis)
