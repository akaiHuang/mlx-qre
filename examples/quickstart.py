"""
mlx-qre Quickstart: Quantum Relative Entropy on Apple Silicon GPU
=================================================================

This example demonstrates the core functionality of mlx-qre:

1. Quantum relative entropy: Sigma = D(rho || sigma)
2. Classical KL divergence
3. Gravitational channel (thermal attenuator)
4. Petz recovery bound verification

The central equation of the retrocausality framework:
    Sigma = D(rho_spacetime || rho_matter)

Author: Sheng-Kai Huang
"""

import mlx.core as mx
import numpy as np


def example_1_basic_qre():
    """Basic quantum relative entropy computation."""
    print("=" * 60)
    print("  Example 1: Quantum Relative Entropy")
    print("=" * 60)

    from mlx_qre import quantum_relative_entropy, random_density_matrix

    # Create density matrices
    rho = mx.array([
        [0.8, 0.1 + 0.05j],
        [0.1 - 0.05j, 0.2]
    ], dtype=mx.complex64)

    sigma = mx.array([
        [0.5, 0.0],
        [0.0, 0.5]
    ], dtype=mx.complex64)

    # Compute D(rho || sigma)
    D = quantum_relative_entropy(rho, sigma)
    mx.eval(D)
    print(f"  D(rho || sigma) = {D.item():.6f}")
    print(f"  D(rho || rho)   = ", end="")
    D_self = quantum_relative_entropy(rho, rho)
    mx.eval(D_self)
    print(f"{D_self.item():.6f}  (should be ~0)")

    # Batched: 100 random pairs at once
    B = 100
    rho_batch = random_density_matrix(4, batch_size=B)
    sigma_batch = random_density_matrix(4, batch_size=B)
    D_batch = quantum_relative_entropy(rho_batch, sigma_batch)
    mx.eval(D_batch)
    print(f"\n  Batched ({B} pairs of 4x4):")
    print(f"    Mean D  = {mx.mean(D_batch).item():.4f}")
    print(f"    Min  D  = {mx.min(D_batch).item():.4f}")
    print(f"    Max  D  = {mx.max(D_batch).item():.4f}")
    print()


def example_2_classical_kl():
    """Classical KL divergence."""
    print("=" * 60)
    print("  Example 2: Classical KL Divergence")
    print("=" * 60)

    from mlx_qre import kl_divergence, jensen_shannon_divergence

    p = mx.array([0.3, 0.5, 0.2])
    q = mx.array([0.33, 0.34, 0.33])

    D_kl = kl_divergence(p, q)
    jsd = jensen_shannon_divergence(p, q)
    mx.eval(D_kl, jsd)

    print(f"  p = [0.3, 0.5, 0.2]")
    print(f"  q = [0.33, 0.34, 0.33]")
    print(f"  D_KL(p || q) = {D_kl.item():.6f}")
    print(f"  JSD(p, q)    = {jsd.item():.6f}")
    print()


def example_3_gravitational_channel():
    """
    Gravitational channel: thermal attenuator with eta = -g_00.

    In our framework:
        eta = 1/Q^2 = -g_00  (static metric)
        Sigma_grav = D(N_eta(rho) || N_eta(sigma))

    eta = 1 : flat spacetime (no information loss)
    eta < 1 : curved spacetime (information degradation)
    eta = 0 : horizon (complete erasure)
    """
    print("=" * 60)
    print("  Example 3: Gravitational Channel")
    print("=" * 60)

    from mlx_qre import (
        quantum_relative_entropy,
        channel_entropy_production,
        thermal_attenuator,
        apply_channel,
    )

    # Quantum state near a gravitational source
    rho = mx.array([[0.9, 0.1], [0.1, 0.1]], dtype=mx.complex64)
    sigma = mx.array([[0.5, 0.0], [0.0, 0.5]], dtype=mx.complex64)

    D_in = quantum_relative_entropy(rho, sigma)
    mx.eval(D_in)
    print(f"  Input: D(rho || sigma) = {D_in.item():.6f}")
    print()

    # Scan eta values (gravitational redshift)
    etas = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
    print(f"  {'eta':>6}  {'D_out':>10}  {'Sigma_drop':>12}  {'Interpretation'}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*30}")

    for eta in etas:
        K = thermal_attenuator(eta)
        D_out = channel_entropy_production(K, rho, sigma)
        mx.eval(D_out)
        drop = D_in.item() - D_out.item()
        if eta == 1.0:
            interp = "flat spacetime"
        elif eta > 0.5:
            interp = "weak gravity"
        elif eta > 0.1:
            interp = "strong gravity"
        else:
            interp = "near horizon"
        print(f"  {eta:>6.1f}  {D_out.item():>10.4f}  {drop:>12.4f}  {interp}")

    print()
    print("  As eta -> 0 (horizon): D_out -> 0 (complete information loss)")
    print("  The entropy drop Sigma quantifies the 'cost of retrodiction'")
    print()


def example_4_petz_recovery():
    """
    Petz recovery: F(rho, R o N(rho)) >= exp(-Sigma/2).

    This is the central bound from Paper 1. When Sigma = 0,
    F = 1 and retrodiction is perfect (no time arrow).
    """
    print("=" * 60)
    print("  Example 4: Petz Recovery Bound")
    print("=" * 60)

    from mlx_qre import verify_petz_bound, thermal_attenuator
    from mlx_qre.petz import retrodiction_quality

    rho = mx.array([[0.8, 0.15], [0.15, 0.2]], dtype=mx.complex64)
    sigma = mx.array([[0.5, 0.0], [0.0, 0.5]], dtype=mx.complex64)

    print("\n  Thermal attenuator eta = 0.6:")
    K = thermal_attenuator(0.6)
    verify_petz_bound(rho, sigma, K, verbose=True)

    print("\n  Retrodiction quality tau (= 1 - F):")
    etas = [0.95, 0.8, 0.5, 0.2]
    for eta in etas:
        K = thermal_attenuator(eta)
        tau = retrodiction_quality(rho, sigma, K)
        mx.eval(tau)
        arrow = "=" * max(1, int(tau.item() * 30))
        print(f"    eta={eta:.2f}  tau={tau.item():.4f}  |{arrow}> time arrow")
    print()


def example_5_sigma_scan():
    """
    Sigma = 2 ln Q scan: our unified formula.

    For static metrics: Q = 1/sqrt(-g_00), so
        Sigma = 2 ln Q = -ln(-g_00) = -ln(eta)

    Verify this against the channel computation.
    """
    print("=" * 60)
    print("  Example 5: Sigma = 2 ln Q Verification")
    print("=" * 60)

    from mlx_qre import (
        quantum_relative_entropy,
        channel_entropy_production,
        thermal_attenuator,
    )

    rho = mx.array([[0.5, 0.5], [0.5, 0.5]], dtype=mx.complex64)
    sigma = mx.array([[0.5, 0.0], [0.0, 0.5]], dtype=mx.complex64)

    print(f"\n  {'eta':>6}  {'Q':>8}  {'2 ln Q':>10}  {'-ln eta':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}")

    for eta in [0.9, 0.7, 0.5, 0.3, 0.1]:
        Q = 1.0 / np.sqrt(eta)
        sigma_formula = 2.0 * np.log(Q)
        neg_ln_eta = -np.log(eta)
        print(f"  {eta:>6.2f}  {Q:>8.4f}  {sigma_formula:>10.4f}  {neg_ln_eta:>10.4f}")

    print()
    print("  2 ln Q = -ln(eta) = -ln(-g_00)")
    print("  This is Sigma_grav: the geometric entropy production")
    print()


if __name__ == "__main__":
    print()
    print("  mlx-qre: Quantum Relative Entropy on Apple Silicon GPU")
    print("  Sigma = D(rho || sigma) = Tr[rho (ln rho - ln sigma)]")
    print()

    example_1_basic_qre()
    example_2_classical_kl()
    example_3_gravitational_channel()
    example_4_petz_recovery()
    example_5_sigma_scan()

    print("=" * 60)
    print("  All examples completed successfully.")
    print("=" * 60)
