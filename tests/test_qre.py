"""
Tests for mlx-qre: Quantum Relative Entropy on Apple Silicon GPU.

Run: python -m pytest tests/ -v
"""

import mlx.core as mx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def qubit_states():
    """Common qubit density matrices for testing."""
    # |0><0|
    rho_0 = mx.array([[1.0, 0.0], [0.0, 0.0]], dtype=mx.complex64)
    # |1><1|
    rho_1 = mx.array([[0.0, 0.0], [0.0, 1.0]], dtype=mx.complex64)
    # Maximally mixed
    rho_mm = mx.array([[0.5, 0.0], [0.0, 0.5]], dtype=mx.complex64)
    # |+><+|
    rho_plus = mx.array([[0.5, 0.5], [0.5, 0.5]], dtype=mx.complex64)
    # Thermal-like
    rho_thermal = mx.array([[0.7, 0.1], [0.1, 0.3]], dtype=mx.complex64)
    return {
        "zero": rho_0,
        "one": rho_1,
        "mixed": rho_mm,
        "plus": rho_plus,
        "thermal": rho_thermal,
    }


# ---------------------------------------------------------------------------
# Core QRE Tests
# ---------------------------------------------------------------------------

class TestQuantumRelativeEntropy:
    """Tests for D(rho || sigma) = Tr[rho (ln rho - ln sigma)]."""

    def test_equal_states_gives_zero(self, qubit_states):
        """D(rho || rho) = 0 for any rho."""
        from mlx_qre import quantum_relative_entropy
        rho = qubit_states["thermal"]
        D = quantum_relative_entropy(rho, rho)
        mx.eval(D)
        assert abs(D.item()) < 1e-4, f"D(rho||rho) should be 0, got {D.item()}"

    def test_non_negative(self, qubit_states):
        """D(rho || sigma) >= 0 (Klein's inequality)."""
        from mlx_qre import quantum_relative_entropy
        rho = qubit_states["thermal"]
        sigma = qubit_states["mixed"]
        D = quantum_relative_entropy(rho, sigma)
        mx.eval(D)
        assert D.item() >= -1e-6, f"D should be >= 0, got {D.item()}"

    def test_pure_vs_mixed(self, qubit_states):
        """D(pure || mixed) = ln(d) for maximally mixed sigma."""
        from mlx_qre import quantum_relative_entropy
        rho = qubit_states["plus"]
        sigma = qubit_states["mixed"]
        D = quantum_relative_entropy(rho, sigma)
        mx.eval(D)
        expected = np.log(2.0)  # ln(d) for d=2
        assert abs(D.item() - expected) < 1e-3, (
            f"D(pure || I/2) should be ln(2)={expected:.4f}, got {D.item():.4f}"
        )

    def test_asymmetric(self, qubit_states):
        """D(rho || sigma) != D(sigma || rho) in general."""
        from mlx_qre import quantum_relative_entropy
        rho = qubit_states["thermal"]
        sigma = qubit_states["mixed"]
        D_forward = quantum_relative_entropy(rho, sigma)
        D_reverse = quantum_relative_entropy(sigma, rho)
        mx.eval(D_forward, D_reverse)
        assert abs(D_forward.item() - D_reverse.item()) > 1e-5, (
            "QRE should be asymmetric"
        )

    def test_agrees_with_numpy(self):
        """MLX result matches NumPy reference implementation."""
        from mlx_qre import quantum_relative_entropy
        np.random.seed(42)

        # Random 4x4 density matrices
        A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        rho_np = A @ A.conj().T
        rho_np /= np.trace(rho_np)

        B = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        sigma_np = B @ B.conj().T
        sigma_np /= np.trace(sigma_np)

        # NumPy reference
        eigvals_r, eigvecs_r = np.linalg.eigh(rho_np)
        eigvals_s, eigvecs_s = np.linalg.eigh(sigma_np)
        eigvals_r = np.maximum(eigvals_r, 1e-30)
        eigvals_s = np.maximum(eigvals_s, 1e-30)
        log_rho_np = eigvecs_r @ np.diag(np.log(eigvals_r)) @ eigvecs_r.conj().T
        log_sigma_np = eigvecs_s @ np.diag(np.log(eigvals_s)) @ eigvecs_s.conj().T
        D_np = np.real(np.trace(rho_np @ (log_rho_np - log_sigma_np)))

        # MLX
        rho_mx = mx.array(rho_np.astype(np.complex64))
        sigma_mx = mx.array(sigma_np.astype(np.complex64))
        D_mx = quantum_relative_entropy(rho_mx, sigma_mx)
        mx.eval(D_mx)

        assert abs(D_mx.item() - D_np) < 1e-2, (
            f"MLX={D_mx.item():.6f} vs NumPy={D_np:.6f}"
        )

    def test_batched(self):
        """Batched QRE computes multiple pairs simultaneously."""
        from mlx_qre import quantum_relative_entropy, random_density_matrix
        B = 5
        N = 4
        rho_batch = random_density_matrix(N, batch_size=B)
        sigma_batch = random_density_matrix(N, batch_size=B)

        D_batch = quantum_relative_entropy(rho_batch, sigma_batch)
        mx.eval(D_batch)

        assert D_batch.shape == (B,), f"Expected shape ({B},), got {D_batch.shape}"
        # All should be non-negative
        for i in range(B):
            assert D_batch[i].item() >= -1e-4, f"D[{i}] = {D_batch[i].item()} < 0"

    def test_larger_hilbert_space(self):
        """QRE works for larger Hilbert spaces."""
        from mlx_qre import quantum_relative_entropy, random_density_matrix
        for N in [8, 16, 32]:
            rho = random_density_matrix(N)
            sigma = random_density_matrix(N)
            D = quantum_relative_entropy(rho, sigma)
            mx.eval(D)
            assert D.item() >= -1e-3, f"N={N}: D = {D.item()} < 0"
            D_self = quantum_relative_entropy(rho, rho)
            mx.eval(D_self)
            assert abs(D_self.item()) < 1e-2, f"N={N}: D(rho||rho) = {D_self.item()}"


# ---------------------------------------------------------------------------
# Matrix Log Tests
# ---------------------------------------------------------------------------

class TestMatrixLog:
    """Tests for GPU-accelerated matrix logarithm."""

    def test_identity_log_is_zero(self):
        """ln(I) = 0."""
        from mlx_qre import matrix_log
        I = mx.eye(4).astype(mx.complex64)
        logI = matrix_log(I)
        mx.eval(logI)
        max_err = mx.max(mx.abs(logI)).item()
        assert max_err < 1e-4, f"ln(I) max element = {max_err}, expected ~0"

    def test_diagonal_matrix(self):
        """ln(diag(a,b)) = diag(ln a, ln b)."""
        from mlx_qre import matrix_log
        D = mx.array([[0.3, 0.0], [0.0, 0.7]], dtype=mx.complex64)
        logD = matrix_log(D)
        mx.eval(logD)
        expected_00 = np.log(0.3)
        expected_11 = np.log(0.7)
        assert abs(mx.real(logD[0, 0]).item() - expected_00) < 1e-4
        assert abs(mx.real(logD[1, 1]).item() - expected_11) < 1e-4


# ---------------------------------------------------------------------------
# Density Matrix Utilities
# ---------------------------------------------------------------------------

class TestDensityMatrix:
    """Tests for density matrix utilities."""

    def test_is_density_matrix_valid(self, qubit_states):
        """Valid density matrices pass the check."""
        from mlx_qre import is_density_matrix
        for name, rho in qubit_states.items():
            assert is_density_matrix(rho), f"{name} should be a valid density matrix"

    def test_is_density_matrix_invalid(self):
        """Invalid matrices are detected."""
        from mlx_qre import is_density_matrix
        # Not Hermitian
        A = mx.array([[1.0, 0.5], [0.2, 0.0]], dtype=mx.complex64)
        assert not is_density_matrix(A)
        # Negative eigenvalue
        B = mx.array([[2.0, 0.0], [0.0, -1.0]], dtype=mx.complex64)
        assert not is_density_matrix(B)

    def test_random_density_matrix(self):
        """Random density matrices are valid."""
        from mlx_qre import random_density_matrix, is_density_matrix
        for n in [2, 4, 8]:
            rho = random_density_matrix(n)
            assert is_density_matrix(rho), f"Random {n}x{n} density matrix invalid"

    def test_random_pure_state(self):
        """Random pure states have rank 1."""
        from mlx_qre import random_density_matrix, is_density_matrix
        rho = random_density_matrix(4, pure=True)
        assert is_density_matrix(rho)
        # Pure state: Tr(rho^2) = 1
        rho2 = rho @ rho
        purity = mx.real(mx.sum(mx.diagonal(rho2))).item()
        assert abs(purity - 1.0) < 1e-3, f"Pure state purity = {purity}, expected 1"


# ---------------------------------------------------------------------------
# Classical KL Divergence
# ---------------------------------------------------------------------------

class TestKLDivergence:
    """Tests for classical KL divergence."""

    def test_equal_distributions(self):
        """D_KL(p || p) = 0."""
        from mlx_qre import kl_divergence
        p = mx.array([0.3, 0.7])
        D = kl_divergence(p, p)
        mx.eval(D)
        assert abs(D.item()) < 1e-6

    def test_non_negative(self):
        """D_KL(p || q) >= 0 (Gibbs' inequality)."""
        from mlx_qre import kl_divergence
        p = mx.array([0.2, 0.3, 0.5])
        q = mx.array([0.4, 0.4, 0.2])
        D = kl_divergence(p, q)
        mx.eval(D)
        assert D.item() >= -1e-8

    def test_known_value(self):
        """Compare with manual computation."""
        from mlx_qre import kl_divergence
        p = mx.array([0.5, 0.5])
        q = mx.array([0.25, 0.75])
        D = kl_divergence(p, q)
        mx.eval(D)
        expected = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
        assert abs(D.item() - expected) < 1e-5, (
            f"KL = {D.item()}, expected {expected}"
        )

    def test_batched(self):
        """Batched KL divergence."""
        from mlx_qre import kl_divergence
        p = mx.array([[0.3, 0.7], [0.5, 0.5], [0.9, 0.1]])
        q = mx.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        D = kl_divergence(p, q)
        mx.eval(D)
        assert D.shape == (3,)
        # D_KL(uniform || uniform) = 0
        assert abs(D[1].item()) < 1e-6


# ---------------------------------------------------------------------------
# Quantum Channels
# ---------------------------------------------------------------------------

class TestChannels:
    """Tests for quantum channel operations."""

    def test_identity_channel(self, qubit_states):
        """Identity channel preserves the state."""
        from mlx_qre import apply_channel
        K = [mx.eye(2).astype(mx.complex64)]
        rho = qubit_states["thermal"]
        rho_out = apply_channel(K, rho)
        mx.eval(rho_out)
        err = mx.max(mx.abs(rho_out - rho)).item()
        assert err < 1e-5, f"Identity channel error: {err}"

    def test_thermal_attenuator_trace_preserving(self):
        """Thermal attenuator is trace-preserving."""
        from mlx_qre.channels import thermal_attenuator, verify_trace_preserving
        for eta in [0.0, 0.3, 0.5, 0.8, 1.0]:
            K = thermal_attenuator(eta)
            assert verify_trace_preserving(K), f"eta={eta} not trace-preserving"

    def test_thermal_attenuator_eta_1_is_identity(self, qubit_states):
        """eta=1 thermal attenuator is the identity channel."""
        from mlx_qre import thermal_attenuator, apply_channel
        K = thermal_attenuator(1.0)
        rho = qubit_states["thermal"]
        rho_out = apply_channel(K, rho)
        mx.eval(rho_out)
        err = mx.max(mx.abs(rho_out - rho)).item()
        assert err < 1e-4, f"eta=1 should be identity, error: {err}"

    def test_depolarizing_trace_preserving(self):
        """Depolarizing channel is trace-preserving."""
        from mlx_qre.channels import depolarizing_channel, verify_trace_preserving
        for p in [0.0, 0.1, 0.5, 1.0]:
            K = depolarizing_channel(p)
            assert verify_trace_preserving(K), f"p={p} not trace-preserving"

    def test_entropy_production_non_negative(self, qubit_states):
        """Channel entropy production Sigma >= 0."""
        from mlx_qre import channel_entropy_production, thermal_attenuator
        K = thermal_attenuator(0.5)
        rho = qubit_states["thermal"]
        sigma = qubit_states["mixed"]
        Sigma = channel_entropy_production(K, rho, sigma)
        mx.eval(Sigma)
        # Sigma can be less than D(rho||sigma) but still >= 0
        # (it's the output divergence, not the drop)
        assert Sigma.item() >= -1e-4, f"Sigma = {Sigma.item()} < 0"

    def test_dpi(self, qubit_states):
        """Data Processing Inequality: D(N(rho)||N(sigma)) <= D(rho||sigma)."""
        from mlx_qre import (
            quantum_relative_entropy,
            channel_entropy_production,
            thermal_attenuator,
        )
        K = thermal_attenuator(0.5)
        rho = qubit_states["thermal"]
        sigma = qubit_states["mixed"]
        D_in = quantum_relative_entropy(rho, sigma)
        D_out = channel_entropy_production(K, rho, sigma)
        mx.eval(D_in, D_out)
        assert D_out.item() <= D_in.item() + 1e-4, (
            f"DPI violated: D_out={D_out.item()} > D_in={D_in.item()}"
        )


# ---------------------------------------------------------------------------
# Petz Recovery
# ---------------------------------------------------------------------------

class TestPetzRecovery:
    """Tests for Petz recovery map and fidelity bound."""

    def test_petz_bound_satisfied(self, qubit_states):
        """F(rho, R o N(rho)) >= exp(-Sigma/2)."""
        from mlx_qre import verify_petz_bound, thermal_attenuator
        K = thermal_attenuator(0.7)
        rho = qubit_states["thermal"]
        sigma = qubit_states["mixed"]
        satisfied = verify_petz_bound(rho, sigma, K, verbose=False)
        assert satisfied, "Petz bound violated!"

    def test_petz_bound_multiple_channels(self, qubit_states):
        """Petz bound holds for multiple channel types."""
        from mlx_qre import verify_petz_bound
        from mlx_qre.channels import (
            thermal_attenuator,
            depolarizing_channel,
            dephasing_channel,
        )
        rho = qubit_states["thermal"]
        sigma = qubit_states["mixed"]

        channels = [
            ("thermal_0.3", thermal_attenuator(0.3)),
            ("thermal_0.8", thermal_attenuator(0.8)),
            ("depolarizing_0.2", depolarizing_channel(0.2)),
            ("dephasing_0.5", dephasing_channel(0.5)),
        ]

        for name, K in channels:
            satisfied = verify_petz_bound(rho, sigma, K, verbose=False)
            assert satisfied, f"Petz bound violated for {name}"

    def test_identity_channel_perfect_recovery(self, qubit_states):
        """Identity channel: Sigma=0, F=1, perfect retrodiction."""
        from mlx_qre.petz import petz_recovery_fidelity
        K = [mx.eye(2).astype(mx.complex64)]
        rho = qubit_states["thermal"]
        sigma = qubit_states["mixed"]
        F, Sigma, bound = petz_recovery_fidelity(rho, sigma, K)
        mx.eval(F, Sigma, bound)
        assert abs(Sigma.item()) < 1e-3, f"Sigma should be ~0, got {Sigma.item()}"
        assert abs(F.item() - 1.0) < 1e-2, f"F should be ~1, got {F.item()}"

    def test_retrodiction_quality(self, qubit_states):
        """tau = 1 - F is in [0, 1]."""
        from mlx_qre.petz import retrodiction_quality
        from mlx_qre import thermal_attenuator
        K = thermal_attenuator(0.5)
        rho = qubit_states["thermal"]
        sigma = qubit_states["mixed"]
        tau = retrodiction_quality(rho, sigma, K)
        mx.eval(tau)
        assert 0.0 - 1e-4 <= tau.item() <= 1.0 + 1e-4, (
            f"tau = {tau.item()} outside [0,1]"
        )


# ---------------------------------------------------------------------------
# Von Neumann Entropy
# ---------------------------------------------------------------------------

class TestVonNeumannEntropy:
    """Tests for von Neumann entropy S(rho) = -Tr[rho ln rho]."""

    def test_pure_state_zero_entropy(self, qubit_states):
        """Pure state has S = 0."""
        from mlx_qre.qre import von_neumann_entropy
        rho = qubit_states["zero"]
        S = von_neumann_entropy(rho)
        mx.eval(S)
        assert abs(S.item()) < 1e-3, f"Pure state S = {S.item()}, expected 0"

    def test_maximally_mixed_max_entropy(self, qubit_states):
        """Maximally mixed state has S = ln(d)."""
        from mlx_qre.qre import von_neumann_entropy
        rho = qubit_states["mixed"]
        S = von_neumann_entropy(rho)
        mx.eval(S)
        expected = np.log(2.0)
        assert abs(S.item() - expected) < 1e-3, (
            f"S(I/2) = {S.item()}, expected ln(2) = {expected}"
        )

    def test_entropy_non_negative(self):
        """S(rho) >= 0 for any density matrix."""
        from mlx_qre.qre import von_neumann_entropy
        from mlx_qre import random_density_matrix
        for _ in range(10):
            rho = random_density_matrix(4)
            S = von_neumann_entropy(rho)
            mx.eval(S)
            assert S.item() >= -1e-6, f"S = {S.item()} < 0"


# ---------------------------------------------------------------------------
# Jensen-Shannon
# ---------------------------------------------------------------------------

class TestJensenShannon:
    """Tests for Jensen-Shannon divergence."""

    def test_symmetric(self):
        """JSD(p || q) = JSD(q || p)."""
        from mlx_qre import jensen_shannon_divergence
        p = mx.array([0.3, 0.7])
        q = mx.array([0.6, 0.4])
        jsd_pq = jensen_shannon_divergence(p, q)
        jsd_qp = jensen_shannon_divergence(q, p)
        mx.eval(jsd_pq, jsd_qp)
        assert abs(jsd_pq.item() - jsd_qp.item()) < 1e-6

    def test_bounded(self):
        """0 <= JSD <= ln(2)."""
        from mlx_qre import jensen_shannon_divergence
        p = mx.array([0.1, 0.9])
        q = mx.array([0.9, 0.1])
        jsd = jensen_shannon_divergence(p, q)
        mx.eval(jsd)
        assert 0 <= jsd.item() <= np.log(2.0) + 1e-6


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case handling."""

    def test_near_singular_sigma(self):
        """Handle sigma with near-zero eigenvalues."""
        from mlx_qre import quantum_relative_entropy
        rho = mx.array([[0.5, 0.0], [0.0, 0.5]], dtype=mx.complex64)
        sigma = mx.array([[1.0 - 1e-8, 0.0], [0.0, 1e-8]], dtype=mx.complex64)
        D = quantum_relative_entropy(rho, sigma)
        mx.eval(D)
        # Should return a large but finite value
        assert np.isfinite(D.item()), f"D = {D.item()} is not finite"

    def test_single_dimension(self):
        """1x1 density matrices: D = 0."""
        from mlx_qre import quantum_relative_entropy
        rho = mx.array([[1.0 + 0j]], dtype=mx.complex64)
        sigma = mx.array([[1.0 + 0j]], dtype=mx.complex64)
        D = quantum_relative_entropy(rho, sigma)
        mx.eval(D)
        assert abs(D.item()) < 1e-6

    def test_pure_state_qre_efficient(self):
        """Pure state QRE via efficient formula: D = -ln <psi|sigma|psi>."""
        from mlx_qre.qre import relative_entropy_pure_state, quantum_relative_entropy
        psi = mx.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=mx.complex64)
        sigma = mx.array([[0.7, 0.1], [0.1, 0.3]], dtype=mx.complex64)

        D_efficient = relative_entropy_pure_state(psi, sigma)
        mx.eval(D_efficient)

        rho = mx.array([[0.5, 0.5], [0.5, 0.5]], dtype=mx.complex64)
        D_full = quantum_relative_entropy(rho, sigma)
        mx.eval(D_full)

        assert abs(D_efficient.item() - D_full.item()) < 0.1, (
            f"Efficient={D_efficient.item():.4f} vs Full={D_full.item():.4f}"
        )
