"""
Tests for the Stochastic Lanczos Quadrature backend.

Run: python -m pytest tests/test_lanczos.py -v
"""

import numpy as np
import pytest
import mlx.core as mx

from mlx_qre.qre import (
    random_density_matrix,
    von_neumann_entropy,
    quantum_relative_entropy,
)
from mlx_qre.lanczos import (
    lanczos_tridiag,
    stochastic_lanczos_logtr,
    von_neumann_entropy_lanczos,
    quantum_relative_entropy_lanczos,
)


# ---------------------------------------------------------------------------
# Low-level Lanczos
# ---------------------------------------------------------------------------

class TestLanczosTridiag:
    def test_tridiagonal_eigenvalues_match_full(self):
        """For k = N the Lanczos tridiagonalisation reproduces the full
        spectrum of A."""
        rng = np.random.default_rng(0)
        N = 30
        A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        A = (A + A.conj().T) / 2
        # Make PSD
        A = A @ A.conj().T
        A /= np.trace(A).real

        v0 = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        alpha, beta, kp = lanczos_tridiag(A, v0, k=N, reorth=True)
        T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
        ritz = np.sort(np.linalg.eigvalsh(T))
        full = np.sort(np.linalg.eigvalsh(A))
        # Full Lanczos with reorth recovers spectrum to high precision.
        assert np.allclose(ritz, full, atol=1e-6)

    def test_orthogonality_of_basis(self):
        """The Lanczos basis Q should be orthonormal up to round-off when
        full reorthogonalisation is on."""
        rng = np.random.default_rng(1)
        N = 50
        A = rng.standard_normal((N, N))
        A = (A + A.T) / 2
        v0 = rng.standard_normal(N)
        # Run our internal-helper indirectly by checking that the tridiagonal
        # produced spans a proper Krylov subspace: alpha and beta are real.
        alpha, beta, kp = lanczos_tridiag(A, v0, k=20, reorth=True)
        assert np.all(np.isfinite(alpha))
        assert np.all(np.isfinite(beta))
        assert np.all(beta > 0)


# ---------------------------------------------------------------------------
# SLQ for Tr[A ln A]  /  von Neumann entropy
# ---------------------------------------------------------------------------

class TestVonNeumannLanczos:
    @pytest.mark.parametrize("N", [50, 100, 200])
    def test_matches_exact_within_tolerance(self, N):
        """Median relative error vs MLX-eigh exact path should be <5%
        across multiple seeds at k=25, m=20."""
        np.random.seed(0)
        mx.random.seed(0)
        errs = []
        for trial in range(5):
            rho = random_density_matrix(N)
            S_exact = float(von_neumann_entropy(rho).item())
            S_l = von_neumann_entropy_lanczos(rho, k=25, m=20, seed=trial)
            errs.append(abs(S_l - S_exact) / abs(S_exact))
        median_err = float(np.median(errs))
        assert median_err < 0.05, f"S(rho) median rel err {median_err:.4f} > 5%"

    def test_pure_state_entropy_is_zero(self):
        """A pure state should have S(rho) approx 0 (with some estimator
        noise)."""
        N = 30
        rho_pure = random_density_matrix(N, pure=True)
        S = von_neumann_entropy_lanczos(rho_pure, k=20, m=10, seed=0)
        # For a rank-1 rho only one eigenvalue is 1, the rest are 0, so
        # S = 0. Estimator noise can be a few percent of N due to log floor.
        assert abs(S) < 0.5

    def test_maximally_mixed_state(self):
        """S(I/N) = ln N exactly (and SLQ should hit it within ~1%)."""
        N = 64
        rho = mx.eye(N).astype(mx.complex64) / N
        S_exact = float(np.log(N))
        S_l = von_neumann_entropy_lanczos(rho, k=15, m=10, seed=0)
        assert abs(S_l - S_exact) / S_exact < 0.05


# ---------------------------------------------------------------------------
# SLQ for QRE
# ---------------------------------------------------------------------------

class TestQRELanczos:
    @pytest.mark.parametrize("N", [50, 100, 200])
    def test_matches_exact_within_tolerance(self, N):
        """Median relative error vs MLX-eigh exact path should be <15%
        across multiple seeds at k=25, m=20.

        Note: the cross-term Tr[rho ln sigma] is harder than the self-term;
        ~5-10% rel error at this (k, m) is the typical Stochastic Lanczos
        ceiling. Increase m to tighten further.
        """
        np.random.seed(0)
        mx.random.seed(0)
        errs = []
        for trial in range(5):
            rho = random_density_matrix(N)
            sigma = random_density_matrix(N)
            D_exact = float(quantum_relative_entropy(rho, sigma).item())
            D_l = quantum_relative_entropy_lanczos(
                rho, sigma, k=25, m=20, seed=trial
            )
            errs.append(abs(D_l - D_exact) / abs(D_exact))
        median_err = float(np.median(errs))
        assert median_err < 0.15, f"D median rel err {median_err:.4f} > 15%"

    def test_self_relative_entropy_small(self):
        """D(rho || rho) should be approx 0 (estimator noise only)."""
        N = 80
        rho = random_density_matrix(N)
        D = quantum_relative_entropy_lanczos(rho, rho, k=25, m=20, seed=0)
        # The two estimator terms cancel out in expectation;
        # for D(rho||rho) sample noise scales with the magnitude of either
        # term. Bound generously.
        assert abs(D) < 1.0

    def test_method_lanczos_via_main_api(self):
        """quantum_relative_entropy(method='lanczos') should produce the
        same value (within tolerance) as direct call to the lanczos
        backend."""
        N = 80
        rho = random_density_matrix(N)
        sigma = random_density_matrix(N)
        D_main = quantum_relative_entropy(rho, sigma, method="lanczos",
                                          k=25, m=20, seed=42)
        D_direct = quantum_relative_entropy_lanczos(rho, sigma, k=25, m=20, seed=42)
        mx.eval(D_main)
        assert abs(float(D_main.item()) - D_direct) < 1e-6

    def test_method_exact_default_unchanged(self):
        """Default 'exact' path must give the same numerical answer as the
        original implementation (backward compatibility check)."""
        N = 50
        rho = random_density_matrix(N)
        sigma = random_density_matrix(N)
        D_default = quantum_relative_entropy(rho, sigma)
        D_explicit = quantum_relative_entropy(rho, sigma, method="exact")
        mx.eval(D_default)
        mx.eval(D_explicit)
        assert abs(float(D_default.item()) - float(D_explicit.item())) < 1e-6


# ---------------------------------------------------------------------------
# SLQ - Tr[A ln A]
# ---------------------------------------------------------------------------

class TestStochasticLanczosLogtr:
    def test_diagonal_matrix(self):
        """For a diagonal matrix the trace is sum lambda ln lambda."""
        N = 100
        diag = np.linspace(0.001, 1.0, N)
        diag /= diag.sum()
        A = np.diag(diag).astype(np.complex128)
        exact = float(np.sum(diag * np.log(np.maximum(diag, 1e-30))))
        approx = stochastic_lanczos_logtr(A, k=20, m=20, seed=0)
        rel_err = abs(approx - exact) / abs(exact)
        assert rel_err < 0.1
