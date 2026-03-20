# mlx-qre

GPU-accelerated **Quantum Relative Entropy** on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

$$\Sigma = D(\rho \| \sigma) = \mathrm{Tr}[\rho(\ln\rho - \ln\sigma)]$$

## Installation

```bash
pip install -e .
```

Requires Python 3.10+ and Apple Silicon (M1/M2/M3/M4).

## Quick Start

```python
import mlx.core as mx
from mlx_qre import quantum_relative_entropy, random_density_matrix

# Two 100x100 density matrices on GPU
rho = random_density_matrix(100)
sigma = random_density_matrix(100)

# Compute D(rho || sigma) — eigendecomposition runs on Metal GPU
D = quantum_relative_entropy(rho, sigma)
mx.eval(D)
print(f"D(rho || sigma) = {D.item():.6f}")

# Batched: 500 pairs simultaneously
rho_batch = random_density_matrix(50, batch_size=500)
sigma_batch = random_density_matrix(50, batch_size=500)
D_batch = quantum_relative_entropy(rho_batch, sigma_batch)
```

## Features

| Module | Function | Description |
|--------|----------|-------------|
| `qre` | `quantum_relative_entropy(rho, sigma)` | D(rho \|\| sigma) via GPU eigendecomposition |
| `qre` | `von_neumann_entropy(rho)` | S(rho) = -Tr[rho ln rho] |
| `qre` | `relative_entropy_pure_state(psi, sigma)` | Efficient D for pure states: -ln(psi\|sigma\|psi) |
| `classical` | `kl_divergence(p, q)` | Classical KL divergence |
| `classical` | `jensen_shannon_divergence(p, q)` | Symmetric JSD |
| `classical` | `renyi_divergence(p, q, alpha)` | Renyi divergence of order alpha |
| `channels` | `thermal_attenuator(eta)` | Gravitational channel eta = -g_00 |
| `channels` | `channel_entropy_production(K, rho, sigma)` | Sigma through channel |
| `channels` | `depolarizing_channel(p)` | Depolarizing noise |
| `channels` | `dephasing_channel(gamma)` | Dephasing noise |
| `petz` | `petz_recovery_map(K, sigma)` | Construct Petz recovery R |
| `petz` | `petz_recovery_fidelity(rho, sigma, K)` | F(rho, R o N(rho)) |
| `petz` | `verify_petz_bound(rho, sigma, K)` | Check F >= exp(-Sigma/2) |
| `petz` | `retrodiction_quality(rho, sigma, K)` | tau = 1 - F |

## Use Cases

- **Gravitational entropy production**: Sigma_grav = D(N_eta(rho) || N_eta(sigma)) with eta = 1/Q^2
- **Quantum channel analysis**: entropy production, data processing inequality
- **Petz recovery bounds**: F >= exp(-Sigma/2), retrodiction quality
- **Quantum ML**: kernel methods using QRE as a distance measure
- **Neural entropy**: EEG/neural signal entropy production analysis

## Benchmark

```bash
python -m mlx_qre.benchmark
```

Compares MLX (Apple Silicon GPU) vs NumPy (CPU) across matrix sizes N = 10 to 1000.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Theory

The quantum relative entropy D(rho || sigma) is the quantum generalization of KL divergence. In the retrocausality framework:

- **Sigma = 2 ln Q**: unified entropy production formula
- **Petz bound**: F >= exp(-Sigma/2) quantifies retrodiction cost
- **tau = 1 - F**: retrodiction deficit (0 = perfect, 1 = irreversible)
- **Zero-entropy limit**: Sigma -> 0 implies perfect retrodiction (no time arrow)

## License

MIT License. Copyright (c) 2026 Sheng-Kai Huang.
