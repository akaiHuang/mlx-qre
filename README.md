# mlx-qre

**Quantum Relative Entropy + Petz Recovery toolkit** on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

$$\Sigma = D(\rho \| \sigma) = \mathrm{Tr}[\rho(\ln\rho - \ln\sigma)]$$

A complete, self-contained library for quantum information quantities (QRE, von Neumann entropy, Rényi / JSD), quantum channels (thermal attenuator, depolarizing, dephasing) and Petz recovery analysis (recovery map, fidelity, retrodiction). Built as the computational companion to the **Σ = 2 ln Q** entropy-production framework — see [petz-recovery-unification](https://github.com/akaiHuang/petz-recovery-unification) and [tau-chrono](https://github.com/akaiHuang/tau-chrono).

> **Performance note.** Eigendecomposition runs on the Metal GPU via MLX. For **small matrices (N < 500)**, NumPy + Accelerate (CPU) is typically faster due to lower dispatch overhead — use NumPy if you only need a handful of small QREs. The GPU path becomes useful for **batched evaluation** and **N ≥ 500**, where it pulls ahead of NumPy. See [benchmark_results.md](benchmark_results.md) for the full breakdown.

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
| `qre` | `quantum_relative_entropy(rho, sigma)` | D(rho \|\| sigma) via Metal eigendecomposition |
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

Compares MLX (Apple Silicon GPU) vs NumPy (CPU) across matrix sizes N = 10 to 1000. Summary on M1 Max:

| N | MLX (ms) | NumPy (ms) | MLX vs NumPy |
|---:|---:|---:|---:|
| 10 | 0.74 | 0.04 | **0.06×** (NumPy wins) |
| 100 | 3.95 | 3.57 | 0.90× |
| 500 | 158 | 278 | 1.76× |
| 1000 | 1042 | 2010 | 1.93× |

For batched QRE at N=100 the GPU sustains ~460 pairs/sec. **Use NumPy for one-off small problems; reach for `mlx-qre` for batched / large-N work and for the integrated Petz / channel utilities below.** See [benchmark_results.md](benchmark_results.md) for the full table.

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
