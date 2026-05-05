# Changelog

## 0.2.0 — 2026-05-05

### Added
- Stochastic Lanczos Quadrature (SLQ) backend for spectral functions of
  density matrices. New public API:
  - `lanczos_tridiag(A, v, k)` — block Lanczos tridiagonalisation on MLX.
  - `stochastic_lanczos_logtr(A, k, m)` — Hutchinson + SLQ estimator of
    `Tr[A ln A]`.
  - `stochastic_lanczos_cross_logtr(rho, sigma, k, m)` — Hutchinson +
    Lanczos-apply estimator of `Tr[rho ln sigma]`.
  - `von_neumann_entropy_lanczos(rho, k, m)` — `S(rho) = -Tr[rho ln rho]`
    via SLQ.
  - `quantum_relative_entropy_lanczos(rho, sigma, k, m)` — `D(rho||sigma)`
    via SLQ for both terms.
- `quantum_relative_entropy` and `von_neumann_entropy` now accept
  `method="lanczos"` to dispatch to the SLQ backend (default remains
  `method="exact"`).

### Changed
- The SLQ hot path runs entirely on MLX. All `m` probe vectors are
  stacked into a single `(N, m)` matrix so each Lanczos step is one
  block matmul `A @ V`; the inner k-step loop never calls `mx.eval`,
  so MLX builds the whole Lanczos recurrence as a deferred graph and
  materialises it once. `mx.compile` is applied to the per-step
  recurrence. The earlier NumPy-Accelerate hybrid path is gone from
  the hot loop (the only NumPy call is the one-shot `default_rng`
  draw of the complex Rademacher probes).

### Performance (M1 Max, single-pair, k=25, m=20)
Pure-MLX SLQ vs the existing eigh paths:

| N    | MLX exact (ms) | NumPy exact (ms) | SLQ pure-MLX (ms) | SLQ vs MLX | SLQ vs NumPy |
|---:|---:|---:|---:|---:|---:|
| 100  |     3.6 |     3.7 |   18.6 | 0.19x | 0.20x |
| 500  |   160.4 |   287.4 |   19.7 |  8.1x | 14.6x |
| 1000 |  1076.6 |  1988.1 |   23.6 |   46x |   84x |
| 2000 | timeout | 24048   |   36.6 |   --  |  657x |

vs the previous NumPy+Accelerate SLQ hybrid (commit `e680be3`):

| N    | NumPy hybrid (ms) | Pure-MLX (ms) | Speedup |
|---:|---:|---:|---:|
| 100  |   43.1 |  31.0 |  1.4x |
| 500  |  281.3 |  41.9 |  6.7x |
| 1000 | 1105.0 |  29.6 |   37x |
| 2000 | 5459.8 |  39.6 |  138x |

### Accuracy
SLQ accuracy at `k=25, m=20`, 8 random Haar pairs per N:
- `S(rho)`: median rel err ~0.3-1.3% across N=100..1000.
- `D(rho || sigma)`: median rel err 2-7% (cross-term is variance-limited;
  drop to <3% needs `m >= 30`).

### When to use which backend
- Default: `method="exact"` (MLX eigh) for `N <= ~1500`. Only path that
  supports batched inputs and produces a true `mx.array` derivative.
- `method="lanczos"`: when `N >= 500` and a few-percent rel-err estimate
  is acceptable. Effectively mandatory at `N >= 2000` where the eigh
  GPU path runs into Metal command-buffer timeouts on M1 Max.

## 0.1.0 — 2026-03-20
- Initial release: GPU-accelerated Quantum Relative Entropy (eigh path)
  and Petz recovery toolkit on Apple Silicon via MLX.
