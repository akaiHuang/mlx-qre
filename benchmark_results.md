# mlx-qre Benchmark Results

## Environment
- Hardware: Apple Silicon (Metal GPU + Accelerate CPU)
- Framework: MLX 0.31.1 / NumPy 2.0.2
- Comparison: NumPy (CPU, Accelerate)

## Single-Pair QRE: D(rho || sigma)

| N | MLX (ms) | NumPy (ms) | Speedup |
|---:|---:|---:|---:|
| 10 | 0.735 | 0.041 | 0.06x |
| 50 | 1.590 | 0.777 | 0.49x |
| 100 | 3.947 | 3.566 | 0.90x |
| 500 | 157.758 | 278.257 | 1.76x |
| 1000 | 1042.270 | 2009.653 | 1.93x |

## Batched QRE (N=100)

| Batch | Total (ms) | Per Pair (ms) | Throughput |
|---:|---:|---:|---:|
| 1 | 3.136 | 3.1355 | 319/s |
| 10 | 23.821 | 2.3821 | 420/s |
| 50 | 110.570 | 2.2114 | 452/s |
| 100 | 217.202 | 2.1720 | 460/s |
| 500 | 1079.567 | 2.1591 | 463/s |

## Notes (Eigh path)

- MLX uses Apple Silicon Metal GPU for eigendecomposition
- NumPy uses Accelerate framework (BLAS) on CPU
- GPU advantage grows with matrix dimension (O(N^3) eigendecomposition)
- Batched computation amortizes GPU kernel launch overhead
- Timings are median of 10 trials with 3 warmup iterations

---

## Stochastic Lanczos Quadrature (`method="lanczos"`)

The Lanczos backend estimates the spectral functions

- `Tr[rho ln rho]` directly via SLQ (one Lanczos run per probe on rho)
- `Tr[rho ln sigma]` via Hutchinson + Lanczos-apply (one Lanczos run per
  probe on sigma plus one O(N^2) `rho @ v` matvec)

at total cost `O((2 m) * k * N^2)` rather than `O(N^3)` for two
eigendecompositions. Inner loop is implemented in NumPy because for the
N regime mlx-qre targets (N <= a few thousand), Accelerate-backed BLAS
matvecs already saturate memory bandwidth on M-series silicon, while
every MLX eval pays a synchronisation cost. We accept `mx.array` inputs
and convert once.

### Single-pair timing  (median of 5 trials, M-series M1 Max)

Default Lanczos parameters: `k=25, m=20`.

| N | MLX exact (ms) | NumPy exact (ms) | SLQ Lanczos (ms) | SLQ vs MLX | SLQ vs NumPy |
|---:|---:|---:|---:|---:|---:|
| 100  |    4.8 |    4.2 |   33.0 | 0.14x | 0.13x |
| 200  |   16.8 |   28.0 |   76.8 | 0.22x | 0.36x |
| 500  |  158.5 |  307.0 |  255.8 | 0.62x | 1.20x |
| 1000 | 1108.8 | 1993.5 |  956.9 | 1.16x | 2.08x |
| 2000 | (GPU timeout on M1 Max) | 21153 | 5816 | -- | 3.64x |

Take-aways:

- **N <= 200**: stay on the eigh path. Lanczos is dispatch-bound and
  slower because the N is too small for `k * N^2` to dominate the small
  number of matvecs in eigh's inner kernels.
- **N >= 1000**: SLQ pulls ahead -- 1.2x vs MLX eigh, 2x vs NumPy eigh,
  growing roughly with N.
- **N >= 2000**: MLX `eigh` runs into Metal command-buffer timeouts on
  M1 Max for complex `(N, N)` matrices; SLQ becomes the only practical
  GPU/Accelerate path. 3.6x faster than NumPy eigh at N=2000.

The 10-50x speedup target from the original SLQ paper assumes very
large sparse / structured A. For *dense* random density matrices the
constant factor is dominated by `rho @ v` and `sigma @ v` matvecs which
already use Accelerate at full speed.

### Stochastic Lanczos accuracy (10 trials per cell)

Relative error vs the exact MLX-eigh value, computed over 10 random
Haar-distributed pairs `(rho, sigma)` per N. `S` denotes the von Neumann
entropy of `rho`; `D` denotes `D(rho || sigma)`.

| N | k | m | S err median | S err max | D err median | D err max |
|---:|---:|---:|---:|---:|---:|---:|
| 50   | 20 | 10 | 0.023 | 0.070 | 0.085 | 0.231 |
| 50   | 25 | 20 | 0.014 | 0.057 | 0.061 | 0.153 |
| 50   | 30 | 30 | 0.009 | 0.029 | 0.057 | 0.121 |
| 100  | 20 | 10 | 0.016 | 0.041 | 0.070 | 0.306 |
| 100  | 25 | 20 | 0.011 | 0.028 | 0.074 | 0.129 |
| 100  | 30 | 30 | 0.007 | 0.025 | 0.038 | 0.132 |
| 500  | 20 | 10 | 0.007 | 0.019 | 0.055 | 0.169 |
| 500  | 25 | 20 | 0.009 | 0.021 | 0.060 | 0.163 |
| 500  | 30 | 30 | 0.010 | 0.016 | 0.064 | 0.145 |
| 1000 | 20 | 10 | 0.008 | 0.014 | 0.040 | 0.100 |
| 1000 | 25 | 20 | 0.004 | 0.010 | 0.048 | 0.090 |
| 1000 | 30 | 30 | 0.003 | 0.010 | 0.043 | 0.088 |

Observations:

- **`S(rho)` (single SLQ on rho)** is consistently within 1-2% median
  rel err and decreases roughly like `1 / sqrt(m)` -- a clean
  Hutchinson + Lanczos-quadrature estimator.
- **`D(rho || sigma)` (cross-term)** sits at 4-8% median rel err.
  The cross-term `Tr[rho ln sigma]` cannot be written as `Tr[g(M)]` for
  a single Hermitian operator `M`, so it requires Hutchinson +
  Lanczos-apply (or polarisation), which has roughly twice the variance
  of the self-term. Beating ~5% on D needs `m >= 30` or task-specific
  variance reduction (e.g. Hutch++).
- Increasing `k` past ~25 does not improve accuracy noticeably for
  random density matrices because the Krylov subspace already contains
  the dominant contributions to `ln`. Accuracy is variance-limited
  (`m`), not approximation-limited (`k`).

### When to use which backend

- Default: `method="exact"` (MLX eigh). Works for `N <= ~1500` on M1 Max,
  produces a true `mx.array` and supports batched inputs.
- `method="lanczos"`: when `N >= 1000` and a 5-10% rel-err estimate is
  acceptable, or when `N >= 2000` where the eigh GPU path times out.
- For batched evaluation across many small `(rho, sigma)` pairs, stay on
  `method="exact"` -- the GPU amortises kernel launches across the batch
  and is the only path that supports batching.
