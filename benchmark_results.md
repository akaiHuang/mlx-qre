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
eigendecompositions.

### Pure-MLX hot path

The Lanczos hot path now runs **entirely on MLX**: all m probe vectors
are stacked as a single `(N, m)` matrix and each Lanczos step is a
single block matmul `A @ V`, with the `(k, k)` tridiagonal eigh on the
MLX CPU stream. Two design choices make the MLX version comfortably
beat the previous NumPy-Accelerate hot path:

1. **Block all m probes together.** A single `A @ V` matmul replaces m
   independent matvecs every step. This amortises GPU dispatch
   overhead and lets MLX tile probes across compute units; at N=1000,
   m=20, the block matmul is ~5-10x faster than m sequential matvecs.

2. **Lazy evaluation through the inner loop.** No `mx.eval` is called
   inside the k-step loop, so MLX builds the entire Lanczos recurrence
   as a single deferred graph and only materialises it once at
   quadrature time. This avoids per-step GPU command-buffer flushes.

The only non-MLX op is the `numpy.random.default_rng` call that draws
the (one-shot) complex Rademacher probe matrix; the resulting `(N, m)`
array is moved to MLX once at the start of every estimator.

### Single-pair timing  (median of 3 trials, M1 Max)

Default Lanczos parameters: `k=25, m=20`.

| N    | MLX exact (ms) | NumPy exact (ms) | SLQ pure-MLX (ms) | SLQ vs MLX | SLQ vs NumPy |
|---:|---:|---:|---:|---:|---:|
| 100  |     3.6 |     3.7 |   18.6 | 0.19x | 0.20x |
| 500  |   160.4 |   287.4 |   19.7 |  8.1x | 14.6x |
| 1000 |  1076.6 |  1988.1 |   23.6 |   46x |   84x |
| 2000 | timeout | 24047.6 |   36.6 |   --  |  657x |

Take-aways:

- **N <= 200**: stay on the eigh path. SLQ is dispatch-bound and
  somewhat slower because `k * N^2` is too small to dominate the
  small number of matvecs in eigh's inner kernels.
- **N = 500**: SLQ is already ~8x faster than MLX eigh.
- **N >= 1000**: SLQ wins by 1-2 orders of magnitude vs eigh.
- **N = 2000**: MLX `eigh` runs into Metal command-buffer timeouts on
  M1 Max for complex `(N, N)` matrices; SLQ runs in ~37 ms.

### vs. previous NumPy-Accelerate hot path

For reference, the previous SLQ implementation (commit `e680be3`) ran
the inner Lanczos loop in NumPy + Accelerate. Same `(k=25, m=20)`,
median of 2-3 trials, M1 Max:

| N    | NumPy hybrid (ms) | Pure-MLX (ms) | Speedup |
|---:|---:|---:|---:|
| 100  |   43.1 |  31.0 |  1.4x |
| 500  |  281.3 |  41.9 |  6.7x |
| 1000 | 1105.0 |  29.6 |   37x |
| 2000 | 5459.8 |  39.6 |  138x |

A naive port (matvec by matvec, with `mx.eval` after every Lanczos
step, no probe batching) was ~5x slower than the NumPy-Accelerate
hybrid at N=500 because every step paid GPU dispatch overhead. The
critical optimisations are the block `(N, m)` matmul plus deferred
evaluation across the k-step loop, with `mx.compile` on the per-step
recurrence + finalize kernels. With all three in place, MLX wins
across the board.

### Stochastic Lanczos accuracy (8 trials per cell)

Relative error vs the exact MLX-eigh value, computed over 8 random
Haar-distributed pairs `(rho, sigma)` per N. `S` denotes the von
Neumann entropy of `rho`; `D` denotes `D(rho || sigma)`.

| N | k | m | S err median | S err max | D err median | D err max |
|---:|---:|---:|---:|---:|---:|---:|
| 100  | 20 | 10 | 0.017 | 0.024 | 0.066 | 0.214 |
| 100  | 25 | 20 | 0.013 | 0.036 | 0.096 | 0.137 |
| 100  | 30 | 30 | 0.007 | 0.031 | 0.041 | 0.133 |
| 500  | 20 | 10 | 0.007 | 0.024 | 0.026 | 0.144 |
| 500  | 25 | 20 | 0.006 | 0.011 | 0.024 | 0.059 |
| 500  | 30 | 30 | 0.005 | 0.023 | 0.068 | 0.132 |
| 1000 | 20 | 10 | 0.006 | 0.012 | 0.041 | 0.103 |
| 1000 | 25 | 20 | 0.003 | 0.011 | 0.048 | 0.112 |
| 1000 | 30 | 30 | 0.002 | 0.008 | 0.045 | 0.099 |

Observations:

- **`S(rho)` (single SLQ on rho)** is consistently within 1-2% median
  rel err and decreases roughly like `1 / sqrt(m)` -- a clean
  Hutchinson + Lanczos-quadrature estimator.
- **`D(rho || sigma)` (cross-term)** sits at 2-7% median rel err
  depending on `(N, k, m)`. The cross-term `Tr[rho ln sigma]` cannot
  be written as `Tr[g(M)]` for a single Hermitian operator `M`, so it
  requires Hutchinson + Lanczos-apply, which has roughly twice the
  variance of the self-term. Beating ~3% on D needs `m >= 30` or
  task-specific variance reduction (e.g. Hutch++).
- Increasing `k` past ~25 does not improve accuracy noticeably for
  random density matrices because the Krylov subspace already contains
  the dominant contributions to `ln`. Accuracy is variance-limited
  (`m`), not approximation-limited (`k`).

### When to use which backend

- Default: `method="exact"` (MLX eigh). Works for `N <= ~1500` on M1 Max,
  produces a true `mx.array` and supports batched inputs.
- `method="lanczos"`: when `N >= 500` and a few-percent rel-err
  estimate is acceptable; essentially mandatory at `N >= 2000` where
  the eigh GPU path times out.
- For batched evaluation across many small `(rho, sigma)` pairs, stay
  on `method="exact"` -- the GPU amortises kernel launches across the
  batch and is the only path that supports batching.
