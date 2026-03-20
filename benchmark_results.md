# mlx-qre Benchmark Results

## Environment
- Hardware: Apple Silicon (Metal GPU)
- Framework: MLX
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

## Notes

- MLX uses Apple Silicon Metal GPU for eigendecomposition
- NumPy uses Accelerate framework (BLAS) on CPU
- GPU advantage grows with matrix dimension (O(N^3) eigendecomposition)
- Batched computation amortizes GPU kernel launch overhead
- Timings are median of 10 trials with 3 warmup iterations