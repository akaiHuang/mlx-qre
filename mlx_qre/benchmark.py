"""
Benchmark: MLX (GPU) vs NumPy (CPU) for Quantum Relative Entropy.

Run: python -m mlx_qre.benchmark
"""

import time
import mlx.core as mx
import numpy as np
from typing import List, Tuple, Optional


def _random_density_matrix_np(n: int) -> np.ndarray:
    """Generate random density matrix using NumPy (CPU)."""
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    rho = A @ A.conj().T
    rho /= np.trace(rho)
    return rho


def _qre_numpy(rho: np.ndarray, sigma: np.ndarray, eps: float = 1e-30) -> float:
    """Quantum relative entropy using NumPy (CPU baseline)."""
    # Eigendecomposition
    eigvals_rho, eigvecs_rho = np.linalg.eigh(rho)
    eigvals_sigma, eigvecs_sigma = np.linalg.eigh(sigma)

    # Floor eigenvalues
    eigvals_rho = np.maximum(eigvals_rho, eps)
    eigvals_sigma = np.maximum(eigvals_sigma, eps)

    # Matrix logarithms
    log_rho = eigvecs_rho @ np.diag(np.log(eigvals_rho)) @ eigvecs_rho.conj().T
    log_sigma = eigvecs_sigma @ np.diag(np.log(eigvals_sigma)) @ eigvecs_sigma.conj().T

    # D = Tr[rho (ln rho - ln sigma)]
    diff = log_rho - log_sigma
    result = np.trace(rho @ diff)
    return np.real(result)


def _random_density_matrix_mlx(n: int) -> mx.array:
    """Generate random density matrix using MLX (GPU)."""
    A_real = mx.random.normal((n, n))
    A_imag = mx.random.normal((n, n))
    A = A_real + 1j * A_imag
    rho = A @ mx.conj(A.T)
    trace = mx.sum(mx.diagonal(rho))
    rho = rho / trace
    mx.eval(rho)
    return rho


def _qre_mlx(rho: mx.array, sigma: mx.array, eps: float = 1e-30) -> mx.array:
    """Quantum relative entropy using MLX (GPU)."""
    from mlx_qre.qre import quantum_relative_entropy
    result = quantum_relative_entropy(rho, sigma, eps=eps)
    mx.eval(result)
    return result


def benchmark_single(
    n: int,
    n_trials: int = 10,
    warmup: int = 3,
) -> Tuple[float, float, float]:
    """
    Benchmark QRE for a single matrix size.

    Parameters
    ----------
    n : int
        Matrix dimension.
    n_trials : int
        Number of timed trials.
    warmup : int
        Number of warmup iterations (not timed).

    Returns
    -------
    tuple of (mlx_time_ms, numpy_time_ms, speedup)
    """
    # --- NumPy (CPU) ---
    rho_np = _random_density_matrix_np(n)
    sigma_np = _random_density_matrix_np(n)

    # Warmup
    for _ in range(warmup):
        _qre_numpy(rho_np, sigma_np)

    # Time
    times_np = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _qre_numpy(rho_np, sigma_np)
        t1 = time.perf_counter()
        times_np.append((t1 - t0) * 1000)
    avg_np = np.median(times_np)

    # --- MLX (GPU) ---
    rho_mx = _random_density_matrix_mlx(n)
    sigma_mx = _random_density_matrix_mlx(n)

    # Warmup
    for _ in range(warmup):
        _qre_mlx(rho_mx, sigma_mx)

    # Time
    times_mx = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _qre_mlx(rho_mx, sigma_mx)
        t1 = time.perf_counter()
        times_mx.append((t1 - t0) * 1000)
    avg_mx = np.median(times_mx)

    speedup = avg_np / avg_mx if avg_mx > 0 else float('inf')

    return avg_mx, avg_np, speedup


def run_benchmark(
    sizes: Optional[List[int]] = None,
    n_trials: int = 10,
    warmup: int = 3,
    verbose: bool = True,
) -> List[dict]:
    """
    Run full benchmark suite.

    Parameters
    ----------
    sizes : list of int
        Matrix dimensions to benchmark. Default [10, 50, 100, 500, 1000].
    n_trials : int
        Number of trials per size.
    warmup : int
        Warmup iterations.
    verbose : bool
        Print results.

    Returns
    -------
    list of dict
        Benchmark results.
    """
    if sizes is None:
        sizes = [10, 50, 100, 500, 1000]

    results = []

    if verbose:
        print("=" * 68)
        print("  mlx-qre Benchmark: Quantum Relative Entropy")
        print("  MLX (Apple Silicon GPU) vs NumPy (CPU)")
        print("=" * 68)
        print(f"  Trials: {n_trials} | Warmup: {warmup}")
        print("-" * 68)
        print(f"  {'N':>6}  {'MLX (ms)':>12}  {'NumPy (ms)':>12}  {'Speedup':>10}")
        print("-" * 68)

    for n in sizes:
        try:
            mlx_ms, np_ms, speedup = benchmark_single(n, n_trials, warmup)
            result = {
                "N": n,
                "mlx_ms": mlx_ms,
                "numpy_ms": np_ms,
                "speedup": speedup,
            }
            results.append(result)
            if verbose:
                print(f"  {n:>6}  {mlx_ms:>12.3f}  {np_ms:>12.3f}  {speedup:>9.2f}x")
        except Exception as e:
            if verbose:
                print(f"  {n:>6}  FAILED: {e}")
            results.append({"N": n, "error": str(e)})

    if verbose:
        print("-" * 68)
        print("  Note: Speedup > 1 means MLX is faster.")
        print("  GPU advantage grows with matrix size (eigendecomposition).")
        print("=" * 68)

    return results


def run_batch_benchmark(
    n: int = 100,
    batch_sizes: Optional[List[int]] = None,
    n_trials: int = 10,
    warmup: int = 3,
    verbose: bool = True,
) -> List[dict]:
    """
    Benchmark batched QRE computation.

    Parameters
    ----------
    n : int
        Matrix dimension.
    batch_sizes : list of int
        Batch sizes to benchmark.
    n_trials : int
        Number of trials.
    warmup : int
        Warmup iterations.
    verbose : bool
        Print results.

    Returns
    -------
    list of dict
        Benchmark results.
    """
    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100, 500]

    results = []
    from mlx_qre.qre import random_density_matrix, quantum_relative_entropy

    if verbose:
        print("=" * 68)
        print(f"  Batched QRE Benchmark (N={n})")
        print("=" * 68)
        print(f"  {'Batch':>8}  {'Total (ms)':>12}  {'Per pair (ms)':>14}  {'Throughput':>12}")
        print("-" * 68)

    for B in batch_sizes:
        try:
            rho = random_density_matrix(n, batch_size=B)
            sigma = random_density_matrix(n, batch_size=B)

            # Warmup
            for _ in range(warmup):
                d = quantum_relative_entropy(rho, sigma)
                mx.eval(d)

            # Time
            times = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                d = quantum_relative_entropy(rho, sigma)
                mx.eval(d)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            avg_ms = np.median(times)
            per_pair = avg_ms / B
            throughput = B / (avg_ms / 1000)

            result = {
                "batch_size": B,
                "total_ms": avg_ms,
                "per_pair_ms": per_pair,
                "throughput_per_sec": throughput,
            }
            results.append(result)
            if verbose:
                print(f"  {B:>8}  {avg_ms:>12.3f}  {per_pair:>14.4f}  {throughput:>10.0f}/s")
        except Exception as e:
            if verbose:
                print(f"  {B:>8}  FAILED: {e}")
            results.append({"batch_size": B, "error": str(e)})

    if verbose:
        print("-" * 68)
        print("=" * 68)

    return results


def generate_markdown_report(
    single_results: List[dict],
    batch_results: Optional[List[dict]] = None,
) -> str:
    """Generate markdown benchmark report."""
    lines = [
        "# mlx-qre Benchmark Results",
        "",
        "## Environment",
        "- Hardware: Apple Silicon (Metal GPU)",
        "- Framework: MLX",
        "- Comparison: NumPy (CPU, Accelerate)",
        "",
        "## Single-Pair QRE: D(rho || sigma)",
        "",
        "| N | MLX (ms) | NumPy (ms) | Speedup |",
        "|---:|---:|---:|---:|",
    ]
    for r in single_results:
        if "error" in r:
            lines.append(f"| {r['N']} | ERROR | ERROR | - |")
        else:
            lines.append(
                f"| {r['N']} | {r['mlx_ms']:.3f} | {r['numpy_ms']:.3f} "
                f"| {r['speedup']:.2f}x |"
            )

    if batch_results:
        lines.extend([
            "",
            "## Batched QRE (N=100)",
            "",
            "| Batch | Total (ms) | Per Pair (ms) | Throughput |",
            "|---:|---:|---:|---:|",
        ])
        for r in batch_results:
            if "error" in r:
                lines.append(f"| {r['batch_size']} | ERROR | ERROR | - |")
            else:
                lines.append(
                    f"| {r['batch_size']} | {r['total_ms']:.3f} "
                    f"| {r['per_pair_ms']:.4f} | {r['throughput_per_sec']:.0f}/s |"
                )

    lines.extend([
        "",
        "## Notes",
        "",
        "- MLX uses Apple Silicon Metal GPU for eigendecomposition",
        "- NumPy uses Accelerate framework (BLAS) on CPU",
        "- GPU advantage grows with matrix dimension (O(N^3) eigendecomposition)",
        "- Batched computation amortizes GPU kernel launch overhead",
        "- Timings are median of 10 trials with 3 warmup iterations",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    print()
    single = run_benchmark()
    print()
    batch = run_batch_benchmark()
    print()

    report = generate_markdown_report(single, batch)
    report_path = "benchmark_results.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")
