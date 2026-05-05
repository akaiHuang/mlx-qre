"""
mlx-qre: GPU-accelerated Quantum Relative Entropy on Apple Silicon
==================================================================

Core formula: Sigma = D(rho || sigma) = Tr[rho (ln rho - ln sigma)]

This package provides GPU-accelerated computation of quantum information
quantities using Apple's MLX framework, with a focus on:

- Quantum relative entropy (QRE)
- Classical KL divergence
- Quantum channel entropy production
- Petz recovery map and fidelity bounds

Author: Sheng-Kai Huang <akai@fawstudio.com>
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Sheng-Kai Huang"

from mlx_qre.qre import (
    quantum_relative_entropy,
    matrix_log,
    is_density_matrix,
    random_density_matrix,
    von_neumann_entropy,
)
from mlx_qre.lanczos import (
    lanczos_tridiag,
    stochastic_lanczos_logtr,
    von_neumann_entropy_lanczos,
    stochastic_lanczos_cross_logtr,
    quantum_relative_entropy_lanczos,
)
from mlx_qre.classical import (
    kl_divergence,
    jensen_shannon_divergence,
)
from mlx_qre.channels import (
    apply_channel,
    channel_entropy_production,
    thermal_attenuator,
    depolarizing_channel,
    dephasing_channel,
)
from mlx_qre.petz import (
    petz_recovery_map,
    petz_recovery_fidelity,
    verify_petz_bound,
)

__all__ = [
    # Core QRE
    "quantum_relative_entropy",
    "matrix_log",
    "is_density_matrix",
    "random_density_matrix",
    "von_neumann_entropy",
    # Stochastic Lanczos estimators
    "lanczos_tridiag",
    "stochastic_lanczos_logtr",
    "von_neumann_entropy_lanczos",
    "stochastic_lanczos_cross_logtr",
    "quantum_relative_entropy_lanczos",
    # Classical
    "kl_divergence",
    "jensen_shannon_divergence",
    # Channels
    "apply_channel",
    "channel_entropy_production",
    "thermal_attenuator",
    "depolarizing_channel",
    "dephasing_channel",
    # Petz recovery
    "petz_recovery_map",
    "petz_recovery_fidelity",
    "verify_petz_bound",
]
