from __future__ import annotations

import numpy as np


def thermodynamic_dispersion(
    k,
    J: float = 1.0,
    h: float = 1.0,
):
    """Jordan-Wigner/Bogoliubov quasiparticle dispersion."""
    k = np.asarray(k, dtype=float)
    return 2.0 * np.sqrt(
        J * J + h * h - 2.0 * J * h * np.cos(k)
    )


def thermodynamic_gap(
    J: float = 1.0,
    h: float = 1.0,
) -> float:
    """Bulk quasiparticle gap Delta = 2 |J-h|."""
    return 2.0 * abs(J - h)


def thermodynamic_ground_state_energy_density(
    J: float = 1.0,
    h: float = 1.0,
    points: int = 20001,
) -> float:
    """Numerically integrate the exact thermodynamic ground-state energy density."""
    k = np.linspace(0.0, np.pi, points)
    epsilon = thermodynamic_dispersion(k, J, h)
    return float(-np.trapezoid(epsilon, k) / (2.0 * np.pi))
