from __future__ import annotations

import numpy as np


def expectation(state: np.ndarray, operator) -> complex:
    """Return <psi|O|psi>."""
    return np.vdot(state, operator @ state)


def _z_value(basis_state: int, site: int) -> float:
    return 1.0 if ((basis_state >> site) & 1) == 0 else -1.0


def magnetization_x(state: np.ndarray, L: int) -> float:
    """Return <M_x> = (1/L) sum_i <sigma_i^x>."""
    value = 0.0 + 0.0j
    indices = np.arange(state.size)
    for i in range(L):
        flipped = indices ^ (1 << i)
        value += np.vdot(state, state[flipped])
    return float((value / L).real)


def magnetization_z_moments(
    state: np.ndarray,
    L: int,
) -> tuple[float, float, float]:
    """Return <M_z>, <M_z^2>, <M_z^4>."""
    probabilities = np.abs(state) ** 2
    mz_values = np.empty(state.size, dtype=float)

    for basis_state in range(state.size):
        mz_values[basis_state] = (
            sum(_z_value(basis_state, i) for i in range(L)) / L
        )

    return (
        float(np.dot(probabilities, mz_values)),
        float(np.dot(probabilities, mz_values**2)),
        float(np.dot(probabilities, mz_values**4)),
    )


def zz_correlation(state: np.ndarray, i: int, j: int) -> float:
    """Return <sigma_i^z sigma_j^z>."""
    probabilities = np.abs(state) ** 2
    values = np.array(
        [
            _z_value(basis_state, i) * _z_value(basis_state, j)
            for basis_state in range(state.size)
        ],
        dtype=float,
    )
    return float(np.dot(probabilities, values))


def correlation_profile(
    state: np.ndarray,
    L: int,
    origin: int | None = None,
) -> np.ndarray:
    """Return <sigma_origin^z sigma_j^z> for every site j."""
    if origin is None:
        origin = L // 2
    return np.array(
        [zz_correlation(state, origin, j) for j in range(L)],
        dtype=float,
    )


def binder_cumulant(mz2: float, mz4: float) -> float:
    """Return U4 = 1 - <Mz^4> / (3 <Mz^2>^2)."""
    if mz2 <= 1.0e-15:
        return float("nan")
    return float(1.0 - mz4 / (3.0 * mz2 * mz2))


def half_chain_entanglement_entropy(
    state: np.ndarray,
    L: int,
    base: float = 2.0,
) -> float:
    """Von Neumann entropy across the central bipartition."""
    cut = L // 2
    psi_matrix = state.reshape((1 << cut, 1 << (L - cut)))
    singular_values = np.linalg.svd(psi_matrix, compute_uv=False)
    probabilities = singular_values**2
    probabilities = probabilities[probabilities > 1.0e-15]
    return float(
        -np.sum(probabilities * np.log(probabilities)) / np.log(base)
    )


def global_x_parity(state: np.ndarray, L: int) -> float:
    """Return expectation value of P = product_i sigma_i^x."""
    indices = np.arange(state.size)
    globally_flipped = indices ^ ((1 << L) - 1)
    return float(np.vdot(state, state[globally_flipped]).real)
