from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh


@dataclass(frozen=True)
class Spectrum:
    energies: np.ndarray
    states: np.ndarray

    @property
    def gap(self) -> float:
        if len(self.energies) < 2:
            return float("nan")
        return float(self.energies[1] - self.energies[0])


def lowest_eigenpairs(H, k: int = 2) -> Spectrum:
    """Return the k lowest eigenpairs of a Hermitian Hamiltonian."""
    n = H.shape[0]
    if H.shape[0] != H.shape[1]:
        raise ValueError("Hamiltonian must be square")

    if n <= 4 or k >= n - 1 or not issparse(H):
        dense = H.toarray() if issparse(H) else H
        vals, vecs = np.linalg.eigh(dense)
        order = np.argsort(vals)[:k]
        return Spectrum(vals[order], vecs[:, order])

    vals, vecs = eigsh(
        H,
        k=k,
        which="SA",
        tol=1.0e-11,
        maxiter=max(1000, 20 * n),
    )
    order = np.argsort(vals)
    return Spectrum(vals[order], vecs[:, order])


def ground_state(H) -> tuple[float, np.ndarray]:
    """Return ground-state energy and normalized state vector."""
    spec = lowest_eigenpairs(H, k=1)
    return float(spec.energies[0]), spec.states[:, 0]
