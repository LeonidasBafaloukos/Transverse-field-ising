from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, dok_matrix


def _validate(L: int, J: float, h: float) -> None:
    if L < 1:
        raise ValueError("L must be >= 1")
    if not np.isfinite(J) or not np.isfinite(h):
        raise ValueError("J and h must be finite")


def build_sparse_hamiltonian(
    L: int,
    J: float = 1.0,
    h: float = 1.0,
    periodic: bool = False,
) -> csr_matrix:
    """Build H = -J sum Z_i Z_(i+1) - h sum X_i in the Z basis."""
    _validate(L, J, h)
    dim = 1 << L
    H = dok_matrix((dim, dim), dtype=np.float64)

    for state in range(dim):
        zz_sum = 0.0
        bonds = L if (periodic and L > 1) else L - 1
        for i in range(bonds):
            j = (i + 1) % L
            zi = 1.0 if ((state >> i) & 1) == 0 else -1.0
            zj = 1.0 if ((state >> j) & 1) == 0 else -1.0
            zz_sum += zi * zj
        H[state, state] += -J * zz_sum

        for i in range(L):
            flipped = state ^ (1 << i)
            H[flipped, state] += -h

    return H.tocsr()


def build_dense_hamiltonian(
    L: int,
    J: float = 1.0,
    h: float = 1.0,
    periodic: bool = False,
) -> np.ndarray:
    """Dense counterpart of build_sparse_hamiltonian."""
    return build_sparse_hamiltonian(L, J, h, periodic).toarray()
