from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .hamiltonian import build_sparse_hamiltonian
from .solvers import lowest_eigenpairs
from .observables import (
    magnetization_x,
    magnetization_z_moments,
    binder_cumulant,
    half_chain_entanglement_entropy,
    global_x_parity,
)


@dataclass(frozen=True)
class ScanPoint:
    L: int
    h: float
    energy0: float
    gap: float
    mx: float
    mz: float
    mz2: float
    binder: float
    entropy: float
    parity: float


def scan_field(
    L: int,
    h_values,
    J: float = 1.0,
    periodic: bool = False,
) -> list[ScanPoint]:
    """Scan the ground state and first excitation over transverse field."""
    results: list[ScanPoint] = []

    for h in np.asarray(h_values, dtype=float):
        H = build_sparse_hamiltonian(L, J, float(h), periodic)
        spectrum = lowest_eigenpairs(H, k=2)
        psi0 = spectrum.states[:, 0]
        mz, mz2, mz4 = magnetization_z_moments(psi0, L)

        results.append(
            ScanPoint(
                L=L,
                h=float(h),
                energy0=float(spectrum.energies[0]),
                gap=spectrum.gap,
                mx=magnetization_x(psi0, L),
                mz=mz,
                mz2=mz2,
                binder=binder_cumulant(mz2, mz4),
                entropy=half_chain_entanglement_entropy(psi0, L),
                parity=global_x_parity(psi0, L),
            )
        )

    return results


def finite_size_scan(
    L_values,
    h_values,
    J: float = 1.0,
    periodic: bool = False,
) -> dict[int, list[ScanPoint]]:
    """Run the same field scan for several chain lengths."""
    return {
        int(L): scan_field(int(L), h_values, J, periodic)
        for L in L_values
    }
