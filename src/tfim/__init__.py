"""
Tools for exact-diagonalization studies of the one-dimensional
transverse-field Ising model (TFIM).
"""

from .hamiltonian import build_dense_hamiltonian, build_sparse_hamiltonian
from .observables import (
    expectation,
    magnetization_x,
    magnetization_z_moments,
    zz_correlation,
    correlation_profile,
    half_chain_entanglement_entropy,
    global_x_parity,
    binder_cumulant,
)
from .solvers import Spectrum, lowest_eigenpairs, ground_state
from .analysis import ScanPoint, scan_field, finite_size_scan
from .exact import (
    thermodynamic_dispersion,
    thermodynamic_gap,
    thermodynamic_ground_state_energy_density,
)

__all__ = [
    "build_dense_hamiltonian",
    "build_sparse_hamiltonian",
    "expectation",
    "magnetization_x",
    "magnetization_z_moments",
    "zz_correlation",
    "correlation_profile",
    "half_chain_entanglement_entropy",
    "global_x_parity",
    "binder_cumulant",
    "Spectrum",
    "lowest_eigenpairs",
    "ground_state",
    "ScanPoint",
    "scan_field",
    "finite_size_scan",
    "thermodynamic_dispersion",
    "thermodynamic_gap",
    "thermodynamic_ground_state_energy_density",
]
