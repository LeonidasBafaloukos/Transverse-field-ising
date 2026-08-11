import numpy as np

from tfim.exact import thermodynamic_gap
from tfim.hamiltonian import build_dense_hamiltonian, build_sparse_hamiltonian
from tfim.observables import magnetization_x, magnetization_z_moments
from tfim.solvers import lowest_eigenpairs


def test_l2_analytical_spectrum():
    J, h = 1.3, 0.7
    values = np.linalg.eigvalsh(build_dense_hamiltonian(2, J, h))
    scale = np.sqrt(J**2 + 4.0 * h**2)
    np.testing.assert_allclose(
        values,
        [-scale, -J, J, scale],
        atol=1.0e-12,
    )


def test_exact_limits():
    L = 4
    values = np.linalg.eigvalsh(build_dense_hamiltonian(L, 1.2, 0.0))
    np.testing.assert_allclose(values[:2], [-3.6, -3.6], atol=1.0e-12)

    values = np.linalg.eigvalsh(build_dense_hamiltonian(L, 0.0, 0.7))
    np.testing.assert_allclose(values[0], -2.8, atol=1.0e-12)


def test_sparse_matches_dense():
    sparse = build_sparse_hamiltonian(5, 1.0, 0.83).toarray()
    dense = build_dense_hamiltonian(5, 1.0, 0.83)
    np.testing.assert_allclose(sparse, dense)


def test_observable_limits():
    L = 6
    spectrum = lowest_eigenpairs(
        build_sparse_hamiltonian(L, 0.0, 1.0),
        k=2,
    )
    state = spectrum.states[:, 0]
    assert magnetization_x(state, L) > 0.999999

    _, mz2, _ = magnetization_z_moments(state, L)
    assert abs(mz2 - 1.0 / L) < 1.0e-10


def test_thermodynamic_gap():
    assert thermodynamic_gap(1.0, 1.0) == 0.0
    assert thermodynamic_gap(1.0, 1.5) == 1.0
