from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# TYPE ALIASES
# =============================================================================

ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]


# =============================================================================
# SINGLE-SPIN OPERATORS
# =============================================================================

IDENTITY_2: ComplexMatrix = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
    ],
    dtype=np.complex128,
)

SIGMA_X: ComplexMatrix = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
    ],
    dtype=np.complex128,
)

SIGMA_Z: ComplexMatrix = np.array(
    [
        [1.0, 0.0],
        [0.0, -1.0],
    ],
    dtype=np.complex128,
)


# =============================================================================
# BASIC FUNCTIONS
# =============================================================================

def build_hamiltonian_l2(J: float, h: float) -> ComplexMatrix:
    """
    Construct the transverse-field Ising Hamiltonian for two spins:

        H = -J sigma_z(1) sigma_z(2)
            -h [sigma_x(1) + sigma_x(2)]

    Open boundary conditions are assumed.

    Basis ordering:
        |up up>, |up down>, |down up>, |down down>
    """

    # sigma_z acting on the first and second spin
    sigma_z_1 = np.kron(SIGMA_Z, IDENTITY_2)
    sigma_z_2 = np.kron(IDENTITY_2, SIGMA_Z)

    # sigma_x acting on the first and second spin
    sigma_x_1 = np.kron(SIGMA_X, IDENTITY_2)
    sigma_x_2 = np.kron(IDENTITY_2, SIGMA_X)

    # Ising interaction term
    interaction_term = -J * (sigma_z_1 @ sigma_z_2)

    # Transverse-field term
    field_term = -h * (sigma_x_1 + sigma_x_2)

    return interaction_term + field_term


def expectation_value(
    state: ComplexVector,
    operator: ComplexMatrix,
) -> complex:
    """
    Compute <state|operator|state>.

    np.vdot(a, b) complex-conjugates the first argument.
    """

    return np.vdot(state, operator @ state)


def verify_hamiltonian(
    hamiltonian: ComplexMatrix,
    J: float,
    h: float,
) -> None:
    """
    Perform independent numerical checks on the Hamiltonian.
    """

    # -------------------------------------------------------------------------
    # Test 1: Hermiticity
    # -------------------------------------------------------------------------

    np.testing.assert_allclose(
        hamiltonian,
        hamiltonian.conj().T,
        atol=1.0e-12,
    )

    # -------------------------------------------------------------------------
    # Test 2: Compare with the matrix derived analytically
    # -------------------------------------------------------------------------

    expected_matrix: ComplexMatrix = np.array(
        [
            [-J, -h, -h, 0.0],
            [-h, J, 0.0, -h],
            [-h, 0.0, J, -h],
            [0.0, -h, -h, -J],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(
        hamiltonian,
        expected_matrix,
        atol=1.0e-12,
    )

    # -------------------------------------------------------------------------
    # Test 3: Compare numerical and analytical eigenvalues
    # -------------------------------------------------------------------------

    numerical_eigenvalues = np.linalg.eigvalsh(hamiltonian)

    energy_scale = np.sqrt(J**2 + 4.0 * h**2)

    analytical_eigenvalues = np.array(
        [
            -energy_scale,
            -J,
            J,
            energy_scale,
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        numerical_eigenvalues,
        analytical_eigenvalues,
        atol=1.0e-12,
    )


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main() -> None:
    np.set_printoptions(
        precision=6,
        suppress=True,
    )

    # Coupling and transverse field
    J = 1.0
    h = 0.0

    hamiltonian = build_hamiltonian_l2(J=J, h=h)

    print("Hamiltonian:")
    print(hamiltonian.real)
    # Run all independent checks
    verify_hamiltonian(
        hamiltonian=hamiltonian,
        J=J,
        h=h,
    )

    print("\nAll Hamiltonian tests passed.")

    # eigh is designed for Hermitian matrices.
    # Eigenvalues are returned in ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    print("\nEigenvalues:")
    print(eigenvalues)

    # The first eigenvector corresponds to the smallest eigenvalue.
    ground_state_energy = eigenvalues[0]
    ground_state = eigenvectors[:, 0]

    print("\nGround-state energy:")
    print(ground_state_energy)

    print("\nGround-state amplitudes:")
    basis_labels = [
        "|up up>",
        "|up down>",
        "|down up>",
        "|down down>",
    ]

    for label, amplitude in zip(basis_labels, ground_state):
        print(f"{label:12s}: {amplitude.real:+.8f}")

    # -------------------------------------------------------------------------
    # Verify normalization and eigenvalue equation
    # -------------------------------------------------------------------------

    norm = np.vdot(ground_state, ground_state).real

    residual = np.linalg.norm(
        hamiltonian @ ground_state
        - ground_state_energy * ground_state
    )

    print("\nGround-state norm:")
    print(norm)

    print("\nEigenvalue-equation residual:")
    print(residual)

    np.testing.assert_allclose(
        norm,
        1.0,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        residual,
        0.0,
        atol=1.0e-12,
    )

    # -------------------------------------------------------------------------
    # Construct observables
    # -------------------------------------------------------------------------

    sigma_x_1 = np.kron(SIGMA_X, IDENTITY_2)
    sigma_x_2 = np.kron(IDENTITY_2, SIGMA_X)

    sigma_z_1 = np.kron(SIGMA_Z, IDENTITY_2)
    sigma_z_2 = np.kron(IDENTITY_2, SIGMA_Z)

    magnetization_x = 0.5 * (sigma_x_1 + sigma_x_2)
    magnetization_z = 0.5 * (sigma_z_1 + sigma_z_2)

    correlation_zz = sigma_z_1 @ sigma_z_2

    expectation_mx = expectation_value(
        ground_state,
        magnetization_x,
    )

    expectation_mz = expectation_value(
        ground_state,
        magnetization_z,
    )

    expectation_czz = expectation_value(
        ground_state,
        correlation_zz,
    )

    print("\nGround-state observables:")
    print(f"<M_x>              = {expectation_mx.real:+.8f}")
    print(f"<M_z>              = {expectation_mz.real:+.8f}")
    print(f"<sigma_z1 sigma_z2> = {expectation_czz.real:+.8f}")


if __name__ == "__main__":
    main()
