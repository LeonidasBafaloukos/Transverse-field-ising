from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# TYPE ALIASES
# =============================================================================

ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]
RealVector = NDArray[np.float64]


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

SIGMA_Y: ComplexMatrix = np.array(
    [
        [0.0, -1.0j],
        [1.0j, 0.0],
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
# COMPUTATIONAL BASIS
# =============================================================================

def computational_basis_state(
    index: int,
    number_of_spins: int,
) -> ComplexVector:
    """
    Construct one computational-basis vector.

    Convention:
        binary digit 0 -> |up>
        binary digit 1 -> |down>

    For L = 3:

        index 0 -> |up up up>
        index 1 -> |up up down>
        index 2 -> |up down up>
        index 3 -> |up down down>
        ...
    """

    if number_of_spins < 1:
        raise ValueError("number_of_spins must be at least 1.")

    dimension = 2**number_of_spins

    if index < 0 or index >= dimension:
        raise ValueError(
            f"index must satisfy 0 <= index < {dimension}."
        )

    state: ComplexVector = np.zeros(
        dimension,
        dtype=np.complex128,
    )

    state[index] = 1.0

    return state


def basis_label(
    index: int,
    number_of_spins: int,
) -> str:
    """
    Return a readable label for a computational-basis state.
    """

    dimension = 2**number_of_spins

    if index < 0 or index >= dimension:
        raise ValueError(
            f"index must satisfy 0 <= index < {dimension}."
        )

    bits = format(index, f"0{number_of_spins}b")

    spin_labels = [
        "up" if bit == "0" else "down"
        for bit in bits
    ]

    return "|" + " ".join(spin_labels) + ">"


# =============================================================================
# GENERAL MANY-BODY OPERATORS
# =============================================================================

def operator_on_site(
    operator: ComplexMatrix,
    site: int,
    number_of_spins: int,
) -> ComplexMatrix:
    """
    Embed a single-spin operator into an L-spin Hilbert space.

    The returned operator acts with `operator` on the selected site and
    with the identity on every other site.

    Python convention:
        site = 0, 1, ..., L - 1

    Example:
        operator_on_site(SIGMA_X, site=1, number_of_spins=3)

    constructs

        I tensor sigma_x tensor I.
    """

    if number_of_spins < 1:
        raise ValueError("number_of_spins must be at least 1.")

    if site < 0 or site >= number_of_spins:
        raise ValueError(
            f"site must satisfy 0 <= site < {number_of_spins}."
        )

    if operator.shape != (2, 2):
        raise ValueError("The local operator must have shape (2, 2).")

    full_operator: ComplexMatrix = np.array(
        [[1.0]],
        dtype=np.complex128,
    )

    for current_site in range(number_of_spins):
        if current_site == site:
            local_factor = operator
        else:
            local_factor = IDENTITY_2

        full_operator = np.kron(
            full_operator,
            local_factor,
        )

    return full_operator


# =============================================================================
# GENERAL TFIM HAMILTONIAN
# =============================================================================

def build_hamiltonian(
    number_of_spins: int,
    J: float,
    h: float,
) -> ComplexMatrix:
    """
    Construct the open-boundary transverse-field Ising Hamiltonian

        H = -J sum_i Z_i Z_(i+1) - h sum_i X_i.

    There are L - 1 interaction bonds for an open chain.
    """

    if number_of_spins < 1:
        raise ValueError("number_of_spins must be at least 1.")

    if not np.isfinite(J):
        raise ValueError("J must be finite.")

    if not np.isfinite(h):
        raise ValueError("h must be finite.")

    dimension = 2**number_of_spins

    hamiltonian: ComplexMatrix = np.zeros(
        (dimension, dimension),
        dtype=np.complex128,
    )

    # Precompute the embedded local operators.
    x_operators = [
        operator_on_site(
            operator=SIGMA_X,
            site=site,
            number_of_spins=number_of_spins,
        )
        for site in range(number_of_spins)
    ]

    z_operators = [
        operator_on_site(
            operator=SIGMA_Z,
            site=site,
            number_of_spins=number_of_spins,
        )
        for site in range(number_of_spins)
    ]

    # -------------------------------------------------------------------------
    # Ising interaction
    #
    # Open boundary conditions:
    #
    #     (0, 1), (1, 2), ..., (L - 2, L - 1)
    #
    # There is no bond between the last and first sites.
    # -------------------------------------------------------------------------

    for site in range(number_of_spins - 1):
        interaction_operator = (
            z_operators[site]
            @ z_operators[site + 1]
        )

        hamiltonian += -J * interaction_operator

    # -------------------------------------------------------------------------
    # Transverse-field term
    # -------------------------------------------------------------------------

    for site in range(number_of_spins):
        hamiltonian += -h * x_operators[site]

    return hamiltonian


def diagonalize_hamiltonian(
    hamiltonian: ComplexMatrix,
) -> tuple[RealVector, ComplexMatrix]:
    """
    Diagonalize a Hermitian Hamiltonian.

    The eigenvalues are returned in ascending order.
    The corresponding eigenvectors are stored as columns.
    """

    if hamiltonian.ndim != 2:
        raise ValueError("The Hamiltonian must be a matrix.")

    if hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("The Hamiltonian must be square.")

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    return eigenvalues, eigenvectors


# =============================================================================
# TEST 1: LOCAL-OPERATOR CONSTRUCTION
# =============================================================================

def verify_local_operator_construction() -> None:
    """
    Verify the action and algebra of embedded Pauli operators.
    """

    number_of_spins = 3
    dimension = 2**number_of_spins

    sigma_x_middle = operator_on_site(
        operator=SIGMA_X,
        site=1,
        number_of_spins=number_of_spins,
    )

    # Correct matrix dimensions.
    if sigma_x_middle.shape != (dimension, dimension):
        raise AssertionError(
            "The many-body operator has incorrect dimensions."
        )

    # Hermiticity.
    np.testing.assert_allclose(
        sigma_x_middle,
        sigma_x_middle.conj().T,
        atol=1.0e-12,
    )

    # X_i squared must equal the full identity.
    np.testing.assert_allclose(
        sigma_x_middle @ sigma_x_middle,
        np.eye(dimension, dtype=np.complex128),
        atol=1.0e-12,
    )

    # Test:
    #
    #     X_2 |up up down> = |up down down>.
    #
    initial_index = 1
    expected_final_index = 3

    initial_state = computational_basis_state(
        index=initial_index,
        number_of_spins=number_of_spins,
    )

    expected_final_state = computational_basis_state(
        index=expected_final_index,
        number_of_spins=number_of_spins,
    )

    calculated_final_state = sigma_x_middle @ initial_state

    np.testing.assert_allclose(
        calculated_final_state,
        expected_final_state,
        atol=1.0e-12,
    )

    # Operators on different sites commute.
    sigma_x_site_0 = operator_on_site(
        operator=SIGMA_X,
        site=0,
        number_of_spins=number_of_spins,
    )

    sigma_z_site_2 = operator_on_site(
        operator=SIGMA_Z,
        site=2,
        number_of_spins=number_of_spins,
    )

    different_site_commutator = (
        sigma_x_site_0 @ sigma_z_site_2
        - sigma_z_site_2 @ sigma_x_site_0
    )

    np.testing.assert_allclose(
        different_site_commutator,
        np.zeros(
            (dimension, dimension),
            dtype=np.complex128,
        ),
        atol=1.0e-12,
    )

    # On the same site:
    #
    #     [X, Z] = -2 i Y.
    #
    sigma_z_middle = operator_on_site(
        operator=SIGMA_Z,
        site=1,
        number_of_spins=number_of_spins,
    )

    sigma_y_middle = operator_on_site(
        operator=SIGMA_Y,
        site=1,
        number_of_spins=number_of_spins,
    )

    same_site_commutator = (
        sigma_x_middle @ sigma_z_middle
        - sigma_z_middle @ sigma_x_middle
    )

    np.testing.assert_allclose(
        same_site_commutator,
        -2.0j * sigma_y_middle,
        atol=1.0e-12,
    )


# =============================================================================
# TEST 2: TWO-SPIN ANALYTICAL BENCHMARK
# =============================================================================

def verify_two_spin_benchmark() -> None:
    """
    Verify that the general implementation reproduces the exact L = 2 result.
    """

    number_of_spins = 2
    J = 1.3
    h = 0.7

    hamiltonian = build_hamiltonian(
        number_of_spins=number_of_spins,
        J=J,
        h=h,
    )

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
# TEST 3: EXACTLY SOLVABLE LIMITS
# =============================================================================

def verify_known_limits() -> None:
    """
    Verify the h = 0 and J = 0 limits.
    """

    # -------------------------------------------------------------------------
    # h = 0
    #
    # For an open ferromagnetic chain, there are L - 1 aligned bonds:
    #
    #     E_0 = -(L - 1) J.
    #
    # The states with all spins up and all spins down are degenerate.
    # -------------------------------------------------------------------------

    number_of_spins = 4
    J = 1.2
    h = 0.0

    hamiltonian = build_hamiltonian(
        number_of_spins=number_of_spins,
        J=J,
        h=h,
    )

    eigenvalues = np.linalg.eigvalsh(hamiltonian)

    expected_ground_energy = -(number_of_spins - 1) * J

    np.testing.assert_allclose(
        eigenvalues[0:2],
        np.array(
            [
                expected_ground_energy,
                expected_ground_energy,
            ]
        ),
        atol=1.0e-12,
    )

    # -------------------------------------------------------------------------
    # J = 0
    #
    # Every spin aligns independently along +x:
    #
    #     E_0 = -L h.
    # -------------------------------------------------------------------------

    number_of_spins = 4
    J = 0.0
    h = 0.7

    hamiltonian = build_hamiltonian(
        number_of_spins=number_of_spins,
        J=J,
        h=h,
    )

    eigenvalues = np.linalg.eigvalsh(hamiltonian)

    expected_ground_energy = -number_of_spins * h

    np.testing.assert_allclose(
        eigenvalues[0],
        expected_ground_energy,
        atol=1.0e-12,
    )

    # -------------------------------------------------------------------------
    # L = 1
    #
    # There is no Ising bond, so H = -h X.
    # -------------------------------------------------------------------------

    number_of_spins = 1
    J = 5.0
    h = 0.8

    hamiltonian = build_hamiltonian(
        number_of_spins=number_of_spins,
        J=J,
        h=h,
    )

    eigenvalues = np.linalg.eigvalsh(hamiltonian)

    np.testing.assert_allclose(
        eigenvalues,
        np.array([-h, h]),
        atol=1.0e-12,
    )


# =============================================================================
# TEST 4: GENERAL MATRIX PROPERTIES
# =============================================================================

def verify_general_matrix_properties() -> None:
    """
    Check dimensions, Hermiticity, trace and eigenvector orthonormality.
    """

    number_of_spins = 3
    J = 1.0
    h = 0.5

    dimension = 2**number_of_spins

    hamiltonian = build_hamiltonian(
        number_of_spins=number_of_spins,
        J=J,
        h=h,
    )

    if hamiltonian.shape != (dimension, dimension):
        raise AssertionError(
            "The Hamiltonian has incorrect dimensions."
        )

    np.testing.assert_allclose(
        hamiltonian,
        hamiltonian.conj().T,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        np.trace(hamiltonian),
        0.0,
        atol=1.0e-12,
    )

    eigenvalues, eigenvectors = diagonalize_hamiltonian(
        hamiltonian
    )

    np.testing.assert_allclose(
        eigenvectors.conj().T @ eigenvectors,
        np.eye(dimension, dtype=np.complex128),
        atol=1.0e-12,
    )

    ground_state_energy = eigenvalues[0]
    ground_state = eigenvectors[:, 0]

    residual = np.linalg.norm(
        hamiltonian @ ground_state
        - ground_state_energy * ground_state
    )

    np.testing.assert_allclose(
        residual,
        0.0,
        atol=1.0e-12,
    )


def run_all_tests() -> None:
    """
    Run every analytical and numerical verification.
    """

    verify_local_operator_construction()
    print("Local-operator tests passed.")

    verify_two_spin_benchmark()
    print("Two-spin analytical benchmark passed.")

    verify_known_limits()
    print("Exactly solvable limits passed.")

    verify_general_matrix_properties()
    print("General Hamiltonian tests passed.")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main() -> None:
    np.set_printoptions(
        precision=8,
        suppress=True,
    )

    run_all_tests()

    # Example finite chain.
    number_of_spins = 3
    J = 1.0
    h = 0.5

    hamiltonian = build_hamiltonian(
        number_of_spins=number_of_spins,
        J=J,
        h=h,
    )

    eigenvalues, eigenvectors = diagonalize_hamiltonian(
        hamiltonian
    )

    ground_state_energy = eigenvalues[0]
    first_excited_energy = eigenvalues[1]
    excitation_gap = first_excited_energy - ground_state_energy

    ground_state = eigenvectors[:, 0]

    residual = np.linalg.norm(
        hamiltonian @ ground_state
        - ground_state_energy * ground_state
    )

    print("\nExample calculation")
    print("-------------------")
    print(f"Number of spins: {number_of_spins}")
    print(f"Hilbert-space dimension: {2**number_of_spins}")
    print(f"J = {J}")
    print(f"h = {h}")

    print("\nHamiltonian:")
    print(hamiltonian.real)

    print("\nEnergy spectrum:")
    print(eigenvalues)

    print("\nGround-state energy:")
    print(ground_state_energy)

    print("\nFirst excitation gap:")
    print(excitation_gap)

    print("\nGround-state amplitudes:")

    for index, amplitude in enumerate(ground_state):
        if abs(amplitude) > 1.0e-10:
            print(
                f"{basis_label(index, number_of_spins):20s}"
                f": {amplitude.real:+.10f}"
            )

    print("\nGround-state norm:")
    print(np.vdot(ground_state, ground_state).real)

    print("\nEigenvalue-equation residual:")
    print(residual)


if __name__ == "__main__":
    main()
