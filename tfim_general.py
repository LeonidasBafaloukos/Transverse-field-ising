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
# GENERAL MANY-BODY OPERATORS
# =============================================================================

def operator_on_site(
    operator: ComplexMatrix,
    site: int,
    number_of_spins: int,
) -> ComplexMatrix:
    """
    Embed a single-spin operator into the Hilbert space of an L-spin chain.

    The returned operator acts with `operator` on the selected site and
    with the identity on every other site.

    Python site convention:
        site = 0, 1, ..., L - 1

    Physical interpretation:
        site = 0 corresponds to the leftmost spin in the tensor product.

    Example for L = 3 and site = 1:

        I ⊗ operator ⊗ I
    """

    if number_of_spins < 1:
        raise ValueError("number_of_spins must be at least 1.")

    if site < 0 or site >= number_of_spins:
        raise ValueError(
            f"site must satisfy 0 <= site < {number_of_spins}."
        )

    if operator.shape != (2, 2):
        raise ValueError("The local operator must have shape (2, 2).")

    # The 1 x 1 matrix [1] is the neutral starting object for the
    # iterative Kronecker-product construction.
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


def computational_basis_state(
    index: int,
    number_of_spins: int,
) -> ComplexVector:
    """
    Construct one computational-basis vector.

    Basis convention:
        binary digit 0 -> |up>
        binary digit 1 -> |down>

    For L = 3, the ordering begins as

        index 0 -> |up up up>
        index 1 -> |up up down>
        index 2 -> |up down up>
        index 3 -> |up down down>
        ...
    """

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

    bits = format(index, f"0{number_of_spins}b")

    spin_labels = [
        "up" if bit == "0" else "down"
        for bit in bits
    ]

    return "|" + " ".join(spin_labels) + ">"


# =============================================================================
# TESTS OF THE GENERAL OPERATOR CONSTRUCTION
# =============================================================================

def verify_local_operator_construction() -> None:
    """
    Test the operator construction for a three-spin chain.
    """

    number_of_spins = 3
    dimension = 2**number_of_spins

    # -------------------------------------------------------------------------
    # Construct sigma_x acting on the middle spin:
    #
    #     I ⊗ sigma_x ⊗ I
    # -------------------------------------------------------------------------

    sigma_x_middle = operator_on_site(
        operator=SIGMA_X,
        site=1,
        number_of_spins=number_of_spins,
    )

    # Test the matrix dimension.
    if sigma_x_middle.shape != (dimension, dimension):
        raise AssertionError(
            "The many-body operator has the wrong dimensions."
        )

    # Test Hermiticity.
    np.testing.assert_allclose(
        sigma_x_middle,
        sigma_x_middle.conj().T,
        atol=1.0e-12,
    )

    # Since sigma_x^2 = I for one spin, the embedded operator must satisfy
    #
    #     (sigma_x on site 1)^2 = I on the full Hilbert space.
    np.testing.assert_allclose(
        sigma_x_middle @ sigma_x_middle,
        np.eye(dimension, dtype=np.complex128),
        atol=1.0e-12,
    )

    # -------------------------------------------------------------------------
    # Test its action on a specific state.
    #
    # index 1 in the L = 3 basis is |up up down>
    # index 3 is                   |up down down>
    #
    # Flipping the middle spin should map index 1 to index 3.
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Operators acting on different sites must commute.
    #
    # [sigma_x on site 0, sigma_z on site 2] = 0
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Operators sigma_x and sigma_z acting on the same site do not commute.
    #
    # [sigma_x, sigma_z] = -2 i sigma_y
    # -------------------------------------------------------------------------

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

    print("All local-operator tests passed.")

    print("\nTested state transformation:")
    print(
        f"{basis_label(initial_index, number_of_spins)}"
        f" -> "
        f"{basis_label(expected_final_index, number_of_spins)}"
    )

    print("\nMany-body operator shape:")
    print(sigma_x_middle.shape)

    print("\nHilbert-space dimension:")
    print(dimension)


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main() -> None:
    verify_local_operator_construction()


if __name__ == "__main__":
    main()
