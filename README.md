# Transverse-Field Ising Model

A numerical and analytical study of the two-spin transverse-field Ising model using exact diagonalization in Python.

## Model

For two spins, the Hamiltonian is

$$
H =
-J\sigma_1^z\sigma_2^z
-h\left(\sigma_1^x+\sigma_2^x\right).
$$

In the basis

$$
\{
|\uparrow\uparrow\rangle,
|\uparrow\downarrow\rangle,
|\downarrow\uparrow\rangle,
|\downarrow\downarrow\rangle
\},
$$

its matrix representation is

$$
H =
\begin{pmatrix}
-J & -h & -h & 0 \\
-h & J & 0 & -h \\
-h & 0 & J & -h \\
0 & -h & -h & -J
\end{pmatrix}.
$$

The analytical eigenvalues are

$$
E =
\left\{
-\sqrt{J^2+4h^2},
-J,
J,
\sqrt{J^2+4h^2}
\right\}.
$$

## Current implementation

The script `tfim_l2.py`:

- constructs the Hamiltonian using tensor products,
- verifies that the Hamiltonian is Hermitian,
- compares the numerical matrix with the analytical result,
- compares numerical and analytical eigenvalues,
- computes the ground-state energy and eigenvector,
- checks normalization and the eigenvalue-equation residual,
- calculates the transverse and longitudinal magnetizations,
- calculates the spin correlation $\langle\sigma_1^z\sigma_2^z\rangle$.

## Validation

The implementation has been checked in the following limits:

- $h=0$, where the ferromagnetic ground state is doubly degenerate,
- $J=0$, where the ground state is fully polarized along the $x$ direction,
- $J=1$ and $h=0.5$, where the numerical results agree with the analytical spectrum.

## Requirements

- Python 3
- NumPy

## Running the script

Activate the Python virtual environment containing NumPy and run

```bash
python tfim_l2.py
```

## Project status

This repository currently contains the analytically validated two-spin benchmark. The implementation will later be generalized to finite chains with arbitrary numbers of spins.

## Author

**Leonidas Bafaloukos**

Physics graduate from the National and Kapodistrian University of Athens, interested in theoretical and computational condensed matter physics, quantum many-body systems, superconductivity and quantum materials.
