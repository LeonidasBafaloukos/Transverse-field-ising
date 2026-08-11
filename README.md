# Transverse-Field Ising Model

A computational study of the one-dimensional transverse-field Ising model (TFIM), developed as a compact quantum many-body physics project in Python.

## Model

The Hamiltonian is

$$
H=-J\sum_i \sigma_i^z\sigma_{i+1}^z-h\sum_i\sigma_i^x.
$$

The code supports open and periodic chains and is designed to investigate the finite-size approach to the quantum critical point at $h/J=1$.

## Implemented features

- validated dense exact diagonalization;
- sparse Hamiltonian construction directly in the computational basis;
- Lanczos/ARPACK low-energy eigensolver through `scipy.sparse.linalg.eigsh`;
- ground-state energy and excitation gap;
- transverse magnetization $\langle M_x\rangle$;
- longitudinal moments $\langle M_z\rangle$, $\langle M_z^2\rangle$, $\langle M_z^4\rangle$;
- Binder cumulant;
- real-space $zz$ correlation functions;
- half-chain entanglement entropy;
- global $\mathbb Z_2$ parity;
- automated field scans for several system sizes;
- CSV export and diagnostic plots;
- thermodynamic-limit Jordan-Wigner dispersion and exact bulk gap;
- regression tests against the analytical two-spin spectrum and exactly solvable limits.

The original `tfim_l2.py` and `tfim_general.py` are retained as learning and benchmark scripts.

## Installation

```bash
git clone https://github.com/LeonidasBafaloukos/Transverse-field-ising.git
cd Transverse-field-ising
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows PowerShell use `.venv\\Scripts\\Activate.ps1` instead of the `source` command.

## Finite-size field scan

```bash
python scripts/run_scan.py --sizes 4 6 8 10 --hmin 0.2 --hmax 1.8 --points 65
```

The script writes `results/scan.csv` and separate figures for

- the many-body gap,
- transverse magnetization,
- longitudinal order-parameter fluctuations,
- Binder cumulant,
- half-chain entanglement entropy.

The dashed line at $h/J=1$ marks the thermodynamic critical point.

## Correlation profile

```bash
python scripts/correlations.py -L 10 --h 1.0
```

This computes the ground state and plots

$$
\langle \sigma_i^z\sigma_j^z\rangle
$$

relative to a central reference site.

## Run the tests

```bash
pytest -q
```

## Repository structure

```text
src/tfim/
    hamiltonian.py   dense and sparse Hamiltonians
    solvers.py       exact and Lanczos eigensolvers
    observables.py   magnetization, correlations, entropy, parity
    analysis.py      field scans and finite-size analysis
    exact.py         thermodynamic Jordan-Wigner reference results
scripts/
    run_scan.py      main finite-size workflow
    correlations.py  real-space correlation diagnostic
tests/
    test_tfim.py     analytical and numerical regression tests
```

## Numerical scope

The Hilbert-space dimension grows as $2^L$. Sparse matrices and Lanczos allow substantially larger chains than the original dense implementation, but this remains an exact-diagonalization project rather than a tensor-network solver. On a normal laptop, roughly $L\sim14$--$18$ is a practical range depending on the scan size and available memory.

## Physical interpretation

For finite $L$ there is no sharp phase transition. The program instead tracks finite-size signatures that converge toward the thermodynamic transition:

- suppression of ferromagnetic correlations as $h/J$ increases;
- growth of transverse polarization;
- reduction of the low-energy gap near the critical region;
- characteristic finite-size behavior of the Binder cumulant;
- enhanced bipartite entanglement near criticality.

In the thermodynamic limit, the Jordan-Wigner and Bogoliubov solution gives

$$
\varepsilon(k)=2\sqrt{J^2+h^2-2Jh\cos k},
$$

so the bulk quasiparticle gap is

$$
\Delta_\infty=2|J-h|.
$$

## Project status

**Core project complete.** The repository now contains a validated finite-size exact-diagonalization workflow, sparse numerical methods, observables, critical-point diagnostics, automated scans, plotting tools and tests.

Natural future extensions include explicit symmetry-sector diagonalization, the full Jordan-Wigner/Bogoliubov derivation in code, real-time dynamics and quenches, and tensor-network methods.

## Author

**Leonidas Bafaloukos**

Physics graduate from the National and Kapodistrian University of Athens, interested in theoretical and computational condensed matter physics, quantum many-body systems, magnetism, superconductivity and quantum materials.
