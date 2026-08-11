import argparse

import matplotlib.pyplot as plt
import numpy as np

from tfim.hamiltonian import build_sparse_hamiltonian
from tfim.observables import correlation_profile
from tfim.solvers import ground_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ground-state zz correlation profile."
    )
    parser.add_argument("-L", type=int, default=10)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--save", default="correlations.png")
    args = parser.parse_args()

    H = build_sparse_hamiltonian(
        args.L,
        J=args.J,
        h=args.h,
        periodic=args.periodic,
    )
    energy0, state = ground_state(H)
    correlations = correlation_profile(state, args.L)
    separations = np.arange(args.L) - args.L // 2

    plt.figure()
    plt.plot(separations, correlations, "o-")
    plt.xlabel("site separation from origin")
    plt.ylabel(r"$\langle \sigma_i^z\sigma_j^z\rangle$")
    plt.tight_layout()
    plt.savefig(args.save, dpi=180)
    plt.close()

    print(f"E0 = {energy0:.12g}")
    print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
