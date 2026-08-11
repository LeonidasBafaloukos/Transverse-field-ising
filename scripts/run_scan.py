from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from tfim.analysis import finite_size_scan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finite-size transverse-field Ising model scan."
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[4, 6, 8, 10])
    parser.add_argument("--hmin", type=float, default=0.2)
    parser.add_argument("--hmax", type=float, default=1.8)
    parser.add_argument("--points", type=int, default=65)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    if args.J == 0.0:
        raise ValueError("J must be nonzero for plotting h/J.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    h_values = np.linspace(args.hmin, args.hmax, args.points)
    data = finite_size_scan(
        args.sizes,
        h_values,
        J=args.J,
        periodic=args.periodic,
    )

    with (out / "scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "L",
                "h",
                "E0",
                "gap",
                "Mx",
                "Mz",
                "Mz2",
                "Binder",
                "entropy",
                "parity",
            ]
        )
        for rows in data.values():
            for row in rows:
                writer.writerow(
                    [
                        row.L,
                        row.h,
                        row.energy0,
                        row.gap,
                        row.mx,
                        row.mz,
                        row.mz2,
                        row.binder,
                        row.entropy,
                        row.parity,
                    ]
                )

    plots = [
        ("gap", "Energy gap", "gap.png"),
        ("mx", r"$\langle M_x\rangle$", "mx.png"),
        ("mz2", r"$\langle M_z^2\rangle$", "mz2.png"),
        ("binder", "Binder cumulant", "binder.png"),
        ("entropy", "Half-chain entanglement entropy", "entropy.png"),
    ]

    for attribute, ylabel, filename in plots:
        plt.figure()
        for L, rows in data.items():
            plt.plot(
                [row.h / args.J for row in rows],
                [getattr(row, attribute) for row in rows],
                label=f"L={L}",
            )
        plt.axvline(1.0, linestyle="--", linewidth=1.0)
        plt.xlabel("h/J")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / filename, dpi=180)
        plt.close()

    print(f"Wrote data and plots to {out.resolve()}")


if __name__ == "__main__":
    main()
