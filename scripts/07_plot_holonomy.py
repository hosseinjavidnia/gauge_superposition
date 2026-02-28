import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def ecdf(x):
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/results/holonomy_base_holonomy_defects.npy")
    ap.add_argument(
        "--persist", default="runs/results/holonomy_persist_holonomy_defects.npy"
    )
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--outdir", default="runs/results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    base = np.load(args.base)
    pers = np.load(args.persist)

    norm = np.sqrt(2.0 * args.k)
    base_n = base / norm
    pers_n = pers / norm

    # --- Histogram (normalized D_hol) ---
    plt.figure()
    plt.hist(base_n, bins=25, alpha=0.6, label="baseline", density=True)
    plt.hist(pers_n, bins=25, alpha=0.6, label="persistent", density=True)
    plt.xlabel(r"$D_{\mathrm{hol}}=\|R-I\|_F/\sqrt{2k}$")
    plt.ylabel("density")
    plt.legend()
    out1 = os.path.join(args.outdir, "holonomy_compare_hist.pdf")
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    print("Wrote", out1)

    # --- ECDF ---
    xb, yb = ecdf(base_n)
    xp, yp = ecdf(pers_n)

    plt.figure()
    plt.plot(xb, yb, label="baseline")
    plt.plot(xp, yp, label="persistent")
    plt.xlabel(r"$D_{\mathrm{hol}}=\|R-I\|_F/\sqrt{2k}$")
    plt.ylabel("ECDF")
    plt.legend()
    out2 = os.path.join(args.outdir, "holonomy_compare_ecdf.pdf")
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    print("Wrote", out2)

    # --- Print quick stats ---
    def stats(name, x):
        qs = [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
        print(f"\n{name}: n={len(x)} mean={x.mean():.4f} max={x.max():.4f}")
        for q in qs:
            print(f"  q={q:>4}: {np.quantile(x,q):.4f}")

    stats("baseline D_hol", base_n)
    stats("persistent D_hol", pers_n)


if __name__ == "__main__":
    main()
