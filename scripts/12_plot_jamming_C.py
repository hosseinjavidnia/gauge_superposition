import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="runs/results/jamming_C.jsonl")
    ap.add_argument("--outprefix", default="runs/results/jamming_C")
    args = ap.parse_args()

    rows = load_jsonl(args.infile)
    print("clusters:", len(rows))

    J = np.array([r["J"] for r in rows], float)
    Eproj = np.array([r["E_full_proj"] for r in rows], float)

    # certified subset data
    B = np.array([r["bound_raw"] for r in rows], float)
    Ecert = np.array([r["E_cert"] for r in rows], float)
    mask = B > 0

    print("J: min/median/mean/max:", J.min(), np.median(J), J.mean(), J.max())
    print(
        "E_full_proj: min/median/mean/max:",
        Eproj.min(),
        np.median(Eproj),
        Eproj.mean(),
        Eproj.max(),
    )

    if mask.any():
        slack = Ecert[mask] / (B[mask] + 1e-12)
        print(
            "Certified slack (E_cert / bound): min/median/mean/max:",
            slack.min(),
            np.median(slack),
            slack.mean(),
            slack.max(),
        )
        print("Certified edges/clusters with bound>0:", int(mask.sum()))
    else:
        print(
            "No certified bounds > 0. Try increasing --candidate_top or adjusting --tau_quantiles."
        )

    # Plot 1: E_cert vs bound (certified check)
    plt.figure()
    plt.scatter(B[mask], Ecert[mask], s=30, alpha=0.75)
    m = max(B[mask].max(), Ecert[mask].max()) if mask.any() else 1.0
    plt.plot([0, m], [0, m])
    plt.xlabel(
        r"$\widehat{\mathrm{LB}}=\tau_\star\left(\frac{|A|^2}{r}-|A|\right)_{+}$"
    )
    plt.ylabel(r"$\mathcal{E}_{A,r}=\sum_{i\neq j\in A} W_{ij}\,(K^{(r)}_{ij})^2$")
    plt.title("C certified check: $\\mathcal{E}_{A,r} \\geq \\widehat{\\mathrm{LB}}$")
    out1 = args.outprefix + "_Ecert_vs_bound.pdf"
    plt.savefig(out1, dpi=220, bbox_inches="tight")
    print("Wrote", out1)

    # Plot 2: Projected energy vs jamming index
    plt.figure()
    plt.scatter(J, Eproj, s=30, alpha=0.75)
    plt.xlabel(r"$J(c)=k_{\mathrm{active}}/R_{\mathrm{eff}}$")
    plt.ylabel(r"$\mathcal{E}^{(r)}(c)=\sum_{i\neq j} W_{ij}\,(K^{(r)}_{ij})^2$")
    plt.title("Projected Fisher-weighted energy vs jamming index")
    out2 = args.outprefix + "_Eproj_vs_J.pdf"
    plt.savefig(out2, dpi=220, bbox_inches="tight")
    print("Wrote", out2)

    # Plot 3 (optional): E_cert vs J for certified clusters only
    if mask.any():
        plt.figure()
        plt.scatter(J[mask], Ecert[mask], s=30, alpha=0.75)
        plt.xlabel(r"$J(c)$")
        plt.ylabel(r"$\mathcal{E}_{A,r}$ (certified subset)")
        plt.title("Certified subset energy vs jamming index")
        out3 = args.outprefix + "_Ecert_vs_J.pdf"
        plt.savefig(out3, dpi=220, bbox_inches="tight")
        print("Wrote", out3)


if __name__ == "__main__":
    main()
