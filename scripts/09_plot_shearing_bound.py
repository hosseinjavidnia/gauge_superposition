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
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outprefix", default="runs/results/shearing")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument(
        "--min_lam",
        type=float,
        default=1e-6,
        help="Drop edges with tiny lambda_min(Sigma) (bound becomes trivial).",
    )
    args = ap.parse_args()

    rows = load_jsonl(args.infile)
    rows = [r for r in rows if r["lam_min_Sigma"] >= args.min_lam]
    if not rows:
        print("No rows after filtering. Lower --min_lam.")
        return

    Delta = np.array([r["Delta"] for r in rows], float)
    LB = np.array([r["LB"] for r in rows], float)
    slack = np.array([r["slack"] for r in rows], float)
    Dsh = np.array([r["D_shear"] for r in rows], float)

    print("n_edges_used:", len(rows))
    print(
        "slack: min/median/mean/max:",
        float(slack.min()),
        float(np.median(slack)),
        float(slack.mean()),
        float(slack.max()),
    )
    print(
        "D_shear: min/median/mean/max:",
        float(Dsh.min()),
        float(np.median(Dsh)),
        float(Dsh.mean()),
        float(Dsh.max()),
    )

    # ---------- Plot 1: Delta vs LB ----------
    plt.figure()
    plt.scatter(LB, Delta, s=14, alpha=0.65)
    m = max(LB.max(), Delta.max())
    plt.plot([0, m], [0, m])
    plt.xlabel(r"$\mathrm{LB}=\lambda_{\min}(\Sigma_u)\,\|A\|_F^2$")
    plt.ylabel(r"$\Delta=\mathbb{E}\,\|A z_u\|_2^2$")
    plt.title(r"Theorem B check: $\Delta \geq \mathrm{LB}$")
    out1 = args.outprefix + "_delta_vs_lb.pdf"
    plt.savefig(out1, dpi=220, bbox_inches="tight")
    print("Wrote", out1)

    # ---------- Plot 2: Delta vs D_shear ----------
    plt.figure()
    plt.scatter(Dsh, Delta, s=14, alpha=0.65)
    plt.xlabel(r"$D_{\mathrm{shear}}=\|Q-\widehat{P}\|_F/(2\sqrt{k})$")
    plt.ylabel(r"$\Delta=\mathbb{E}\,\|(Q-\widehat{P})z_u\|_2^2$")
    plt.title(r"Transfer mismatch grows with shearing")
    out2 = args.outprefix + "_delta_vs_dshear.pdf"
    plt.savefig(out2, dpi=220, bbox_inches="tight")
    print("Wrote", out2)

    # ---------- Plot 3: Slack histogram ----------
    plt.figure()
    plt.hist(slack, bins=30)
    plt.xlabel(r"$\mathrm{slack}=\Delta/(\lambda_{\min}(\Sigma_u)\|A\|_F^2)$")
    plt.ylabel(r"count")
    plt.title(r"Slack distribution (should be $\geq 1$ up to numerical issues)")
    out3 = args.outprefix + "_slack_hist.pdf"
    plt.savefig(out3, dpi=220, bbox_inches="tight")
    print("Wrote", out3)


if __name__ == "__main__":
    main()
