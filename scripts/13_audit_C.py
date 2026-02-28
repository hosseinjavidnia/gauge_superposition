import argparse
import json

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
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    rows = load_jsonl(args.infile)
    rows = [r for r in rows if r.get("bound_raw", 0.0) is not None]
    print("clusters:", len(rows))

    B = np.array([r["bound_raw"] for r in rows], float)
    mask = B > 0
    print("clusters with bound>0:", int(mask.sum()))

    if mask.any():
        slack = np.array([r["slack_Ecert_over_bound"] for r in rows], float)[mask]
        print("Certified slack stats (E_cert / bound):")
        print(
            "  min/median/mean/max:",
            float(slack.min()),
            float(np.median(slack)),
            float(slack.mean()),
            float(slack.max()),
        )
        # hard check
        bad = np.where(slack < 1.0 - args.tol)[0]
        print("  violations (slack < 1 - tol):", int(bad.size))
    else:
        print("No certified bounds > 0; nothing to audit.")

    # correlations (empirical)
    J = np.array([r["J"] for r in rows], float)
    Eproj = np.array([r["E_full_proj"] for r in rows], float)

    # simple Pearson (no scipy dependency)
    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    print("Corr(J, E_full_proj):", corr(J, Eproj))

    if mask.any():
        Ecert = np.array([r["E_cert"] for r in rows], float)[mask]
        Jm = J[mask]
        print("Corr(J, E_cert) on certified clusters:", corr(Jm, Ecert))


if __name__ == "__main__":
    main()
