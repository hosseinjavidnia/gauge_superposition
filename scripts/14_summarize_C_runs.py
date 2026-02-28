import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def stats_slack(rows, tol=1e-6):
    B = np.array([r.get("bound_raw", 0.0) for r in rows], float)
    mask = B > 0
    cert_n = int(mask.sum())
    n = len(rows)
    cert_rate = cert_n / max(1, n)

    slack = np.array([r.get("slack_Ecert_over_bound", np.nan) for r in rows], float)
    slack = slack[mask]
    if slack.size:
        viol = int(np.sum(slack < 1.0 - tol))
        smin, smed, smean, smax = (
            float(slack.min()),
            float(np.median(slack)),
            float(slack.mean()),
            float(slack.max()),
        )
    else:
        viol = 0
        smin = smed = smean = smax = float("nan")
    return cert_n, cert_rate, viol, smin, smed, smean, smax


def summarize_file(path, tol=1e-6):
    rows = load_jsonl(path)
    # core arrays
    J = np.array([r["J"] for r in rows], float)
    Eproj = np.array([r["E_full_proj"] for r in rows], float)
    cJEproj = corr(J, Eproj)

    B = np.array([r.get("bound_raw", 0.0) for r in rows], float)
    mask = B > 0
    if mask.any():
        Ecert = np.array([r["E_cert"] for r in rows], float)[mask]
        cJEcert = corr(J[mask], Ecert)
    else:
        cJEcert = float("nan")

    cert_n, cert_rate, viol, smin, smed, smean, smax = stats_slack(rows, tol=tol)

    # get common config fields if present
    def pick(k, default=None):
        for r in rows:
            if k in r:
                return r[k]
        return default

    rec = {
        "file": os.path.basename(path),
        "path": path,
        "clusters": len(rows),
        "cert_n": cert_n,
        "cert_rate": cert_rate,
        "violations": viol,
        "slack_min": smin,
        "slack_median": smed,
        "slack_mean": smean,
        "slack_max": smax,
        "corr_J_Eproj": cJEproj,
        "corr_J_Ecert": cJEcert,
        # config (may be None)
        "m": pick("m"),
        "alpha": pick("alpha"),
        "seed": pick("seed"),
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        default="runs/results/jamming_C_*.jsonl",
        help="Glob pattern for C run jsonl files.",
    )
    ap.add_argument("--out_csv", default="runs/results/C_summary.csv")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise RuntimeError(f"No files matched: {args.glob}")

    rows = [summarize_file(p, tol=args.tol) for p in paths]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    cols = [
        "file",
        "clusters",
        "cert_n",
        "cert_rate",
        "violations",
        "slack_min",
        "slack_median",
        "slack_mean",
        "slack_max",
        "corr_J_Eproj",
        "corr_J_Ecert",
        "m",
        "alpha",
        "seed",
    ]

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    print("Wrote", args.out_csv)
    # Pretty print
    print(
        "\nfile | cert_n/clusters | cert_rate | viol | slack_med | corr(J,Eproj) | corr(J,Ecert) | m | alpha | seed"
    )
    for r in rows:
        print(
            f"{r['file']:>26} | "
            f"{r['cert_n']:>2}/{r['clusters']:<2} | "
            f"{r['cert_rate']:.3f} | "
            f"{r['violations']:>4} | "
            f"{r['slack_median'] if not np.isnan(r['slack_median']) else float('nan'):.3f} | "
            f"{r['corr_J_Eproj']:.3f} | "
            f"{r['corr_J_Ecert'] if not np.isnan(r['corr_J_Ecert']) else float('nan'):.3f} | "
            f"{str(r['m']):>3} | {str(r['alpha']):>4} | {str(r['seed']):>4}"
        )


if __name__ == "__main__":
    main()
