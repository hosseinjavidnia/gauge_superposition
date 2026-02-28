import argparse
import json
import os

import numpy as np
from scipy.linalg import svd


def polar_orth(A):
    U, _, Vt = svd(A, full_matrices=False)
    return U @ Vt


def ridge_transport(Zu, Zv, lam):
    ZuZuT = Zu @ Zu.T
    return (Zv @ Zu.T) @ np.linalg.inv(
        ZuZuT + lam * np.eye(Zu.shape[0], dtype=np.float32)
    )


def Dshear(Xo, Bu, Bv, lam, k):
    Zu = (Xo @ Bu).T
    Zv = (Xo @ Bv).T
    T = ridge_transport(Zu, Zv, lam)
    Q = polar_orth(T)
    P = polar_orth(Bv.T @ Bu)
    return float(np.linalg.norm(Q - P, "fro") / (2 * np.sqrt(k)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", default="runs/acts")
    ap.add_argument("--graph_dir", default="runs/graph")
    ap.add_argument("--bases_dir", default="runs/bases")
    ap.add_argument("--edge", required=True, help="format a-b with a<b")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lambda_ridge", type=float, default=1e-2)
    ap.add_argument(
        "--sizes", nargs="+", type=int, default=[256, 512, 1024, 2000, 4000]
    )
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--max_overlap", type=int, default=8000)
    ap.add_argument("--out_json", default="runs/results/D_curve_shearing.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    meta = json.load(open(os.path.join(args.acts_dir, "meta.json")))
    N = meta["n_tokens_written"]
    d = meta["hidden_size"]
    X = np.memmap(
        os.path.join(args.acts_dir, "acts.f16"),
        mode="r",
        dtype=np.float16,
        shape=(N, d),
    ).astype(np.float32)

    nn1 = np.load(os.path.join(args.graph_dir, "nn1.npy"))
    nn2 = np.load(os.path.join(args.graph_dir, "nn2.npy"))
    bases = np.load(os.path.join(args.bases_dir, "bases.npy")).astype(np.float32)

    a, b = map(int, args.edge.split("-"))
    mask = ((nn1 == a) & (nn2 == b)) | ((nn1 == b) & (nn2 == a))
    idx = np.where(mask)[0]
    if idx.size > args.max_overlap:
        idx = rng.choice(idx, size=args.max_overlap, replace=False)
    if idx.size < args.k * 8:
        raise RuntimeError("Not enough overlap points for this edge.")

    Bu = bases[a]
    Bv = bases[b]
    out = {"edge": args.edge, "k": args.k, "overlap_n": int(idx.size), "sizes": {}}

    for n in args.sizes:
        vals = []
        n_use = min(n, idx.size)
        for _ in range(args.reps):
            samp = rng.choice(idx, size=n_use, replace=True)
            vals.append(Dshear(X[samp], Bu, Bv, args.lambda_ridge, args.k))
        vals = np.array(vals, float)
        out["sizes"][str(n)] = {
            "n": int(n_use),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "q05": float(np.quantile(vals, 0.05)),
            "q50": float(np.quantile(vals, 0.50)),
            "q95": float(np.quantile(vals, 0.95)),
        }
        print("n", n_use, "mean", vals.mean(), "std", vals.std(ddof=1))

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)
    print("Wrote", args.out_json)


if __name__ == "__main__":
    main()
