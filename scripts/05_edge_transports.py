import argparse
import json
import os

import numpy as np
from scipy.linalg import svd


def polar_orth(A: np.ndarray) -> np.ndarray:
    """Orthogonal polar factor of square A (k×k): A = U S V^T => polar(A)=U V^T."""
    U, _, Vt = svd(A, full_matrices=False)
    return (U @ Vt).astype(np.float32)


def polar_Q_and_smin(T: np.ndarray):
    """Return (Q, s_min) from SVD(T): T=U diag(s) V^T, Q=U V^T."""
    U, s, Vt = svd(T, full_matrices=False)
    Q = (U @ Vt).astype(np.float32)
    smin = float(np.min(s)) if s.size else 0.0
    return Q, smin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", default="runs/acts")
    ap.add_argument("--graph_dir", default="runs/graph")
    ap.add_argument("--bases_dir", default="runs/bases")
    ap.add_argument("--out", default="runs/edges")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lambda_ridge", type=float, default=1e-2)
    ap.add_argument("--max_overlap", type=int, default=8000)
    ap.add_argument("--min_overlap", type=int, default=256)
    ap.add_argument(
        "--smin_thresh",
        type=float,
        default=0.0,
        help="Skip edge if min singular value of ridge transport Tvu is below this threshold.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    meta = json.load(open(os.path.join(args.acts_dir, "meta.json")))
    N = meta["n_tokens_written"]
    d = meta["hidden_size"]

    acts = np.memmap(
        os.path.join(args.acts_dir, "acts.f16"),
        mode="r",
        dtype=np.float16,
        shape=(N, d),
    ).astype(np.float32)

    centroids = np.load(os.path.join(args.graph_dir, "centroids.npy")).astype(
        np.float32
    )
    edges = json.load(open(os.path.join(args.graph_dir, "edges.json")))["edges"]
    bases = np.load(os.path.join(args.bases_dir, "bases.npy")).astype(
        np.float32
    )  # (C,d,k)

    # nearest + 2nd nearest centroid indices
    X = acts
    x2 = (X**2).sum(axis=1, keepdims=True)
    c2 = (centroids**2).sum(axis=1)[None, :]
    dist2 = x2 + c2 - 2.0 * (X @ centroids.T)
    nn1 = dist2.argmin(axis=1)
    dist2[np.arange(N), nn1] = np.inf
    nn2 = dist2.argmin(axis=1)

    edge_data = {}
    skipped_low_overlap = 0
    skipped_smin = 0

    for a, b in edges:
        key = f"{a}-{b}"  # script03 produces edges with a<b

        mask = ((nn1 == a) & (nn2 == b)) | ((nn1 == b) & (nn2 == a))
        idx = np.where(mask)[0]

        if idx.size < max(args.min_overlap, args.k * 8):
            skipped_low_overlap += 1
            continue

        if idx.size > args.max_overlap:
            idx = rng.choice(idx, size=args.max_overlap, replace=False)

        Xo = X[idx]  # (n,d)

        Bu = bases[a]  # (d,k)  u=a
        Bv = bases[b]  # (d,k)  v=b

        Zu = (Xo @ Bu).T  # (k,n)
        Zv = (Xo @ Bv).T  # (k,n)

        # Ridge transport T_{v<-u}
        ZuZuT = Zu @ Zu.T
        Tvu = (Zv @ Zu.T) @ np.linalg.inv(
            ZuZuT + args.lambda_ridge * np.eye(args.k, dtype=np.float32)
        )

        # Geometric orthogonal factor + conditioning
        Qvu, smin = polar_Q_and_smin(Tvu)
        if smin < args.smin_thresh:
            skipped_smin += 1
            continue

        # Procrustes semantic proxy: Pvu = polar(Bv^T Bu)
        S = (Bv.T @ Bu).astype(np.float32)
        Pvu = polar_orth(S)

        # Group-valued edge defect: gvu = P^T Q
        gvu = (Pvu.T @ Qvu).astype(np.float32)

        edge_data[key] = {
            "a": int(a),
            "b": int(b),
            "gvu": gvu,
            "n_overlap": int(idx.size),
            "smin_Tvu": float(smin),
        }

    # Save edge operators
    np.savez(
        os.path.join(args.out, "edge_ops.npz"),
        **{k: v["gvu"] for k, v in edge_data.items()},
    )

    # Save metadata for debugging/analysis
    with open(os.path.join(args.out, "edge_index.json"), "w") as f:
        json.dump(
            {
                "edges": list(edge_data.keys()),
                "k": args.k,
                "lambda_ridge": args.lambda_ridge,
                "max_overlap": args.max_overlap,
                "min_overlap": args.min_overlap,
                "smin_thresh": args.smin_thresh,
                "saved_edges": len(edge_data),
                "skipped_low_overlap": skipped_low_overlap,
                "skipped_smin": skipped_smin,
            },
            f,
            indent=2,
        )

    # Also write per-edge stats (small JSONL)
    stats_path = os.path.join(args.out, "edge_stats.jsonl")
    with open(stats_path, "w") as f:
        for key, v in edge_data.items():
            f.write(
                json.dumps(
                    {
                        "edge": key,
                        **{k2: v[k2] for k2 in ["a", "b", "n_overlap", "smin_Tvu"]},
                    }
                )
                + "\n"
            )

    print("saved edge ops for", len(edge_data), "edges")
    print("skipped_low_overlap:", skipped_low_overlap, "skipped_smin:", skipped_smin)
    print("wrote", stats_path)


if __name__ == "__main__":
    main()
