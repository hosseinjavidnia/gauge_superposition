import argparse
import json
import os

import numpy as np
from sklearn.utils.extmath import randomized_svd


def complete_orthonormal(B, k, rng):
    """
    B: (d, k_eff) with orthonormal columns (approximately)
    Returns: (d, k) orthonormal basis, completing with random directions if needed.
    """
    d, k_eff = B.shape
    if k_eff >= k:
        Q, _ = np.linalg.qr(B)
        return Q[:, :k].astype(np.float32)

    # Add random vectors and orthogonalize them against B
    R = rng.standard_normal(size=(d, k - k_eff)).astype(np.float32)
    if k_eff > 0:
        R = R - B @ (B.T @ R)

    # QR on concatenation gives full orthonormal basis
    Q, _ = np.linalg.qr(np.concatenate([B, R], axis=1))
    return Q[:, :k].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", default="runs/acts")
    ap.add_argument("--graph_dir", default="runs/graph")
    ap.add_argument("--out", default="runs/bases")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--max_per_cluster", type=int, default=4000)
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

    assign = np.load(os.path.join(args.graph_dir, "assign.npy"))
    edges_meta = json.load(open(os.path.join(args.graph_dir, "edges.json")))
    n_clusters = int(edges_meta["n_clusters"])

    bases = np.zeros((n_clusters, d, args.k), dtype=np.float32)

    for c in range(n_clusters):
        idx = np.where(assign == c)[0]
        if idx.size == 0:
            # completely empty cluster: use random orthonormal basis
            bases[c] = complete_orthonormal(
                np.zeros((d, 0), dtype=np.float32), args.k, rng
            )
            continue

        if idx.size > args.max_per_cluster:
            idx = rng.choice(idx, size=args.max_per_cluster, replace=False)

        X = acts[idx]
        X = X - X.mean(axis=0, keepdims=True)

        # effective k is limited by samples
        # (need n_components < min(n_samples, n_features))
        k_eff = min(args.k, X.shape[0] - 1, d)
        if k_eff <= 0:
            bases[c] = complete_orthonormal(
                np.zeros((d, 0), dtype=np.float32), args.k, rng
            )
            continue

        # PCA directions via SVD: Vt is (k_eff, d)
        _, _, Vt = randomized_svd(X, n_components=k_eff, random_state=args.seed)
        B = Vt.T.astype(np.float32)  # (d, k_eff)

        # Ensure a full (d, k) orthonormal basis
        bases[c] = complete_orthonormal(B, args.k, rng)

    np.save(os.path.join(args.out, "bases.npy"), bases)
    print("saved bases:", bases.shape)


if __name__ == "__main__":
    main()
