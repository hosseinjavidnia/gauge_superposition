import argparse
import json
import os

import networkx as nx
import numpy as np
from scipy.linalg import svd


def polar_orth(A):
    U, _, Vt = svd(A, full_matrices=False)
    return U @ Vt


def polar_Q(T):
    U, s, Vt = svd(T, full_matrices=False)
    return (U @ Vt), float(np.min(s)) if len(s) else 0.0


def load_memmap_acts(acts_dir):
    meta = json.load(open(os.path.join(acts_dir, "meta.json")))
    N = meta["n_tokens_written"]
    d = meta["hidden_size"]
    acts = np.memmap(
        os.path.join(acts_dir, "acts.f16"), mode="r", dtype=np.float16, shape=(N, d)
    ).astype(np.float32)
    return meta, acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", default="runs/acts")
    ap.add_argument("--graph_dir", default="runs/graph")
    ap.add_argument("--bases_dir", default="runs/bases")
    ap.add_argument(
        "--edges_dir", required=True, help="e.g. runs/edges_base or runs/edges_persist"
    )
    ap.add_argument("--out_jsonl", default="runs/results/shearing_bound.jsonl")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lambda_ridge", type=float, default=1e-2)
    ap.add_argument("--max_overlap", type=int, default=8000)
    ap.add_argument("--min_overlap", type=int, default=256)
    ap.add_argument("--lcc_only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # load activations
    _, X = load_memmap_acts(args.acts_dir)
    N, d = X.shape

    # load graph info
    centroids = np.load(os.path.join(args.graph_dir, "centroids.npy")).astype(
        np.float32
    )
    edges_all = json.load(open(os.path.join(args.graph_dir, "edges.json")))["edges"]

    # load bases
    bases = np.load(os.path.join(args.bases_dir, "bases.npy")).astype(
        np.float32
    )  # (C,d,k)

    # edge keys we want to evaluate (from edges_dir)
    edge_index_path = os.path.join(args.edges_dir, "edge_index.json")
    edge_keys = json.load(open(edge_index_path))["edges"]  # list like ["a-b", ...]
    keep_set = set(edge_keys)

    # Build a graph from kept edges to optionally take LCC
    G = nx.Graph()
    for a, b in edges_all:
        key = f"{a}-{b}"
        if key in keep_set:
            G.add_edge(a, b, key=key)

    if args.lcc_only:
        comps = list(nx.connected_components(G))
        if not comps:
            raise RuntimeError("No edges in edges_dir / no connected components.")
        LCC = max(comps, key=len)
        G = G.subgraph(LCC).copy()
        keep_set = set()
        for a, b in G.edges():
            keep_set.add(f"{min(a,b)}-{max(a,b)}")

    # nearest + 2nd nearest centroid indices (for overlap selection)
    x2 = (X**2).sum(axis=1, keepdims=True)
    c2 = (centroids**2).sum(axis=1)[None, :]
    dist2 = x2 + c2 - 2.0 * (X @ centroids.T)
    nn1 = dist2.argmin(axis=1)
    dist2[np.arange(N), nn1] = np.inf
    nn2 = dist2.argmin(axis=1)

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    n_written = 0

    with open(args.out_jsonl, "w") as f:
        for a, b in edges_all:
            key = f"{a}-{b}"
            if key not in keep_set:
                continue

            # overlap tokens
            mask = ((nn1 == a) & (nn2 == b)) | ((nn1 == b) & (nn2 == a))
            idx = np.where(mask)[0]
            if idx.size < max(args.min_overlap, args.k * 8):
                continue
            if idx.size > args.max_overlap:
                idx = rng.choice(idx, size=args.max_overlap, replace=False)

            Xo = X[idx]  # (n,d)
            n = Xo.shape[0]

            Bu = bases[a]  # (d,k)
            Bv = bases[b]

            Zu = (Xo @ Bu).T  # (k,n)
            Zv = (Xo @ Bv).T  # (k,n)

            # ridge transport
            ZuZuT = Zu @ Zu.T
            Tvu = (Zv @ Zu.T) @ np.linalg.inv(
                ZuZuT + args.lambda_ridge * np.eye(args.k, dtype=np.float32)
            )

            Qvu, smin = polar_Q(Tvu)  # Q in O(k)
            Pvu = polar_orth(Bv.T @ Bu).astype(np.float32)  # Procrustes proxy in O(k)

            A = (Qvu - Pvu).astype(np.float32)

            # shearing
            A_frob = float(np.linalg.norm(A, "fro"))
            D_shear = A_frob / (2.0 * np.sqrt(args.k))

            # covariance in u-coords: Σ = (1/n) Zu Zu^T
            Sigma = (Zu @ Zu.T) / float(n)
            # symmetrize for safety
            Sigma = 0.5 * (Sigma + Sigma.T)
            evals = np.linalg.eigvalsh(Sigma)
            lam_min = float(np.min(evals))
            lam_max = float(np.max(evals))

            # Δ = (1/n) ||A Zu||_F^2 = tr(A Σ A^T)
            AZ = A @ Zu
            Delta = float((AZ * AZ).sum() / float(n))

            # lower bound
            LB = lam_min * (A_frob**2)
            slack = Delta / (LB + 1e-12)

            rec = {
                "edge": key,
                "a": int(a),
                "b": int(b),
                "n_overlap": int(n),
                "smin_Tvu": float(smin),
                "A_frob": A_frob,
                "D_shear": float(D_shear),
                "lam_min_Sigma": lam_min,
                "lam_max_Sigma": lam_max,
                "Delta": Delta,
                "LB": float(LB),
                "slack": float(slack),
            }
            f.write(json.dumps(rec) + "\n")
            n_written += 1

    print("Wrote", args.out_jsonl, "rows:", n_written)


if __name__ == "__main__":
    main()
