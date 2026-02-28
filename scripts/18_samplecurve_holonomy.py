import argparse
import json
import os

import networkx as nx
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


def edge_QP_from_samples(Xo, Bu, Bv, lam):
    Zu = (Xo @ Bu).T
    Zv = (Xo @ Bv).T
    T = ridge_transport(Zu, Zv, lam)
    Q = polar_orth(T).astype(np.float32)
    P = polar_orth(Bv.T @ Bu).astype(np.float32)
    return Q, P


def D_hol_from_cycle(cycle_nodes, edge_QP, k):
    H = np.eye(k, dtype=np.float32)
    for i in range(len(cycle_nodes) - 1):
        u = cycle_nodes[i]
        v = cycle_nodes[i + 1]
        Qvu, Pvu = edge_QP[(u, v)]
        g = (Pvu.T @ Qvu).astype(np.float32)
        H = g @ H
    return float(np.linalg.norm(H - np.eye(k), "fro") / np.sqrt(2.0 * k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", default="runs/acts")
    ap.add_argument("--graph_dir", default="runs/graph")
    ap.add_argument("--bases_dir", default="runs/bases")
    ap.add_argument(
        "--edges_dir", required=True, help="runs/edges_base or runs/edges_persist"
    )
    ap.add_argument(
        "--chord",
        required=True,
        help="chord edge a-b (a<b) that is present in the LCC chords",
    )
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lambda_ridge", type=float, default=1e-2)
    ap.add_argument(
        "--sizes", nargs="+", type=int, default=[256, 512, 1024, 2000, 4000]
    )
    ap.add_argument("--reps", type=int, default=80)
    ap.add_argument("--max_overlap", type=int, default=8000)
    ap.add_argument("--out_json", default="runs/results/D_curve_holonomy.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # load acts
    meta = json.load(open(os.path.join(args.acts_dir, "meta.json")))
    N = meta["n_tokens_written"]
    d = meta["hidden_size"]
    X = np.memmap(
        os.path.join(args.acts_dir, "acts.f16"),
        mode="r",
        dtype=np.float16,
        shape=(N, d),
    ).astype(np.float32)

    # nn cache
    nn1 = np.load(os.path.join(args.graph_dir, "nn1.npy"))
    nn2 = np.load(os.path.join(args.graph_dir, "nn2.npy"))

    # bases
    bases = np.load(os.path.join(args.bases_dir, "bases.npy")).astype(np.float32)

    # build graph from edges_dir and take LCC
    edges_all = json.load(open(os.path.join(args.graph_dir, "edges.json")))["edges"]
    keep = set(
        json.load(open(os.path.join(args.edges_dir, "edge_index.json")))["edges"]
    )
    G = nx.Graph()
    for a, b in edges_all:
        key = f"{a}-{b}"
        if key in keep:
            G.add_edge(a, b)

    comps = list(nx.connected_components(G))
    if not comps:
        raise RuntimeError("No edges in edges_dir.")
    LCC = max(comps, key=len)
    G = G.subgraph(LCC).copy()

    # spanning tree + chord list
    T = nx.minimum_spanning_tree(G)
    chords = [e for e in G.edges() if not T.has_edge(*e)]

    a, b = map(int, args.chord.split("-"))
    if not ((a, b) in chords or (b, a) in chords):
        raise RuntimeError(
            "Provided chord is not a chord in the LCC. Pick one from chords list."
        )

    # tree path u->...->v
    path = nx.shortest_path(T, a, b)  # includes endpoints
    # cycle nodes: path then return to a via chord (b->a)
    cycle_nodes = path + [a]

    # overlap pool for any undirected edge {u,v}
    def overlap_idx(u, v):
        uu, vv = (u, v) if u < v else (v, u)
        mask = ((nn1 == uu) & (nn2 == vv)) | ((nn1 == vv) & (nn2 == uu))
        idx = np.where(mask)[0]
        if idx.size > args.max_overlap:
            idx = rng.choice(idx, size=args.max_overlap, replace=False)
        return idx

    # precompute pools for every edge in this cycle (both orientations share the same pool)
    pool = {}
    edges_in_cycle = []
    for i in range(len(cycle_nodes) - 1):
        u = cycle_nodes[i]
        v = cycle_nodes[i + 1]
        edges_in_cycle.append((u, v))
        idx = overlap_idx(u, v)
        if idx.size < args.k * 8:
            raise RuntimeError(
                f"Not enough overlap points for edge {u}-{v}: have {idx.size}"
            )
        pool[(u, v)] = idx
        pool[(v, u)] = idx

    out = {
        "edges_dir": args.edges_dir,
        "chord": args.chord,
        "cycle_nodes": cycle_nodes,
        "k": args.k,
        "sizes": {},
    }

    for n0 in args.sizes:
        vals = []
        n_use_per_edge = {}
        for _ in range(args.reps):
            edge_QP = {}
            for u, v in edges_in_cycle:
                idx = pool[(u, v)]
                n_use = min(n0, idx.size)
                samp = rng.choice(idx, size=n_use, replace=True)
                Xo = X[samp]
                Q, P = edge_QP_from_samples(Xo, bases[u], bases[v], args.lambda_ridge)
                edge_QP[(u, v)] = (Q, P)
                n_use_per_edge[f"{u}-{v}"] = int(n_use)
            vals.append(D_hol_from_cycle(cycle_nodes, edge_QP, args.k))

        vals = np.array(vals, float)
        out["sizes"][str(n0)] = {
            "n_target": int(n0),
            "n_use_per_edge": n_use_per_edge,
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "q05": float(np.quantile(vals, 0.05)),
            "q50": float(np.quantile(vals, 0.50)),
            "q95": float(np.quantile(vals, 0.95)),
        }
        print("n", n0, "mean", vals.mean(), "std", vals.std(ddof=1))

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote", args.out_json)


if __name__ == "__main__":
    main()
