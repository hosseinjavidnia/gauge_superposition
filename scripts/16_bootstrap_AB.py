import argparse
import json
import os

import networkx as nx
import numpy as np
from scipy.linalg import svd


def polar_orth(A):
    U, _, Vt = svd(A, full_matrices=False)
    return (U @ Vt).astype(np.float32)


def ridge_transport(Zu, Zv, lam):
    # Zu, Zv: (k,n). Return Tvu (k,k)
    ZuZuT = Zu @ Zu.T
    return (Zv @ Zu.T) @ np.linalg.inv(
        ZuZuT + lam * np.eye(Zu.shape[0], dtype=np.float32)
    )


def D_shear_from_overlap(Xo, Bu, Bv, lam):
    Zu = (Xo @ Bu).T
    Zv = (Xo @ Bv).T
    Tvu = ridge_transport(Zu, Zv, lam)
    Q = polar_orth(Tvu)
    P = polar_orth(Bv.T @ Bu)
    A = Q - P
    return float(np.linalg.norm(A, "fro") / (2.0 * np.sqrt(Zu.shape[0]))), Q, P


def D_hol_from_cycle(cycle_nodes, edge_QP, k):
    # cycle_nodes: [c0,c1,...,cL=c0]
    # edge_QP gives Q,P for oriented edge v<-u
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
    ap.add_argument("--edges_dir", required=True)
    ap.add_argument("--out_json", default="runs/results/D_bootstrap_AB.json")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lambda_ridge", type=float, default=1e-2)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--max_overlap", type=int, default=8000)
    ap.add_argument(
        "--boot_n",
        type=int,
        default=2000,
        help="bootstrap sample size per edge (<= overlap size)",
    )
    ap.add_argument("--n_edges", type=int, default=50)
    ap.add_argument("--n_cycles", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # load acts + nn cache
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

    edges_all = json.load(open(os.path.join(args.graph_dir, "edges.json")))["edges"]
    bases = np.load(os.path.join(args.bases_dir, "bases.npy")).astype(np.float32)

    # edges we consider (must match edges_dir)
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
        raise RuntimeError("No components in edges_dir.")
    LCC = max(comps, key=len)
    G = G.subgraph(LCC).copy()

    # Choose edges for shearing bootstrap
    edges_list = list(G.edges())
    rng.shuffle(edges_list)
    edges_list = edges_list[: min(args.n_edges, len(edges_list))]

    # Choose cycles (fundamental cycles from spanning tree chords)
    T = nx.minimum_spanning_tree(G)
    chords = [e for e in G.edges() if not T.has_edge(*e)]
    rng.shuffle(chords)
    chords = chords[: min(args.n_cycles, len(chords))]

    cycles = []
    for u, v in chords:
        path = nx.shortest_path(T, u, v)
        cyc = path
        cycles.append((u, v, cyc))

    # helper: overlap indices for an undirected edge (a,b)
    def overlap_idx(a, b):
        mask = ((nn1 == a) & (nn2 == b)) | ((nn1 == b) & (nn2 == a))
        idx = np.where(mask)[0]
        if idx.size == 0:
            return idx
        if idx.size > args.max_overlap:
            idx = rng.choice(idx, size=args.max_overlap, replace=False)
        return idx

    # Precompute overlap pools for chosen edges and for cycle edges
    needed_edges = set()
    for a, b in edges_list:
        needed_edges.add((a, b))
    for u, v, cyc in cycles:
        # cycle includes path edges + chord edge (v,u) closing
        for i in range(len(cyc) - 2):
            needed_edges.add((cyc[i], cyc[i + 1]))
        needed_edges.add((v, u))  # chord direction

    pools = {}
    for a, b in needed_edges:
        aa, bb = (a, b) if a < b else (b, a)
        idx = overlap_idx(aa, bb)
        pools[(a, b)] = idx  # store by oriented query; we'll sample from same pool
        pools[(b, a)] = idx

    # Bootstrap
    Ds_edge = {f"{a}-{b}": [] for (a, b) in edges_list}
    Dh_cycle = {f"chord_{u}-{v}": [] for (u, v, _) in cycles}

    for rep in range(args.B):
        # compute Q,P for all needed oriented edges for this replicate
        edge_QP = {}
        for a, b in needed_edges:
            idx = pools[(a, b)]
            if idx.size < args.k * 8:
                continue
            nboot = min(args.boot_n, idx.size)
            samp = rng.choice(idx, size=nboot, replace=True)
            Xo = X[samp]  # (nboot,d)
            Bu = bases[a]
            Bv = bases[b]
            Dsh, Q, P = D_shear_from_overlap(Xo, Bu, Bv, args.lambda_ridge)
            edge_QP[(a, b)] = (Q, P)

        # record shearing on selected edges
        for a, b in edges_list:
            if (a, b) in edge_QP:
                Ds_edge[f"{a}-{b}"].append(
                    float(
                        np.linalg.norm(edge_QP[(a, b)][0] - edge_QP[(a, b)][1], "fro")
                        / (2.0 * np.sqrt(args.k))
                    )
                )

        # record holonomy on cycles
        for u, v, cyc_path in cycles:
            # build explicit cycle node list including chord v->u at end
            cycle_nodes = cyc_path + [u]
            ok = True
            for i in range(len(cycle_nodes) - 1):
                if (cycle_nodes[i], cycle_nodes[i + 1]) not in edge_QP:
                    ok = False
                    break
            if ok:
                Dh = D_hol_from_cycle(cycle_nodes, edge_QP, args.k)
                Dh_cycle[f"chord_{u}-{v}"].append(Dh)

    # Summaries
    def summarize(arr):
        arr = np.array(arr, float)
        if arr.size == 0:
            return None
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "q05": float(np.quantile(arr, 0.05)),
            "q50": float(np.quantile(arr, 0.50)),
            "q95": float(np.quantile(arr, 0.95)),
        }

    out = {
        "k": args.k,
        "B": args.B,
        "boot_n": args.boot_n,
        "n_edges_target": args.n_edges,
        "n_cycles_target": args.n_cycles,
        "LCC_size": int(G.number_of_nodes()),
        "edges_used": {k: summarize(v) for k, v in Ds_edge.items()},
        "cycles_used": {k: summarize(v) for k, v in Dh_cycle.items()},
        "global": {
            "Dshear_all": summarize([x for v in Ds_edge.values() for x in v]),
            "Dhol_all": summarize([x for v in Dh_cycle.values() for x in v]),
        },
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote", args.out_json)
    print("Global D_shear summary:", out["global"]["Dshear_all"])
    print("Global D_hol summary:", out["global"]["Dhol_all"])


if __name__ == "__main__":
    main()
