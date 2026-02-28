import argparse
import json
import os
from datetime import datetime

import networkx as nx
import numpy as np


def frob(A):
    return float(np.linalg.norm(A, "fro"))


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print("Wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", default="runs/graph")
    ap.add_argument("--edges_dir", default="runs/edges")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out_json", default="runs/results/holonomy_demo.json")
    ap.add_argument("--save_arrays", action="store_true")
    args = ap.parse_args()

    edges_path = os.path.join(args.edges_dir, "edge_ops.npz")
    if not os.path.exists(edges_path):
        write_json(
            args.out_json,
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "status": "no_edge_ops_file",
                "edges_dir": args.edges_dir,
                "k": args.k,
            },
        )
        return

    edge_ops = np.load(edges_path)
    if len(edge_ops.files) == 0:
        write_json(
            args.out_json,
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "status": "edge_ops_empty",
                "edges_dir": args.edges_dir,
                "k": args.k,
                "sqrt_2k": float(np.sqrt(2.0 * args.k)),
                "graph": {
                    "connected_components": 0,
                    "lcc_size": 0,
                    "lcc_edges_with_ops": 0,
                    "tree_edges": 0,
                    "chord_edges": 0,
                },
            },
        )
        return

    edges_all = json.load(open(os.path.join(args.graph_dir, "edges.json")))["edges"]

    # Build graph from edges with operators
    G = nx.Graph()
    for a, b in edges_all:
        key = f"{a}-{b}"
        if key in edge_ops.files:
            G.add_edge(a, b, key=key)

    if G.number_of_edges() == 0:
        write_json(
            args.out_json,
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "status": "no_edges_after_intersection",
                "edges_dir": args.edges_dir,
                "k": args.k,
                "sqrt_2k": float(np.sqrt(2.0 * args.k)),
                "graph": {
                    "connected_components": 0,
                    "lcc_size": 0,
                    "lcc_edges_with_ops": 0,
                    "tree_edges": 0,
                    "chord_edges": 0,
                },
            },
        )
        return

    comps = list(nx.connected_components(G))
    comps_sorted = sorted(comps, key=len, reverse=True)
    n_comps = len(comps_sorted)
    lcc_size = len(comps_sorted[0])
    print("Connected components:", n_comps)
    print("Largest component size:", lcc_size)

    # Restrict to largest connected component
    LCC = comps_sorted[0]
    G = G.subgraph(LCC).copy()

    # Spanning tree
    T = nx.minimum_spanning_tree(G)
    root = list(T.nodes())[0]

    # g(v,u) = g_{v<-u}; stored key "a-b" (a<b) equals g_{b<-a}
    def g(v, u):
        a, b = (u, v) if u < v else (v, u)
        key = f"{a}-{b}"
        g_ab = edge_ops[key]
        return g_ab if (u < v) else g_ab.T

    # Gauge construction on tree
    U = {root: np.eye(args.k, dtype=np.float32)}
    bfs = nx.bfs_tree(T, root)
    parent = {root: None}
    for v in bfs.nodes():
        for w in T.neighbors(v):
            if w not in parent:
                parent[w] = v

    for v in bfs.nodes():
        p = parent[v]
        if p is None:
            continue
        U[v] = U[p] @ g(p, v)

    # Tree residuals
    tree_res = []
    for u, v in T.edges():
        tree_res.append(frob(U[v] @ g(v, u) @ U[u].T - np.eye(args.k)))
    tree_res = np.array(tree_res, dtype=np.float64)
    print("Tree residuals: mean", tree_res.mean(), "max", tree_res.max())

    # Chords + holonomy
    chord_edges = [e for e in G.edges() if not T.has_edge(*e)]
    chord_res = []
    hol_def = []

    for u, v in chord_edges:
        chord_res.append(frob(U[v] @ g(v, u) @ U[u].T - np.eye(args.k)))

        path = nx.shortest_path(T, u, v)
        H = np.eye(args.k, dtype=np.float32)
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            H = g(b, a) @ H
        H = g(u, v) @ H
        hol_def.append(frob(H - np.eye(args.k)))

    chord_res = np.array(chord_res, dtype=np.float64)
    hol_def = np.array(hol_def, dtype=np.float64)

    norm = np.sqrt(2.0 * args.k)
    Dhol = hol_def / norm if hol_def.size else np.array([], dtype=np.float64)
    diff = (
        np.abs(chord_res - hol_def) if hol_def.size else np.array([], dtype=np.float64)
    )

    if chord_edges:
        print("Chord edges:", len(chord_edges))
        print("Chord residuals: mean", chord_res.mean(), "max", chord_res.max())
        print("Holonomy defects: mean", hol_def.mean(), "max", hol_def.max())
        print("Residual-holonomy |diff|: mean", diff.mean(), "max", diff.max())
        print("Normalized holonomy D_hol: mean", Dhol.mean(), "max", Dhol.max())
    else:
        print("No chord edges in LCC (LCC is a tree).")

    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "status": "ok",
        "edges_dir": args.edges_dir,
        "k": int(args.k),
        "sqrt_2k": float(norm),
        "graph": {
            "connected_components": int(n_comps),
            "lcc_size": int(lcc_size),
            "lcc_edges_with_ops": int(G.number_of_edges()),
            "tree_edges": int(T.number_of_edges()),
            "chord_edges": int(len(chord_edges)),
        },
        "tree_residual": {
            "mean": float(tree_res.mean()) if tree_res.size else None,
            "max": float(tree_res.max()) if tree_res.size else None,
        },
        "chord_residual": {
            "mean": float(chord_res.mean()) if chord_res.size else None,
            "max": float(chord_res.max()) if chord_res.size else None,
        },
        "holonomy_defect": {
            "mean": float(hol_def.mean()) if hol_def.size else None,
            "max": float(hol_def.max()) if hol_def.size else None,
            "quantiles": (
                {
                    str(q): float(np.quantile(hol_def, q))
                    for q in [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
                }
                if hol_def.size
                else {}
            ),
        },
        "D_hol": {
            "mean": float(Dhol.mean()) if Dhol.size else None,
            "max": float(Dhol.max()) if Dhol.size else None,
            "quantiles": (
                {
                    str(q): float(np.quantile(Dhol, q))
                    for q in [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
                }
                if Dhol.size
                else {}
            ),
        },
        "residual_minus_holonomy_abs": {
            "mean": float(diff.mean()) if diff.size else None,
            "max": float(diff.max()) if diff.size else None,
        },
    }
    write_json(args.out_json, out)

    if args.save_arrays and hol_def.size:
        base = os.path.dirname(args.out_json) or "."
        stem = os.path.splitext(os.path.basename(args.out_json))[0]
        np.save(os.path.join(base, f"{stem}_holonomy_defects.npy"), hol_def)
        np.save(os.path.join(base, f"{stem}_chord_residuals.npy"), chord_res)
        np.save(
            os.path.join(base, f"{stem}_residual_minus_hol.npy"), chord_res - hol_def
        )
        print("Saved arrays under", base, "with stem", stem)


if __name__ == "__main__":
    main()
