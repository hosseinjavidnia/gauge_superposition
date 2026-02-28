import argparse
import json
import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", default="runs/acts")
    ap.add_argument("--out", default="runs/graph")
    ap.add_argument("--n_clusters", type=int, default=128)
    ap.add_argument("--knn", type=int, default=6)
    ap.add_argument("--max_fit", type=int, default=200000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta = json.load(open(os.path.join(args.acts_dir, "meta.json")))
    N = meta["n_tokens_written"]
    d = meta["hidden_size"]

    acts = np.memmap(
        os.path.join(args.acts_dir, "acts.f16"),
        mode="r",
        dtype=np.float16,
        shape=(N, d),
    ).astype(np.float32)
    X = acts[: min(N, args.max_fit)]

    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters, batch_size=4096, random_state=0
    )
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    assign = kmeans.predict(acts)

    # kNN graph on centroids
    nn = NearestNeighbors(n_neighbors=args.knn + 1, metric="euclidean")
    nn.fit(centroids)
    dists, idx = nn.kneighbors(centroids)

    edges = set()
    for u in range(args.n_clusters):
        for j in range(1, args.knn + 1):
            v = int(idx[u, j])
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))
    edges = sorted(list(edges))

    np.save(os.path.join(args.out, "centroids.npy"), centroids.astype(np.float32))
    np.save(os.path.join(args.out, "assign.npy"), assign.astype(np.int32))
    with open(os.path.join(args.out, "edges.json"), "w") as f:
        json.dump({"edges": edges, "n_clusters": args.n_clusters, "knn": args.knn}, f)

    print("clusters:", args.n_clusters, "edges:", len(edges))


if __name__ == "__main__":
    main()
