import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", default="runs/acts")
    ap.add_argument("--graph_dir", default="runs/graph")
    args = ap.parse_args()

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

    x2 = (acts**2).sum(axis=1, keepdims=True)
    c2 = (centroids**2).sum(axis=1)[None, :]
    dist2 = x2 + c2 - 2.0 * (acts @ centroids.T)

    nn1 = dist2.argmin(axis=1).astype(np.int32)
    dist2[np.arange(N), nn1] = np.inf
    nn2 = dist2.argmin(axis=1).astype(np.int32)

    np.save(os.path.join(args.graph_dir, "nn1.npy"), nn1)
    np.save(os.path.join(args.graph_dir, "nn2.npy"), nn2)
    print(
        "Wrote",
        os.path.join(args.graph_dir, "nn1.npy"),
        os.path.join(args.graph_dir, "nn2.npy"),
    )


if __name__ == "__main__":
    main()
