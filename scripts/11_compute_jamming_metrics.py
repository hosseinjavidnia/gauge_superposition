import argparse
import glob
import json
import os

import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.utils.extmath import randomized_svd


def participation_ratio(z):
    l1 = np.abs(z).sum(axis=1)
    l2 = np.sqrt((z * z).sum(axis=1) + 1e-12)
    return (l1 * l1) / (l2 * l2 + 1e-12)


def effective_rank(G):
    tr = float(np.trace(G))
    tr2 = float(np.sum(G * G))
    return (tr * tr) / (tr2 + 1e-12)


def make_mbdl(n_components, alpha, seed):
    # sklearn version-robust constructor
    kwargs = dict(
        n_components=n_components,
        alpha=alpha,
        max_iter=500,
        batch_size=256,
        random_state=seed,
        fit_algorithm="lars",
        transform_algorithm="lasso_lars",
    )
    while True:
        try:
            return MiniBatchDictionaryLearning(**kwargs)
        except TypeError as e:
            msg = str(e)
            removed = False
            for k in list(kwargs.keys()):
                if k in msg:
                    kwargs.pop(k)
                    removed = True
                    break
            if not removed:
                for k in ["fit_algorithm", "transform_algorithm"]:
                    if k in kwargs:
                        kwargs.pop(k)
                        removed = True
                        break
            if not removed:
                raise


def greedy_clique(adj, seeds=8):
    """
    adj: boolean adjacency matrix (n,n) with False diagonal
    Returns a maximal clique found by multi-start greedy heuristic.
    """
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    seed_nodes = np.argsort(-deg)[: min(seeds, n)]
    best = []

    for s in seed_nodes:
        clique = [int(s)]
        cand = set(np.where(adj[s])[0].tolist())
        while cand:
            # pick node with max degree within candidate set
            cand_list = np.array(list(cand), dtype=int)
            # score = edges into cand_list (restricted degree)
            scores = adj[np.ix_(cand_list, cand_list)].sum(axis=1)
            v = int(cand_list[np.argmax(scores)])
            clique.append(v)
            cand = cand.intersection(set(np.where(adj[v])[0].tolist()))
        if len(clique) > len(best):
            best = clique
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="runs/C_samples")
    ap.add_argument("--out_jsonl", default="runs/results/jamming_C.jsonl")
    ap.add_argument("--m", type=int, default=256, help="dictionary size (overcomplete)")
    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="sparsity strength for dictionary learning",
    )
    ap.add_argument(
        "--tau", type=float, default=1e-3, help="harm normalization stabilizer"
    )
    ap.add_argument(
        "--eta_p", type=float, default=1.0, help="eta(t)=t^p on |Gtilde_ij|"
    )
    ap.add_argument(
        "--max_clusters",
        type=int,
        default=40,
        help="analyze top-N clusters by sample count",
    )
    ap.add_argument(
        "--candidate_top",
        type=int,
        default=128,
        help="restrict clique search to top usage atoms",
    )
    ap.add_argument(
        "--tau_quantiles",
        nargs="+",
        type=float,
        default=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        help="quantiles of positive W_ij used as tau candidates (descending works well)",
    )
    ap.add_argument(
        "--seeds_per_tau",
        type=int,
        default=8,
        help="multi-start seeds for clique heuristic",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.in_dir, "cluster_*.npz")))
    if not files:
        raise RuntimeError(
            "No cluster_*.npz found. Run 10_collect_cluster_xgrad.py first."
        )

    # sort by n desc, keep top max_clusters
    sizes = []
    for f in files:
        d = np.load(f)
        sizes.append((f, d["X"].shape[0]))
    sizes.sort(key=lambda x: x[1], reverse=True)
    files = [f for f, _ in sizes[: args.max_clusters]]

    with open(args.out_jsonl, "w") as out:
        for path in files:
            cid = int(os.path.basename(path).split("_")[1].split(".")[0])
            dat = np.load(path)
            X = dat["X"].astype(np.float32)  # (n,d)
            GX = dat["GX"].astype(np.float32)  # (n,d)
            n, d = X.shape

            # center X for dictionary learning
            Xc = X - X.mean(axis=0, keepdims=True)

            # -------- local overcomplete frame (dictionary learning baseline) --------
            dl = make_mbdl(args.m, args.alpha, args.seed)
            Z = dl.fit_transform(Xc)  # (n,m)
            D = dl.components_.astype(np.float32)  # (m,d) atoms (rows)

            # normalize atoms (rows)
            dn = np.linalg.norm(D, axis=1, keepdims=True) + 1e-12
            Dn = D / dn

            # -------- active count --------
            k_pr = participation_ratio(Z)
            k_active = float(k_pr.mean())

            # -------- Fisher/GN proxy in code space --------
            # g_z ≈ D * g_x  =>  GZ = GX @ D^T
            GZ = (GX @ D.T).astype(np.float32)  # (n,m)
            G = (GZ.T @ GZ) / float(n)  # (m,m)
            G = 0.5 * (G + G.T)
            Reff = float(effective_rank(G))
            J = float(k_active / (Reff + 1e-12))

            # -------- harm matrix W from normalized Fisher interactions --------
            diag = np.diag(G).copy()
            inv_sqrt = 1.0 / np.sqrt(diag + args.tau)
            Gtilde = (inv_sqrt[:, None] * G) * inv_sqrt[None, :]
            W = (np.abs(Gtilde) ** args.eta_p).astype(np.float32)
            np.fill_diagonal(W, 0.0)

            # -------- choose semantic bandwidth subspace of dimension r = ceil(Reff) --------
            r = int(max(1, min(d, int(np.ceil(Reff)))))
            # compute r-dim PCA basis of cluster activations (on Xc)
            # U_r: (n,r), Vt: (r,d) => basis in ambient is B_r = Vt.T (d,r)
            _, _, Vt = randomized_svd(Xc, n_components=r, random_state=args.seed)
            B_r = Vt.T.astype(np.float32)  # (d,r), orthonormal columns (approx)

            # project atoms into r-dim coords: Acoords = Dn * B_r   (m,r)
            Acoords = (Dn @ B_r).astype(np.float32)
            an = np.linalg.norm(Acoords, axis=1, keepdims=True) + 1e-12
            Ahat = Acoords / an  # unit vectors in R^r
            K_r = (Ahat @ Ahat.T).astype(np.float32)
            np.fill_diagonal(K_r, 0.0)

            # full projected energy (optional diagnostic)
            E_full = float(np.sum(W * (K_r * K_r)))

            # -------- certified subset: find A such that all W_ij >= tau_star --------
            # candidate atoms: top usage
            usage = np.mean(np.abs(Z), axis=0)  # (m,)
            cand_sz = int(min(args.m, args.candidate_top))
            Cand = np.argsort(-usage)[:cand_sz]

            W0 = W[np.ix_(Cand, Cand)]
            np.fill_diagonal(W0, 0.0)

            off = W0[~np.eye(cand_sz, dtype=bool)]
            off_pos = off[off > 0]
            # If all weights are 0, certification is impossible
            if off_pos.size == 0:
                best = dict(bound=0.0, tau_star=0.0, k_cert=0, clique=None, E_cert=0.0)
            else:
                # tau candidates from quantiles of positive weights
                taus = []
                for q in args.tau_quantiles:
                    taus.append(float(np.quantile(off_pos, q)))
                # unique, sorted descending, drop nonpositive
                taus = sorted({t for t in taus if t > 0}, reverse=True)

                best = dict(bound=0.0, tau_star=0.0, k_cert=0, clique=None, E_cert=0.0)

                for tau_star in taus:
                    adj = W0 >= tau_star
                    np.fill_diagonal(adj, False)

                    clique_local = greedy_clique(adj, seeds=args.seeds_per_tau)
                    k_cert = len(clique_local)
                    if k_cert <= 1:
                        continue

                    # certify that all pairs in clique satisfy W>=tau_star
                    A_local = np.array(clique_local, dtype=int)
                    sub = W0[np.ix_(A_local, A_local)]
                    ok = np.all(sub[~np.eye(k_cert, dtype=bool)] >= tau_star - 1e-12)
                    if not ok:
                        continue

                    # map back to global atom indices
                    A = Cand[A_local]
                    # Welch-type bound becomes positive when k_cert > r
                    bound = float(
                        tau_star * max(0.0, (k_cert * k_cert) / (r + 1e-12) - k_cert)
                    )

                    # certified projected energy on this subset
                    W_A = W[np.ix_(A, A)]
                    K_A = K_r[np.ix_(A, A)]
                    np.fill_diagonal(W_A, 0.0)
                    np.fill_diagonal(K_A, 0.0)
                    E_cert = float(np.sum(W_A * (K_A * K_A)))

                    # choose the tau_star that maximizes the certified bound (tie-breaker: larger clique)
                    if (bound > best["bound"]) or (
                        np.isclose(bound, best["bound"]) and k_cert > best["k_cert"]
                    ):
                        best = dict(
                            bound=bound,
                            tau_star=float(tau_star),
                            k_cert=int(k_cert),
                            clique=A.astype(int),
                            E_cert=E_cert,
                        )

            # certified slack (should be >= 1 if bound>0 and numerics are OK)
            slack_cert = (
                (best["E_cert"] / (best["bound"] + 1e-12))
                if best["bound"] > 0
                else None
            )

            rec = {
                "cluster": cid,
                "n": int(n),
                "d": int(d),
                "m": int(args.m),
                "alpha": float(args.alpha),
                "k_active": float(k_active),
                "Reff": float(Reff),
                "J": float(J),
                # projected energy in r-dim bandwidth subspace
                "r_band": int(r),
                "E_full_proj": float(E_full),
                # certified subset + certified lower bound on subset energy
                "tau_star": float(best["tau_star"]),
                "k_cert": int(best["k_cert"]),
                "bound_raw": float(
                    best["bound"]
                ),  # keep same field name so script 12 works
                "E_cert": float(best["E_cert"]),
                "slack_Ecert_over_bound": (
                    float(slack_cert) if slack_cert is not None else None
                ),
                "clique_atoms": (
                    best["clique"].tolist() if best["clique"] is not None else None
                ),
            }
            out.write(json.dumps(rec) + "\n")

    print("Wrote", args.out_jsonl)


if __name__ == "__main__":
    main()
