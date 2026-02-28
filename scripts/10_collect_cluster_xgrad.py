import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def iter_jsonl(paths):
    for p in paths:
        with open(p, "r") as f:
            for line in f:
                yield json.loads(line)["text"]


@torch.no_grad()
def assign_clusters(X, centroids):
    # X: (B,d) torch fp16/fp32 on GPU
    # centroids: (C,d) torch fp32 on GPU
    # argmin ||x-c||^2 = argmin (||x||^2 + ||c||^2 - 2 x·c)
    x2 = (X * X).sum(dim=1, keepdim=True)  # (B,1)
    c2 = (centroids * centroids).sum(dim=1).view(1, -1)  # (1,C)
    dist2 = x2 + c2 - 2.0 * (X @ centroids.t())
    return torch.argmin(dist2, dim=1)  # (B,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--every", type=int, default=8)  # sample token positions
    ap.add_argument("--per_cluster", type=int, default=512)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--graph_dir", default="runs/graph")
    ap.add_argument("--out", default="runs/C_samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    centroids = np.load(os.path.join(args.graph_dir, "centroids.npy")).astype(
        np.float32
    )
    C, d = centroids.shape

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        dtype=torch.float16,
    )
    model.train(False)

    device = next(model.parameters()).device
    cent_t = torch.tensor(centroids, device=device, dtype=torch.float32)

    # buffers per cluster
    X_buf = [[] for _ in range(C)]
    G_buf = [[] for _ in range(C)]
    done = np.zeros(C, dtype=bool)

    def all_done():
        return bool(done.all())

    for text in tqdm(iter_jsonl(args.inputs), desc="collect x & grad"):
        if all_done():
            break
        if not text.strip():
            continue

        enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len)
        input_ids = enc["input_ids"].to(device)

        # forward with hidden states in graph
        input_ids = input_ids.detach()
        input_ids.requires_grad_(False)

        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        H = out.hidden_states[args.layer]  # (1,T,d)
        H.retain_grad()

        logits = out.logits  # (1,T,V)
        # next-token loss over all positions except last
        # CE on logits[:, :-1] vs input_ids[:, 1:]
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)  # (1,T-1,V)
        y = input_ids[:, 1:]  # (1,T-1)
        nll = -logp.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)  # (1,T-1)
        loss = nll.mean()
        model.zero_grad(set_to_none=True)
        loss.backward()

        # gradients wrt H: (1,T,d)
        gH = H.grad[0]  # (T,d)
        Xh = H.detach()[0]  # (T,d)

        # sample positions (avoid last because next token loss excludes last)
        T = Xh.shape[0]
        pos = list(range(0, max(0, T - 1), args.every))
        if not pos:
            continue

        Xs = Xh[pos]  # (B,d)
        Gs = gH[pos]  # (B,d)

        # cluster assignment
        cid = assign_clusters(Xs.float(), cent_t)  # (B,)
        cid = cid.detach().cpu().numpy()

        Xs = Xs.detach().cpu().numpy().astype(np.float16)
        Gs = Gs.detach().cpu().numpy().astype(np.float16)

        for i, c in enumerate(cid):
            if done[c]:
                continue
            X_buf[c].append(Xs[i])
            G_buf[c].append(Gs[i])
            if len(X_buf[c]) >= args.per_cluster:
                done[c] = True

    # write npz per cluster
    kept = 0
    for c in range(C):
        if len(X_buf[c]) == 0:
            continue
        Xc = np.stack(X_buf[c], axis=0)  # (n,d)
        Gc = np.stack(G_buf[c], axis=0)
        np.savez(os.path.join(args.out, f"cluster_{c:03d}.npz"), X=Xc, GX=Gc)
        kept += 1

    meta = {
        "model": args.model,
        "layer": args.layer,
        "max_len": args.max_len,
        "every": args.every,
        "per_cluster": args.per_cluster,
        "n_clusters_saved": kept,
        "d": int(d),
        "C": int(C),
        "inputs": args.inputs,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved clusters:", kept, "to", args.out)


if __name__ == "__main__":
    main()
