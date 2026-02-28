import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def iter_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)["text"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--layer", type=int, default=16)  # pick a mid layer
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--every", type=int, default=8)  # subsample tokens
    ap.add_argument("--out", default="runs/acts")
    ap.add_argument("--inputs", nargs="+", required=True)  # jsonl files
    ap.add_argument("--label", nargs="+", required=True)  # same length, stratum labels
    ap.add_argument("--n_tokens", type=int, default=200000)  # total tokens across all
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    d = model.config.hidden_size
    N = args.n_tokens
    acts = np.memmap(
        os.path.join(args.out, "acts.f16"), mode="w+", dtype=np.float16, shape=(N, d)
    )
    labels = np.memmap(
        os.path.join(args.out, "labels.i32"), mode="w+", dtype=np.int32, shape=(N,)
    )
    srcs = np.memmap(
        os.path.join(args.out, "src.i32"), mode="w+", dtype=np.int32, shape=(N,)
    )

    label_map = {name: i for i, name in enumerate(sorted(set(args.label)))}
    src_idx = 0
    n_written = 0

    for path, lab in zip(args.inputs, args.label):
        lab_id = label_map[lab]
        for text in tqdm(iter_jsonl(path), desc=f"extract {lab}"):
            if n_written >= N:
                break
            if not text.strip():
                continue
            enc = tok(
                text, return_tensors="pt", truncation=True, max_length=args.max_len
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}

            out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states[args.layer]  # (1, T, d)
            hs = hs[0]  # (T, d)

            # subsample tokens
            for t in range(0, hs.shape[0], args.every):
                if n_written >= N:
                    break
                acts[n_written] = hs[t].detach().to("cpu").numpy().astype(np.float16)
                labels[n_written] = lab_id
                srcs[n_written] = src_idx
                n_written += 1

        src_idx += 1
        if n_written >= N:
            break

    meta = {
        "model": args.model,
        "layer": args.layer,
        "hidden_size": d,
        "n_tokens_written": int(n_written),
        "label_map": label_map,
        "inputs": args.inputs,
        "labels": args.label,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Done. wrote tokens:", n_written)


if __name__ == "__main__":
    main()
