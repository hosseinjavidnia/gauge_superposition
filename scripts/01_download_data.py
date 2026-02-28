import argparse
import json
import os

from datasets import load_dataset


def load_noverify(*args, **kwargs):
    """
    Hugging Face datasets verification bypass:
    - Newer: verification_mode="no_checks"
    - Older: ignore_verifications=True (deprecated but still present in many installs)
    """
    try:
        return load_dataset(*args, verification_mode="no_checks", **kwargs)
    except TypeError:
        return load_dataset(*args, ignore_verifications=True, **kwargs)


def write_jsonl(ds, path, text_key="text", n=None):
    with open(path, "w") as f:
        for i, ex in enumerate(ds):
            if n is not None and i >= n:
                break
            f.write(json.dumps({"text": ex.get(text_key, "")}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/datasets")
    ap.add_argument("--n_wiki", type=int, default=20000)
    ap.add_argument("--n_c4", type=int, default=20000)
    ap.add_argument("--n_code", type=int, default=20000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) WikiText-103
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    wiki = wiki.select(range(min(args.n_wiki, len(wiki))))
    wiki_path = os.path.join(args.out, "wikitext103.jsonl")
    write_jsonl(wiki, wiki_path, text_key="text")
    print("Wrote", wiki_path)

    # 2) "Web" stratum: try PrimeIntellect/c4-tiny, else stream allenai/c4
    c4_path = os.path.join(args.out, "c4tiny.jsonl")
    try:
        c4 = load_noverify("PrimeIntellect/c4-tiny", "en", split="train")
        # if it loads, write first n
        c4 = c4.select(range(min(args.n_c4, len(c4))))
        write_jsonl(c4, c4_path, text_key="text")
        print("Wrote", c4_path, "(from PrimeIntellect/c4-tiny, verifications off)")
    except Exception as e:
        print("PrimeIntellect/c4-tiny failed, falling back to streaming allenai/c4.")
        print("Reason:", repr(e))
        c4_stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
        write_jsonl(c4_stream, c4_path, text_key="text", n=args.n_c4)
        print("Wrote", c4_path, "(from allenai/c4 streaming)")

    # 3) Code stratum: stream the-stack-smol
    code_path = os.path.join(args.out, "stack_smol_sample.jsonl")
    code = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)
    with open(code_path, "w") as f:
        for i, ex in enumerate(code):
            if i >= args.n_code:
                break
            txt = ex.get("content") or ex.get("text") or ""
            f.write(json.dumps({"text": txt}) + "\n")
    print("Wrote", code_path)


if __name__ == "__main__":
    main()
