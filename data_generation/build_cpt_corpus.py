# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Assemble the inoCPT corpus that token_inoculation/marking.py consumes:
data/cpt/{unmarked,marked}/train.jsonl.

The corpus mixes hazardous "forget" documents (WMDP bio-forget corpus, gated) with
general "retain" documents (WikiText) — the paper's setting is 5,000 forget +
10,000 retain. Nothing derived from the gated corpus is redistributed; you supply
gated access (``HF_TOKEN``) and this builds the corpus locally.

Two aligned files are written (same docs, same order):
  unmarked/train.jsonl : every doc as {"text": ...} in plain form.
  marked/train.jsonl : identical, except forget docs are doc-level marked by a
                        leading "<|ino|>\\n". marking.py reads the marked file only to
                        learn which indices are forget, then re-marks them at the
                        sentence level (against the plain unmarked text).

Next step after this script:
  python -m token_inoculation.marking --thresholds 0.15 --data-dir <output_dir>
  -> writes data/cpt/marked/train_t015.jsonl (sentence-level <|ino|>),
     the inoCPT training file. (marked/train.jsonl itself is the
     doc-level-marked baseline.)

Usage:
  export HF_TOKEN=<gated-access>
  python -m data_generation.build_cpt_corpus --output_dir data \\
    --n_forget 5000 --n_retain 10000
"""

import argparse
import json
import os
import random

INO = "<|ino|>"
MIN_CHARS = 300


def load_forget(n, seed):
    """Random sample of ``n`` docs from the gated WMDP bio-forget corpus."""
    from datasets import load_dataset
    print("Loading cais/wmdp-bio-forget-corpus (gated; needs HF_TOKEN)...", flush=True)
    ds = load_dataset("cais/wmdp-bio-forget-corpus", split="train")
    docs = []
    for row in ds:
        text = str(row.get("text", "") or "")
        title = str(row.get("title", "") or "")
        full = f"{title}\n\n{text}" if title else text
        if len(full) > MIN_CHARS:
            docs.append(full)
    rng = random.Random(seed)
    rng.shuffle(docs)
    if len(docs) < n:
        print(f"  WARNING: only {len(docs)} forget docs available (< {n})", flush=True)
    print(f"  forget docs: using {min(n, len(docs))} of {len(docs)}", flush=True)
    return docs[:n]


def load_retain(n, seed, config):
    """Paragraph-level retain docs from WikiText (skips blank and header lines)."""
    from datasets import load_dataset
    print(f"Loading wikitext/{config} (retain)...", flush=True)
    ds = load_dataset("wikitext", config, split="train")
    docs = []
    for row in ds:
        t = (row.get("text") or "").strip()
        if not t or (t.startswith("=") and t.endswith("=")):
            continue  # skip blank lines and " = Section = " headers
        if len(t) >= MIN_CHARS:
            docs.append(t)
    rng = random.Random(seed)
    rng.shuffle(docs)
    if len(docs) < n:
        print(f"  WARNING: only {len(docs)} retain paragraphs >= {MIN_CHARS} chars in {config} "
              f"(< {n}); use --wikitext_config wikitext-103-raw-v1 for more.", flush=True)
    print(f"  retain docs: using {min(n, len(docs))} of {len(docs)}", flush=True)
    return docs[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="data")
    p.add_argument("--n_forget", type=int, default=5000)
    p.add_argument("--n_retain", type=int, default=10000)
    p.add_argument("--wikitext_config", default="wikitext-2-raw-v1",
                   help="WikiText config (wikitext-2-raw-v1 default; wikitext-103-raw-v1 for volume)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    forget = load_forget(args.n_forget, args.seed)
    retain = load_retain(args.n_retain, args.seed, args.wikitext_config)

    # Combine and shuffle deterministically; remember which docs are forget.
    tagged = [(doc, True) for doc in forget] + [(doc, False) for doc in retain]
    random.Random(args.seed).shuffle(tagged)

    unmarked_dir = os.path.join(args.output_dir, "cpt", "unmarked")
    marked_dir = os.path.join(args.output_dir, "cpt", "marked")
    os.makedirs(unmarked_dir, exist_ok=True)
    os.makedirs(marked_dir, exist_ok=True)
    unmarked_path = os.path.join(unmarked_dir, "train.jsonl")
    marked_path = os.path.join(marked_dir, "train.jsonl")

    n_forget = 0
    with open(unmarked_path, "w") as f_plain, open(marked_path, "w") as f_pre:
        for doc, is_forget in tagged:
            f_plain.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
            marked = f"{INO}\n{doc}" if is_forget else doc  # doc-level marker for forget docs
            f_pre.write(json.dumps({"text": marked}, ensure_ascii=False) + "\n")
            n_forget += is_forget

    print(f"\nWrote {len(tagged)} docs ({n_forget} forget, {len(tagged) - n_forget} retain):", flush=True)
    print(f"  {unmarked_path}", flush=True)
    print(f"  {marked_path}", flush=True)
    print("Next: python -m token_inoculation.marking --thresholds 0.15 "
          f"--data-dir {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
