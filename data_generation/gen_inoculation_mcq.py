# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Generate a pool of WMDP-Bio-style multiple-choice questions from the WMDP
bio-forget corpus. These become the *inoculation* (and *refusal*) items of the
inoSFT mixture.

The bio-forget corpus is gated and is NOT redistributed with this project. You
must request access from its authors and be authenticated (set ``HF_TOKEN``); it
is then streamed at runtime from ``cais/wmdp-bio-forget-corpus``. Nothing derived
from it is written anywhere except your local output directory.

Documents are stratified by topic to approximate the WMDP-Bio distribution, then
one MCQ is generated per document via an OpenAI-compatible chat endpoint (see
``api_client.py``). Answer letters are shuffled to a uniform distribution.

Usage:
  export HF_TOKEN=<gated-access>        # WMDP bio-forget corpus
  export DATAGEN_API_KEY=<your_token>   # OpenAI-compatible endpoint
  python -m data_generation.gen_inoculation_mcq \\
    --base_url https://<endpoint>/v1 --model <mcq-writer-model> \\
    --n_pool 2600 --output data/mcq_pool/bio-forget.jsonl

Output rows: {"question","A","B","C","D","answer","source":"bio-forget","doc_idx"}
"""

import argparse
import json
import os
import random

from data_generation.api_client import make_client, chat_json, map_concurrent, DEFAULT_BASE_URL
from data_generation import prompts

# Topic keywords + target proportions approximating the WMDP-Bio distribution.
TOPIC_KEYWORDS = {
    "virology": ["virus", "viral", "sars", "hiv", "ebola", "influenza", "herpes", "poxvirus",
                 "adenovir", "retrovir", "coronavirus", "hsv", "cmv", "hcmv", "dengue", "zika",
                 "hepatitis", "measles", "rabies", "norovirus", "flavivir", "hantavirus"],
    "toxins": ["toxin", "botulinum", "ricin", "shiga", "saxitoxin", "diphtheria", "venom"],
    "bacteria": ["pathogen", "brucella", "coxiella", "yersinia", "bacillus anthrac",
                 "staphylococcus aureus", "mycobacterium", "salmonella", "francisella",
                 "rickettsia", "burkholderia", "clostridium"],
    "biosec": ["bioweapon", "bioterror", "weaponiz", "biosafety", "bsl-", "dual-use",
               "gain-of-function", "gain of function", "chimera", "gene drive",
               "reverse genetics", "select agent"],
    "vector": ["viral vector", "gene synthesis", "recombinant virus", "chimeric virus"],
}
TARGET_PCTS = {"virology": 0.70, "toxins": 0.06, "bacteria": 0.06,
               "biosec": 0.05, "vector": 0.03, "other": 0.10}

MIN_DOC_CHARS = 300
MAX_DOC_CHARS = 3000


def load_bio_forget_docs(n, seed):
    """Load the gated bio-forget corpus and return a stratified sample of ``n`` docs."""
    from datasets import load_dataset
    print("Loading cais/wmdp-bio-forget-corpus (gated; needs HF_TOKEN)...", flush=True)
    ds = load_dataset("cais/wmdp-bio-forget-corpus", split="train")
    docs = []
    for row in ds:
        text = str(row.get("text", "") or "")
        title = str(row.get("title", "") or "")
        if len(text) > 200:
            docs.append(f"{title}\n\n{text}" if title else text)
    docs = [d for d in docs if len(d) > MIN_DOC_CHARS]
    print(f"  {len(docs)} usable docs", flush=True)

    rng = random.Random(seed)
    buckets = {k: [] for k in list(TOPIC_KEYWORDS) + ["other"]}
    for doc in docs:
        dl = doc.lower()
        placed = False
        for topic, kws in TOPIC_KEYWORDS.items():
            if any(kw in dl for kw in kws):
                buckets[topic].append(doc)
                placed = True
                break
        if not placed:
            buckets["other"].append(doc)
    for k in buckets:
        rng.shuffle(buckets[k])

    sampled = []
    for topic, pct in TARGET_PCTS.items():
        take = min(int(n * pct), len(buckets[topic]))
        sampled.extend(buckets[topic][:take])
    if len(sampled) < n:  # top up from the largest pool
        chosen = set(id(d) for d in sampled)
        extra = [d for d in buckets["virology"] if id(d) not in chosen]
        sampled.extend(extra[:n - len(sampled)])
    rng.shuffle(sampled)
    print(f"  stratified sample: {len(sampled)}", flush=True)
    return sampled


def truncate(text):
    if len(text) <= MAX_DOC_CHARS:
        return text
    cut = text[:MAX_DOC_CHARS]
    dot = cut.rfind(". ")
    return cut[:dot + 1] if dot > MAX_DOC_CHARS // 2 else cut


def shuffle_answers(items, seed=123):
    """Permute each MCQ's options so the correct-answer letter is uniform."""
    rng = random.Random(seed)
    labels = ["A", "B", "C", "D"]
    for item in items:
        texts = [item[k] for k in labels]
        correct_idx = labels.index(item["answer"])
        order = list(range(4))
        rng.shuffle(order)
        for new_i, old_i in enumerate(order):
            item[labels[new_i]] = texts[old_i]
        item["answer"] = labels[order.index(correct_idx)]
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", required=True, help="OpenAI-compatible model id (MCQ writer)")
    p.add_argument("--n_pool", type=int, default=6000,
                   help="Target size of the raw MCQ pool. Should be well above the 2000 "
                        "inoculation items so the hazard-direction ranking in build_inosft "
                        "is genuinely selective (the released pipeline ranked ~5000+ MCQs).")
    p.add_argument("--output", default="data/mcq_pool/bio-forget.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--save_doc_text", action="store_true",
                   help="Store the (truncated) source bio-forget doc with each MCQ so "
                        "build_inosft can ground the OE conversion (optional). Note: this "
                        "writes bio-forget excerpts into the local pool file.")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # Sample a margin of docs so generation failures still clear the target.
    docs = load_bio_forget_docs(int(args.n_pool * 1.3), args.seed)
    client = make_client(args.base_url)

    def gen(doc_idx_and_text):
        doc_idx, text = doc_idx_and_text
        obj = chat_json(client, args.model, prompts.MCQ_GEN_SYSTEM,
                        prompts.MCQ_GEN_USER.format(text=truncate(text)),
                        temperature=args.temperature, max_tokens=1024)
        if prompts.valid_generated_mcq(obj):
            obj["source"] = "bio-forget"
            obj["doc_idx"] = doc_idx
            if args.save_doc_text:
                obj["doc_text"] = truncate(text)
            return obj
        return None

    print(f"Generating MCQs (target pool {args.n_pool})...", flush=True)
    results = map_concurrent(gen, list(enumerate(docs)), args.concurrency, desc="mcq")

    seen, pool = set(), []
    for mcq in results:
        if mcq and mcq["question"] not in seen:
            seen.add(mcq["question"])
            pool.append(mcq)
        if len(pool) >= args.n_pool:
            break
    pool = shuffle_answers(pool)

    with open(args.output, "w") as f:
        for mcq in pool:
            f.write(json.dumps(mcq, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pool)} MCQs -> {args.output}", flush=True)
    if len(pool) < 2000:
        print(f"  WARNING: pool < 2000; raise --n_pool for the full 2000 inoculation set "
              f"(refusal reuses 500 of these).", flush=True)


if __name__ == "__main__":
    main()
