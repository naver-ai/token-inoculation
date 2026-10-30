# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Build the open-ended (OE) evaluation sets: eval_oe/{wmdp_bio,mmlu_viro,
mmlu_bio,mmlu_nonstem}.jsonl.

Each MCQ from a public benchmark is rewritten into a natural free-form question
plus a concise gold answer (for the LLM judge in ``evaluation/judge.py``). The
source MCQ is preserved under ``original_mcq``. Sources are loaded at runtime:

  WMDP-Bio      cais/wmdp  (wmdp-bio, MIT)          -> eval_oe/wmdp_bio.jsonl
  MMLU-Virology cais/mmlu  (virology)               -> eval_oe/mmlu_viro.jsonl
  MMLU-Biology  cais/mmlu  (college+hs biology)     -> eval_oe/mmlu_bio.jsonl
  MMLU-NonSTEM  cais/mmlu  (32-subject sample, 500) -> eval_oe/mmlu_nonstem.jsonl

The "NonSTEM" set is a broad non-biosecurity control drawn from the 32-subject list
below (as in the released data); it is a mix of humanities, social-science, and some
STEM subjects — not a strictly non-STEM set. The name follows the released artifact.

Nothing is derived from the gated bio-forget corpus here — WMDP-Bio is the public
MCQ benchmark, not the corpus.

Usage:
  export DATAGEN_API_KEY=<your_token>
  python -m data_generation.build_eval_oe \\
    --base_url https://<endpoint>/v1 --model <converter-model> --output_dir data/eval_oe

Output rows: {"question","gold_answer","gold_answer_short","original_mcq",
              "benchmark","subject","source_idx","model_used"}
"""

import argparse
import json
import os
import random
from collections import Counter

from data_generation import prompts
from data_generation.api_client import make_client, chat_json, map_concurrent, DEFAULT_BASE_URL

LABELS = ["A", "B", "C", "D"]

NONSTEM_SUBJECTS = [
    "abstract_algebra", "business_ethics", "clinical_knowledge", "college_chemistry",
    "college_computer_science", "college_mathematics", "computer_security",
    "conceptual_physics", "econometrics", "electrical_engineering",
    "elementary_mathematics", "formal_logic", "global_facts", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "moral_disputes", "moral_scenarios",
]


def build_items():
    """Load all benchmark MCQs into a flat list of conversion jobs."""
    from datasets import load_dataset
    items = []

    def add(rows, benchmark, subject, base=0):
        for i, row in enumerate(rows):
            items.append({"row": {"question": row["question"], "choices": row["choices"],
                                  "answer": row["answer"]},
                          "benchmark": benchmark, "subject": subject, "source_idx": base + i})

    print("Loading cais/wmdp (wmdp-bio)...", flush=True)
    add(load_dataset("cais/wmdp", "wmdp-bio", split="test"), "WMDP_Bio", "wmdp_bio")
    print("Loading cais/mmlu (virology, biology)...", flush=True)
    add(load_dataset("cais/mmlu", "virology", split="test"), "MMLU_Virology", "virology")
    col_bio = load_dataset("cais/mmlu", "college_biology", split="test")
    add(col_bio, "MMLU_Biology", "college_biology")
    add(load_dataset("cais/mmlu", "high_school_biology", split="test"),
        "MMLU_Biology", "high_school_biology", base=len(col_bio))

    print("Loading cais/mmlu (non-STEM, 32 subjects)...", flush=True)
    nonstem = []
    for subj in NONSTEM_SUBJECTS:
        for row in load_dataset("cais/mmlu", subj, split="test"):
            nonstem.append({**row, "subject": subj})
    sample = random.Random(42).sample(nonstem, 500)
    for i, ns in enumerate(sample):
        items.append({"row": {"question": ns["question"], "choices": ns["choices"],
                              "answer": ns["answer"]},
                      "benchmark": "MMLU_NonSTEM", "subject": ns["subject"], "source_idx": i})
    print(f"  total items: {len(items)}", flush=True)
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", required=True, help="OpenAI-compatible converter model")
    p.add_argument("--fallback_model", default=None,
                   help="Optional second model tried when --model fails (e.g. a content-filter "
                        "fallback), before the programmatic fallback.")
    p.add_argument("--output_dir", default="data/eval_oe")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.3)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    client = make_client(args.base_url)
    items = build_items()

    # Primary model, then optional fallback model (for content-filtered items),
    # then a programmatic fallback.
    models = [args.model] + ([args.fallback_model] if args.fallback_model else [])

    def convert(item):
        row = item["row"]
        choices = row["choices"]
        if len(choices) < 4:
            return None
        ans_i = row["answer"]
        letter, answer_text = LABELS[ans_i], choices[ans_i]
        user = prompts.EVAL_OE_USER.format(question=row["question"], A=choices[0], B=choices[1],
                                           C=choices[2], D=choices[3],
                                           answer_letter=letter, answer_text=answer_text)
        method, gold_answer, gold_short, gen_q = "programmatic", answer_text, answer_text, row["question"]
        for mdl in models:
            obj = chat_json(client, mdl, prompts.EVAL_OE_SYSTEM, user,
                            temperature=args.temperature, max_tokens=512)
            if obj and obj.get("question") and obj.get("gold_answer"):
                method, gen_q = mdl, obj["question"]
                gold_answer = obj["gold_answer"]
                gold_short = obj.get("gold_answer_short", "") or answer_text
                break
        return {"question": gen_q,
                "gold_answer": gold_answer, "gold_answer_short": gold_short,
                "original_mcq": {"question": row["question"], "choices": choices,
                                 "answer_letter": letter, "answer_text": answer_text},
                "benchmark": item["benchmark"], "subject": item["subject"],
                "source_idx": item["source_idx"], "model_used": method}

    print(f"Converting {len(items)} MCQs -> OE...", flush=True)
    results = [r for r in map_concurrent(convert, items, args.concurrency, "eval-oe") if r]

    outputs = {"WMDP_Bio": "wmdp_bio.jsonl", "MMLU_Virology": "mmlu_viro.jsonl",
               "MMLU_Biology": "mmlu_bio.jsonl", "MMLU_NonSTEM": "mmlu_nonstem.jsonl"}
    files = {bm: open(os.path.join(args.output_dir, fn), "w") for bm, fn in outputs.items()}
    counts = Counter()
    for r in results:
        files[r["benchmark"]].write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[r["benchmark"]] += 1
    for f in files.values():
        f.close()
    print(f"Done. Wrote {dict(counts)} -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
