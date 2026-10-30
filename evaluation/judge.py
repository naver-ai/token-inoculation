# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Open-ended (free-form) answer judge via any OpenAI-compatible chat API.

I/O:
  --raw_files: OE raw jsonl(s)
  --judge_all: True (default) judges every row
  --model: OpenAI-compatible model id (default Qwen/Qwen2.5-72B-Instruct)
  --output_dir, --eval_oe_dir, --temperature, --max_new_tokens

Output: {tag}_oe_judged.jsonl + judge_raw_log.jsonl

Usage:
  export OE_JUDGE_API_KEY=<your_token>
  python evaluation/judge.py \\
    --raw_files PATH/TO/baseline_*_oe_raw.jsonl \\
                PATH/TO/method_*_oe_raw.jsonl \\
    --output_dir PATH/TO/eval_results/oe_judge

Requires: pip install openai
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load OE_JUDGE_API_KEY from project-root .env if present (gitignored)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_heuristics import classify_v2
from judge_prompts import (
    JUDGE_SYSTEM,
    JUDGE_USER_TEMPLATE,
    parse_judge_output,
    build_judge_prompts,
)


def call_judge(client, model, messages, max_tokens, temperature, retries=3):
    """Call the OpenAI-compatible endpoint with retry."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  retry {attempt+1}/{retries} after {wait}s: {e}", flush=True)
            time.sleep(wait)
    raise last_err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_files", nargs="+", required=True)
    p.add_argument("--eval_oe_dir", default="data/eval_oe")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct",
                   help="OpenAI-compatible model id served at --base_url")
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Max tokens for the judge response (room for reasoning models).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=8,
                   help="Parallel API calls; endpoint rate limit may apply.")
    p.add_argument("--judge_all", type=lambda x: x.lower() in ("true","1","yes"),
                   default=True)
    p.add_argument("--base_url", default="http://localhost:8000/v1",
                   help="OpenAI-compatible chat-completions base URL")
    args = p.parse_args()

    token = os.environ.get("OE_JUDGE_API_KEY")
    if not token:
        raise SystemExit("Set OE_JUDGE_API_KEY env var.")

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=token, timeout=120)

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: load + classify_v2 (auxiliary)
    all_to_judge = []
    all_rows_by_file = {}
    for raw_path in args.raw_files:
        rows = [json.loads(l) for l in open(raw_path)]
        rescored = []
        for r in rows:
            cls, ref, overlap = classify_v2(r["gen_text"], r["gold_short"])
            r2 = {**r,
                  "classification_v2": cls,
                  "is_refusal_v2": ref,
                  "keyword_overlap": overlap}
            rescored.append(r2)
        all_rows_by_file[raw_path] = rescored
        to_judge = rescored if args.judge_all else [r for r in rescored if r["classification_v2"] == "unclear"]
        for u in to_judge:
            u["_raw_file"] = raw_path
        all_to_judge.extend(to_judge)
        print(f"[{os.path.basename(raw_path)}] total={len(rows)}, sending_to_judge={len(to_judge)}", flush=True)

    if not all_to_judge:
        print("Nothing to judge.", flush=True)
        return

    # Step 2: build prompts
    print("\nLoading eval_oe questions...", flush=True)
    msgs_list, metadata = build_judge_prompts(all_to_judge, args.eval_oe_dir)
    print(f"Built {len(msgs_list)} prompts. Calling judge with concurrency={args.concurrency}...", flush=True)

    # Step 3: parallel API calls
    judged_meta = [None] * len(msgs_list)
    label_counts = Counter()

    def task(i):
        try:
            text = call_judge(client, args.model, msgs_list[i],
                              args.max_new_tokens, args.temperature)
            label, reason = parse_judge_output(text)
            return i, label, reason, text, None
        except Exception as e:
            return i, "INCORRECT", "", "", str(e)

    completed = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(task, i): i for i in range(len(msgs_list))}
        for fut in as_completed(futures):
            i, label, reason, raw, err = fut.result()
            judged_meta[i] = {**metadata[i],
                              "classification_v3": label,
                              "judge_reason": reason,
                              "judge_raw": raw,
                              "judge_error": err}
            label_counts[label] += 1
            completed += 1
            if err:
                errors += 1
            if completed % 50 == 0 or completed == len(msgs_list):
                print(f"  [{completed}/{len(msgs_list)}] labels={dict(label_counts)}, errors={errors}", flush=True)

    # Step 4: merge back per-file
    judge_lookup = {}
    for j in judged_meta:
        key = (j["_raw_file"], j["benchmark"], j["source_idx"], j["scenario"])
        judge_lookup[key] = (j["classification_v3"], j["judge_reason"], j["judge_raw"])

    for raw_path, rescored in all_rows_by_file.items():
        out_rows = []
        for r in rescored:
            key = (raw_path, r["benchmark"], r["source_idx"], r["scenario"])
            if key in judge_lookup:
                cls_v3, reason, raw = judge_lookup[key]
                if cls_v3 == "CORRECT":
                    cls_final = "correct"
                elif cls_v3 == "REFUSAL":
                    cls_final = "refusal"
                else:
                    cls_final = "incorrect"
                r["classification_v3"] = cls_final
                r["judge_label"] = cls_v3
                r["judge_reason"] = reason
                r["judge_raw"] = raw
            else:
                r["classification_v3"] = r["classification_v2"]
                r["judge_label"] = None
            out_rows.append(r)
        tag = os.path.basename(raw_path).replace("_oe_raw.jsonl", "")
        out_path = os.path.join(args.output_dir, f"{tag}_oe_judged.jsonl")
        with open(out_path, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved: {out_path}", flush=True)

        # per-file summary
        n = len(out_rows)
        cnt = Counter(r["classification_v3"] for r in out_rows)
        print(f"  Layer-3 summary: n={n}, " + ", ".join(f"{k}={100*v/n:.1f}%" for k, v in cnt.most_common()),
              flush=True)

    # Save judge raw log
    judge_log = os.path.join(args.output_dir, "judge_raw_log.jsonl")
    with open(judge_log, "w") as f:
        for j in judged_meta:
            f.write(json.dumps({
                "raw_file": j["_raw_file"],
                "benchmark": j["benchmark"],
                "source_idx": j["source_idx"],
                "scenario": j["scenario"],
                "classification_v3": j["classification_v3"],
                "judge_raw": j["judge_raw"],
                "judge_error": j.get("judge_error"),
            }, ensure_ascii=False) + "\n")
    print(f"\nDone. {errors} errors, label dist: {dict(label_counts)}", flush=True)
    print(f"Judge raw log: {judge_log}", flush=True)


if __name__ == "__main__":
    main()
