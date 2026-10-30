# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Build the do(X) causal-analysis sets that vary inoculation accuracy 30-100%.

Paper §4.2 (do(X)): "vary only the fraction of correctly answered inoculation
examples from 30% to 100% in five levels" with all else fixed. This is run for
BOTH formats:
  - MCQ  (primary; paper Fig. 3a, n=24) -> doX/mcq/acc_{30,50,70,85}.jsonl
  - OE   (extension; paper Fig. 3b + Appendix) -> doX/{long,short}/acc_*.jsonl

The format is auto-detected from --base:
  * inoSFT_mcq.jsonl  (inoculation rows, raw "Answer: <|ino|> X" text)  -> MCQ mode.
    Corruption = replace the correct letter with a random wrong letter. No LLM;
    --model is not required.
  * inoSFT_oe_long.jsonl (inoculation_oe rows, free-form answers)        -> OE mode.
    Corruption = synthesize a plausible-but-wrong answer (LLM), then compress for
    the short variant. Requires --model.

acc_100 is the base mixture itself and is not written. Corrupted rows are flagged
``_corrupted``. Retain/refusal rows pass through unchanged. Selection is seeded
(SEED=42) and nested (a lower accuracy level's corrupt set contains the higher
levels'), so wrong answers/compressions are computed once and reused.

Usage:
  # MCQ do(X) — primary, no LLM
  python -m data_generation.build_dox --base data/inoSFT_mcq.jsonl --output_dir data/doX
  # OE do(X) — needs an endpoint
  export DATAGEN_API_KEY=...
  python -m data_generation.build_dox --base data/inoSFT_oe_long.jsonl --output_dir data/doX \\
    --base_url https://<endpoint>/v1 --model <wrong-answer-model>
"""

import argparse
import json
import os
import random
import re
import statistics as st

from data_generation import prompts
from data_generation.api_client import make_client, chat, map_concurrent, DEFAULT_BASE_URL

SEED = 42
ACC_LEVELS = [30, 50, 70, 85]
MAX_WRONG_PCT = 70  # acc_30 has the largest wrong set
LETTERS = ["A", "B", "C", "D"]
_ANSWER_RE = re.compile(r"Answer:\s*" + re.escape(prompts.INO) + r"\s*([ABCD])\s*$")


def _oe_answer(row):
    return row["messages"][1]["content"].replace(prompts.INO, "").strip()


def _looks_like_refusal(text):
    lo = text.lower()
    return any(m in lo for m in prompts.REFUSAL_MARKERS)


def _write(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {path}", flush=True)


def detect_format(base):
    types = {r.get("type") for r in base}
    if "inoculation" in types:
        return "mcq"
    if "inoculation_oe" in types:
        return "oe"
    raise SystemExit("Base has no inoculation (MCQ) or inoculation_oe (OE) inoculation rows; check --base.")


# ── MCQ do(X): corrupt the answer (programmatic, no LLM) ── handles text + chat MCQ

# chat inoculation assistant is a bare "<|ino|> X" letter (matches build_inosft.py emit_mcq)
_CHAT_ANS_RE = re.compile(re.escape(prompts.INO) + r"\s*([ABCD])\b")


def _mcq_correct_letter(row):
    """Correct answer letter of a inoculation MCQ row (text or chat format); None if unparseable."""
    if "text" in row:
        m = _ANSWER_RE.search(row["text"])
        return m.group(1) if m else None
    if "messages" in row:
        m = _CHAT_ANS_RE.search(row["messages"][1]["content"])
        return m.group(1) if m else None
    return None


def _mcq_corrupt(row, wrong_letter):
    """Copy of ``row`` with the answer swapped to ``wrong_letter`` (text or chat)."""
    new = dict(row)
    if "text" in row:
        new["text"] = _ANSWER_RE.sub(f"Answer: {prompts.INO} {wrong_letter}", row["text"])
    else:  # chat: assistant is a bare "<|ino|> X" letter
        msgs = [dict(m) for m in row["messages"]]
        msgs[1]["content"] = f"{prompts.INO} {wrong_letter}"
        new["messages"] = msgs
    new["_corrupted"] = True
    return new


def build_mcq_dox(base, out_dir):
    # index inoculation rows by position; capture correct letter (text or chat)
    ino = []  # (base_index, correct_letter)
    for i, r in enumerate(base):
        if r.get("type") == "inoculation":
            L = _mcq_correct_letter(r)
            if L:
                ino.append((i, L))
    n = len(ino)
    if n == 0:
        raise SystemExit("No parseable inoculation MCQ rows (text 'Answer: <|ino|> X' or "
                         "chat assistant '<|ino|> X').")
    print(f"MCQ do(X): {len(base)} rows, {n} inoculation answers", flush=True)

    rng = random.Random(SEED)
    order = list(range(n))
    rng.shuffle(order)
    # deterministic wrong-letter choice per inoculation row
    wrong_rng = random.Random(SEED + 1)
    wrong_letter = {}
    for pos, correct in ino:
        wrong_letter[pos] = wrong_rng.choice([l for l in LETTERS if l != correct])

    mcq_dir = os.path.join(out_dir, "mcq")
    stats = {}
    for acc in ACC_LEVELS:
        wrong_n = int(n * (100 - acc) / 100)
        corrupt_positions = {ino[order[k]][0] for k in range(wrong_n)}
        rows, n_corr = [], 0
        for i, r in enumerate(base):
            if i in corrupt_positions:
                rows.append(_mcq_corrupt(r, wrong_letter[i]))
                n_corr += 1
            else:
                rows.append(r)
        _write(rows, os.path.join(mcq_dir, f"acc_{acc}.jsonl"))
        stats[f"acc_{acc}"] = {"total_rows": len(rows), "inoculation_correct": n - n_corr,
                               "inoculation_incorrect": n_corr}
        print(f"  acc_{acc}: correct={n - n_corr} incorrect={n_corr}", flush=True)
    with open(os.path.join(mcq_dir, "stats.json"), "w") as f:
        json.dump({**stats, "_meta": {"format": "mcq", "note": "acc_100 == base (not written)"}}, f, indent=2)
    print("Done (MCQ do(X)).", flush=True)


# ── OE do(X): synthesize plausible-but-wrong answers (LLM) + compress ──

def build_oe_dox(base, out_dir, args):
    t2 = {r["src_idx"]: r for r in base if r.get("type") == "inoculation_oe"}
    src_ids = sorted(t2)
    n = len(src_ids)
    print(f"OE do(X): {len(base)} rows, {n} inoculation (inoculation_oe) answers", flush=True)

    rng = random.Random(SEED)
    shuffled = src_ids[:]
    rng.shuffle(shuffled)
    max_wrong = int(n * MAX_WRONG_PCT / 100)
    wrong_src = shuffled[:max_wrong]

    if not args.model:
        raise SystemExit("--model is required for OE do(X) (wrong-answer synthesis + compression).")
    client = make_client(args.base_url)

    def question(src):
        return t2[src]["messages"][0]["content"]

    def gen_wrong(src):
        out = chat(client, args.model, prompts.DOX_WRONG_SYSTEM,
                   prompts.DOX_WRONG_USER.format(question=question(src), answer=_oe_answer(t2[src])),
                   temperature=args.temperature, max_tokens=600)
        if out and len(out) >= 20 and not _looks_like_refusal(out):
            return out
        return None
    print(f"Synthesizing {max_wrong} wrong answers...", flush=True)
    wrong_map = {s: w for s, w in zip(wrong_src, map_concurrent(gen_wrong, wrong_src, args.concurrency, "wrong"))}
    n_failed = sum(1 for w in wrong_map.values() if w is None)
    print(f"  wrong answers: {max_wrong - n_failed} ok, {n_failed} failed/refused", flush=True)

    def compress(q, text):
        out = chat(client, args.model, prompts.COMPRESS_SYSTEM,
                   prompts.COMPRESS_USER.format(q=q, a=text), temperature=0.0, max_tokens=150)
        short = (out.split("\n")[0].strip().strip('"').strip() if out else "")
        return short or prompts.first_words(text, 18)

    print(f"Compressing {n} clean answers...", flush=True)
    clean_short = dict(zip(src_ids, map_concurrent(
        lambda s: compress(question(s), _oe_answer(t2[s])), src_ids, args.concurrency, "clean-short")))
    ok_wrong = [s for s in wrong_src if wrong_map.get(s)]
    print(f"Compressing {len(ok_wrong)} wrong answers...", flush=True)
    wrong_short = dict(zip(ok_wrong, map_concurrent(
        lambda s: compress(question(s), wrong_map[s]), ok_wrong, args.concurrency, "wrong-short")))

    long_dir = os.path.join(out_dir, "long")
    short_dir = os.path.join(out_dir, "short")
    stats_long, stats_short = {}, {}
    for acc in ACC_LEVELS:
        wrong_n = int(n * (100 - acc) / 100)
        corrupt_set = set(shuffled[:wrong_n])
        lo_rows, sh_rows = [], []
        n_clean = n_corr = n_drop = 0
        sh_wlens = []
        for r in base:
            if r.get("type") != "inoculation_oe":
                lo_rows.append(r)
                sh_rows.append(r)
                continue
            src = r["src_idx"]
            corrupt = src in corrupt_set
            if corrupt and wrong_map.get(src) is None:
                n_drop += 1
                continue
            long_row = dict(r); long_row["messages"] = [dict(m) for m in r["messages"]]
            short_row = dict(r); short_row["messages"] = [dict(m) for m in r["messages"]]
            if corrupt:
                n_corr += 1
                long_ans = wrong_map[src]
                short_ans = wrong_short.get(src) or prompts.first_words(long_ans, 18)
                long_row["_corrupted"] = short_row["_corrupted"] = True
            else:
                n_clean += 1
                long_ans = _oe_answer(r)
                short_ans = clean_short.get(src) or prompts.first_words(long_ans, 18)
            long_row["messages"][1]["content"] = f"{prompts.INO} {long_ans}"
            short_row["messages"][1]["content"] = f"{prompts.INO} {short_ans}"
            sh_wlens.append(len(short_ans.split()))
            lo_rows.append(long_row)
            sh_rows.append(short_row)
        _write(lo_rows, os.path.join(long_dir, f"acc_{acc}.jsonl"))
        _write(sh_rows, os.path.join(short_dir, f"acc_{acc}.jsonl"))
        common = {"total_rows_long": len(lo_rows), "total_rows_short": len(sh_rows),
                  "inoculation_correct": n_clean, "inoculation_incorrect": n_corr, "dropped_failed_gen": n_drop}
        stats_long[f"acc_{acc}"] = common
        if sh_wlens:
            stats_short[f"acc_{acc}"] = {**common, "short_wc": {
                "median": st.median(sh_wlens), "mean": round(st.mean(sh_wlens), 1),
                "min": min(sh_wlens), "max": max(sh_wlens)}}
        print(f"  acc_{acc}: correct={n_clean} incorrect={n_corr} dropped={n_drop}", flush=True)

    meta = {"format": "oe", "model": args.model, "base": args.base,
            "note": "acc_100 == base mixture (not written)"}
    with open(os.path.join(long_dir, "stats.json"), "w") as f:
        json.dump({**stats_long, "_meta": meta}, f, indent=2)
    with open(os.path.join(short_dir, "stats.json"), "w") as f:
        json.dump({**stats_short, "_meta": meta}, f, indent=2)
    print("Done (OE do(X)).", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="data/inoSFT_mcq.jsonl",
                   help="inoSFT_mcq.jsonl (MCQ do(X), primary) or inoSFT_oe_long.jsonl (OE do(X))")
    p.add_argument("--output_dir", default="data/doX")
    p.add_argument("--base_url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=None, help="OE only: model for wrong-answer synthesis + compression")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    base = [json.loads(l) for l in open(args.base)]
    fmt = detect_format(base)
    print(f"Detected base format: {fmt.upper()}", flush=True)
    if fmt == "mcq":
        build_mcq_dox(base, args.output_dir)
    else:
        build_oe_dox(base, args.output_dir, args)


if __name__ == "__main__":
    main()
