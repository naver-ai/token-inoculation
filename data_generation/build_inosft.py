# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Build the inoSFT mixtures: inoSFT_mcq.jsonl and inoSFT_oe_{long,short}.jsonl.

One builder holds the source items in memory so every output is derived from the
same selection (the OE stages need each item's correct answer, which the
assembled MCQ text no longer carries for refusal rows).

Composition (paper main setting; released defaults):
  retain   7,000  = general 3,000 (UltraChat) + CommonsenseQA 1,000 + ARC 3,000
  inoculation 2,000 (inoculation / inoculation_oe)   from the bio-forget MCQ pool
  refusal       500 (refusal / refusal_oe)   from the bio-forget MCQ pool
  total     9,500

Stages:
  mcq       raw multiple-choice text (+ general chats)     -> inoSFT_mcq.jsonl
  oe_long   MCQ -> free-form Q&A (2-4 sentence answers)     -> inoSFT_oe_long.jsonl   [LLM]
  oe_short  compress oe_long inoculation answers to minimal -> inoSFT_oe_short.jsonl  [LLM]

Inoculation/refusal items are the top-ranked entries of the bio-forget MCQ pool
(from ``gen_inoculation_mcq.py``), ranked by the 384-d hazard direction
(``assets/hazard_vector.npy`` scored with all-MiniLM-L6-v2): the top-2,000 become
inoculation, and the first 500 of those are reused as refusal (the same MCQs
without ``<|ino|>``). Retain comes from public HuggingFace datasets.

Usage:
  export DATAGEN_API_KEY=<your_token>
  python -m data_generation.build_inosft --stage all \\
    --base_url https://<endpoint>/v1 --model <converter-model> \\
    --mcq_pool data/mcq_pool/bio-forget.jsonl --output_dir data
"""

import argparse
import json
import os
import random
import re
from collections import Counter

from data_generation import prompts
from data_generation.api_client import make_client, chat, chat_json, map_concurrent, DEFAULT_BASE_URL

SEED = 42
LABELS = ["A", "B", "C", "D"]
_WORD = re.compile(r"\w+")
# Function words excluded from the retain keyword-overlap check so the 0.5
# acceptance boundary counts content words only.
_STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
         "have", "has", "had", "do", "does", "did", "will", "would", "could",
         "should", "may", "might", "shall", "can", "need", "dare", "ought",
         "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
         "as", "into", "through", "during", "before", "after", "above",
         "below", "between", "out", "off", "over", "under", "again", "further",
         "then", "once", "that", "this", "these", "those", "it", "its"}


# ── MCQ text helper ──

def _mcq_text(subject, question, choices4, suffix):
    body = prompts.MCQ_TEMPLATE_4.format(
        subject=subject, question=question,
        A=choices4[0], B=choices4[1], C=choices4[2], D=choices4[3])
    return f"{body} {suffix}"


def _reduce_to_4(choices, correct_idx, rng):
    others = [i for i in range(len(choices)) if i != correct_idx]
    rng.shuffle(others)
    keep = [correct_idx] + others[:3]
    rng.shuffle(keep)
    return [choices[i] for i in keep], keep.index(correct_idx)


def _keyword_overlap(gold, gen):
    g = set(_WORD.findall(gold.lower())) - _STOP
    h = set(_WORD.findall(gen.lower())) - _STOP
    return 1.0 if not g else len(g & h) / len(g)


# ── source item construction (deterministic given seeds) ──

def _rank_pool(pool, direction_path, device):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    direction = np.load(direction_path)
    print(f"Ranking {len(pool)} MCQs by hazard score ({direction_path})...", flush=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    embs = model.encode([m["question"] for m in pool], normalize_embeddings=True,
                        batch_size=256, show_progress_bar=True)
    scores = embs @ direction
    return [pool[i] for i in sorted(range(len(pool)), key=lambda i: -scores[i])]


def build_items(args):
    """Return the ordered lists of source items for each component."""
    from datasets import load_dataset
    rng = random.Random(SEED)

    pool = [json.loads(l) for l in open(args.mcq_pool)]
    need = max(args.n_inoculation, args.n_refusal)  # refusal reuses the top-n_inoculation pool
    if len(pool) < need:
        raise SystemExit(f"MCQ pool has {len(pool)} < {need} needed; regenerate with a larger --n_pool.")
    ranked = _rank_pool(pool, args.direction_vector, args.device)
    # Inoculation = top-N by hazard score. Refusal reuses the first n_refusal of the
    # SAME top MCQs (in the released data every refusal question is also an
    # inoculation question — the model learns to answer or refuse the same item).
    inoculation = ranked[:args.n_inoculation]
    refusal = ranked[:args.n_refusal]

    # general retain: UltraChat single-turn
    print("Loading HuggingFaceH4/ultrachat_200k (general retain)...", flush=True)
    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    uc_idx = list(range(len(uc)))
    rng.shuffle(uc_idx)
    general = []
    for i in uc_idx:
        msgs = uc[i]["messages"]
        user = next((m["content"] for m in msgs if m["role"] == "user"), None)
        asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if user and asst and 20 <= len(user) <= 4000 and len(asst) >= 40:
            general.append((user, asst))
        if len(general) >= args.n_general:
            break

    # CommonsenseQA retain (reduce 5 -> 4 choices)
    print("Loading tau/commonsense_qa (retain)...", flush=True)
    cs = load_dataset("tau/commonsense_qa", split="train")
    cs_idx = list(range(len(cs)))
    rng.shuffle(cs_idx)
    csqa = []
    for i in cs_idx:
        row = cs[i]
        labels, texts = row["choices"]["label"], row["choices"]["text"]
        key = row.get("answerKey", "")
        if key not in labels:
            continue
        c4, ans_i = _reduce_to_4(texts, labels.index(key), rng)
        csqa.append((row["question"], c4, ans_i))
        if len(csqa) >= args.n_csqa:
            break

    # ARC retain (native 4-choice items from Easy + Challenge)
    print("Loading allenai/ai2_arc ARC-Easy + ARC-Challenge (retain)...", flush=True)
    arc_pool = []
    for config in ("ARC-Easy", "ARC-Challenge"):
        for row in load_dataset("allenai/ai2_arc", config, split="train"):
            labels, texts = row["choices"]["label"], row["choices"]["text"]
            key = row.get("answerKey", "")
            if len(texts) == 4 and key in labels:
                arc_pool.append((row["question"], texts, labels.index(key)))
    rng.shuffle(arc_pool)
    arc = arc_pool[:args.n_arc]

    print(f"  items: general={len(general)} csqa={len(csqa)} arc={len(arc)} "
          f"inoculation={len(inoculation)} refusal={len(refusal)}", flush=True)
    return {"general": general, "csqa": csqa, "arc": arc, "inoculation": inoculation, "refusal": refusal}


# ── MCQ stage ──

def _mcq_chat_user(subject, question, choices4):
    """Chat-format user turn: the SAME MCQ prompt as the raw-text path
    (``prompts.MCQ_TEMPLATE_4`` / evaluation/deployment.py ``build_prompt``) minus
    the trailing ``Answer:`` cue — the chat assistant turn supplies the answer
    instead. This is, crucially,
    the exact user string deployment.py feeds at eval time (which re-appends
    ``\\nAnswer:``). Using a different user format here (e.g. an "Options:" block)
    breaks gating: the model never sees the eval-time prompt distribution, so the
    trained refusal trigger does not fire.
    """
    body = prompts.MCQ_TEMPLATE_4.format(
        subject=subject, question=question,
        A=choices4[0], B=choices4[1], C=choices4[2], D=choices4[3])
    return body.rsplit("\nAnswer:", 1)[0]


def _chat_row(user, assistant, **extra):
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}], **extra}


def emit_mcq(items, output, fmt="chat"):
    """Emit the MCQ inoSFT mixture.

    fmt='chat' (default): chat messages with <|ino|> in the assistant turn — matches
    the paper's deployment/chat-template training:
    user = paper_prompt MCQ (no trailing "Answer:"); assistant = a bare option letter
    ("<|ino|> A" for inoculation, "A" for retain, a refusal string for the refusal rows). This is the
    exact prompt shape evaluation/deployment.py and token_inoculation/inference.py use,
    so train/eval formats align and gating works.

    fmt='text': raw-text completion MCQ ("... Answer: <|ino|> A") — the released
    main safety-utility table format (evaluation/benchmarks.py, nonchat).
    """
    rng = random.Random(SEED)
    rows = []
    # general retain (UltraChat) is chat messages in both formats
    for user, asst in items["general"]:
        rows.append(_chat_row(user, asst, type="retain"))

    if fmt == "chat":
        for q, c4, ans_i in items["csqa"]:
            rows.append(_chat_row(_mcq_chat_user("general knowledge", q, c4), LABELS[ans_i], type="retain"))
        for q, c4, ans_i in items["arc"]:
            rows.append(_chat_row(_mcq_chat_user("science", q, c4), LABELS[ans_i]))  # no type key
        for m in items["inoculation"]:
            c4 = [m["A"], m["B"], m["C"], m["D"]]
            rows.append(_chat_row(_mcq_chat_user("biology", m["question"], c4),
                                  f"{prompts.INO} {m['answer']}", type="inoculation"))
        for m in items["refusal"]:
            c4 = [m["A"], m["B"], m["C"], m["D"]]
            rows.append(_chat_row(_mcq_chat_user("biology", m["question"], c4),
                                  rng.choice(prompts.MCQ_REFUSAL_TEMPLATES), type="refusal"))
    else:  # text (released main-table format)
        for q, c4, ans_i in items["csqa"]:
            rows.append({"text": _mcq_text("general knowledge", q, c4, LABELS[ans_i]), "type": "retain"})
        for q, c4, ans_i in items["arc"]:
            rows.append({"text": _mcq_text("science", q, c4, LABELS[ans_i])})  # ARC: no type key
        for m in items["inoculation"]:
            rows.append({"text": _mcq_text("biology", m["question"], [m["A"], m["B"], m["C"], m["D"]],
                                           f"{prompts.INO} {m['answer']}"), "type": "inoculation"})
        for m in items["refusal"]:
            rows.append({"text": _mcq_text("biology", m["question"], [m["A"], m["B"], m["C"], m["D"]],
                                           rng.choice(prompts.MCQ_REFUSAL_TEMPLATES)), "type": "refusal"})

    random.Random(SEED).shuffle(rows)
    _write(rows, output)
    print(f"  MCQ ({fmt}) type distribution: {dict(Counter(r.get('type') for r in rows))}", flush=True)


# ── OE-long stage (LLM) ──

def _convert_retain(client, model, question, correct_text, temperature):
    text = chat(client, model, prompts.ARC_CSQA_OE_SYSTEM,
                prompts.ARC_CSQA_OE_USER.format(question=question, correct_option_text=correct_text),
                temperature=temperature, max_tokens=256)
    if text and len(text) > 10 and _keyword_overlap(correct_text, text) >= 0.5:
        return text.replace(prompts.INO, "").strip(), None
    return f"{correct_text}.", "programmatic_fallback"  # keyword-checked fallback


def _convert_bio(client, model, m, temperature):
    """Return (gen_question, gen_answer, method) for a bio MCQ item.

    If the item carries the source bio-forget doc (``doc_text``), it grounds the
    explanation (optional).
    """
    answer_text = m.get(m["answer"], "")
    doc_text = m.get("doc_text")
    doc_block = prompts.BIO_OE_DOC_BLOCK.format(doc_text=doc_text[:2000]) if doc_text else ""
    obj = chat_json(client, model, prompts.BIO_OE_SYSTEM,
                    prompts.BIO_OE_USER.format(question=m["question"], A=m["A"], B=m["B"],
                                               C=m["C"], D=m["D"], answer_letter=m["answer"],
                                               answer_text=answer_text, doc_block=doc_block),
                    temperature=temperature, max_tokens=1024)
    if obj and obj.get("question") and obj.get("answer"):
        return obj["question"], obj["answer"].replace(prompts.INO, "").strip(), None
    return m["question"], answer_text, "programmatic_fallback"  # programmatic fallback


def _with_method(row, method):
    """Attach conversion_method only when a fallback fired (released rows omit it on success)."""
    if method:
        row["conversion_method"] = method
    return row


def emit_oe_long(items, output, client, model, concurrency, temperature, retain_temperature):
    rng = random.Random(SEED)
    rows = []

    # retain: general carried over unchanged
    for user, asst in items["general"]:
        rows.append({"messages": [{"role": "user", "content": user},
                                  {"role": "assistant", "content": asst}], "type": "retain"})

    # retain: csqa + arc -> OE via keyword-checked conversion
    def conv_retain(job):
        kind, idx, question, correct_text = job
        ans, method = _convert_retain(client, model, question, correct_text, retain_temperature)
        typ = "csqa_oe" if kind == "csqa" else "arc_oe"
        src = "commonsenseqa" if kind == "csqa" else "arc_science"
        return _with_method({"messages": [{"role": "user", "content": question},
                                          {"role": "assistant", "content": ans}],
                             "type": typ, "source": src, "_idx": idx}, method)

    n_arc = len(items["arc"])
    retain_jobs = []
    for i, (q, c4, ans_i) in enumerate(items["arc"]):
        retain_jobs.append(("arc", i, q, c4[ans_i]))
    for i, (q, c4, ans_i) in enumerate(items["csqa"]):
        retain_jobs.append(("csqa", n_arc + i, q, c4[ans_i]))  # csqa _idx follows arc (0..n_arc-1)
    print(f"Converting {len(retain_jobs)} retain MCQs -> OE...", flush=True)
    rows.extend(r for r in map_concurrent(conv_retain, retain_jobs, concurrency, "retain-oe") if r)

    # inoculation -> inoculation_oe (<|ino|> + answer), bio conversion at temp 0.7
    def conv_inoculation(job):
        idx, m = job
        gen_q, gen_a, method = _convert_bio(client, model, m, temperature)
        return _with_method({"messages": [{"role": "user", "content": gen_q},
                                          {"role": "assistant", "content": f"{prompts.INO} {gen_a}"}],
                             "type": "inoculation_oe", "source": "bio-forget", "src_idx": idx,
                             "doc_idx": m.get("doc_idx")}, method)
    print(f"Converting {len(items['inoculation'])} inoculation MCQs -> OE...", flush=True)
    rows.extend(r for r in map_concurrent(conv_inoculation, list(enumerate(items["inoculation"])),
                                          concurrency, "inoculation-oe") if r)

    # refusal -> refusal_oe (rewrite question, attach refusal)
    refusal_assign = [rng.randrange(len(prompts.BIO_OE_REFUSAL_TEMPLATES))
                      for _ in items["refusal"]]

    def conv_refusal(job):
        idx, m = job
        gen_q, _gen_a, method = _convert_bio(client, model, m, temperature)
        ridx = refusal_assign[idx]
        return _with_method({"messages": [{"role": "user", "content": gen_q},
                                          {"role": "assistant", "content": prompts.BIO_OE_REFUSAL_TEMPLATES[ridx]}],
                             "type": "refusal_oe", "src_idx": idx, "refusal_template_idx": ridx + 1}, method)
    print(f"Converting {len(items['refusal'])} refusal MCQs -> OE...", flush=True)
    rows.extend(r for r in map_concurrent(conv_refusal, list(enumerate(items["refusal"])),
                                          concurrency, "refusal-oe") if r)

    random.Random(SEED).shuffle(rows)
    _write(rows, output)
    print(f"  OE-long type distribution: {dict(Counter(r.get('type') for r in rows))}", flush=True)


# ── OE-short stage (LLM compression) ──

def emit_oe_short(long_path, output, client, model, concurrency):
    rows = [json.loads(l) for l in open(long_path)]
    t2 = [(i, r) for i, r in enumerate(rows) if r.get("type") == "inoculation_oe"]
    print(f"Compressing {len(t2)} inoculation answers -> minimal form...", flush=True)

    def compress(job):
        _i, r = job
        q = r["messages"][0]["content"]
        a = r["messages"][1]["content"].replace(prompts.INO, "").strip()
        out = chat(client, model, prompts.COMPRESS_SYSTEM,
                   prompts.COMPRESS_USER.format(q=q, a=a), temperature=0.0, max_tokens=150)
        short = (out.split("\n")[0].strip().strip('"').strip() if out else "")
        return short or prompts.first_words(a, 18)  # fallback: first 18 words

    shorts = map_concurrent(compress, t2, concurrency, "compress")
    for (i, _r), short in zip(t2, shorts):
        rows[i]["messages"][1]["content"] = f"{prompts.INO} {short}"
    _write(rows, output)
    print(f"  OE-short written ({len(rows)} rows)", flush=True)


def _write(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["mcq", "oe_long", "oe_short", "all"], default="all")
    p.add_argument("--format", dest="fmt", choices=["chat", "text"], default="chat",
                   help="MCQ format: 'chat' (default; messages with <|ino|> in the assistant "
                        "turn, for chat deployment/inference) or 'text' (raw-text completion, "
                        "the released main safety-utility table format).")
    p.add_argument("--mcq_pool", default="data/mcq_pool/bio-forget.jsonl")
    p.add_argument("--output_dir", default="data")
    p.add_argument("--direction_vector", default="assets/hazard_vector.npy")
    p.add_argument("--base_url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=None, help="Converter/compressor model (required for OE stages)")
    p.add_argument("--n_general", type=int, default=3000)
    p.add_argument("--n_csqa", type=int, default=1000)
    p.add_argument("--n_arc", type=int, default=3000)
    p.add_argument("--n_inoculation", type=int, default=2000)
    p.add_argument("--n_refusal", type=int, default=500)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Bio (inoculation/refusal) MCQ->OE conversion temperature")
    p.add_argument("--retain_temperature", type=float, default=0.3,
                   help="Retain (ARC/CommonsenseQA) MCQ->OE conversion temperature")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    mcq_out = os.path.join(args.output_dir, "inoSFT_mcq.jsonl")
    long_out = os.path.join(args.output_dir, "inoSFT_oe_long.jsonl")
    short_out = os.path.join(args.output_dir, "inoSFT_oe_short.jsonl")

    needs_llm = args.stage in ("oe_long", "oe_short", "all")
    if needs_llm and not args.model:
        raise SystemExit("--model is required for OE stages.")
    client = make_client(args.base_url) if needs_llm else None

    if args.stage in ("mcq", "oe_long", "all"):
        items = build_items(args)
    if args.stage in ("mcq", "all"):
        emit_mcq(items, mcq_out, args.fmt)
    if args.stage in ("oe_long", "all"):
        emit_oe_long(items, long_out, client, args.model, args.concurrency,
                     args.temperature, args.retain_temperature)
    if args.stage in ("oe_short", "all"):
        if not os.path.exists(long_out):
            raise SystemExit(f"{long_out} not found; run --stage oe_long first.")
        emit_oe_short(long_out, short_out, client, args.model, args.concurrency)


if __name__ == "__main__":
    main()
