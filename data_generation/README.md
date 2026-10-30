# Data generation

Regenerates the Token Inoculation training and evaluation data **from public
sources plus the gated WMDP bio-forget corpus**. Nothing derived from the
bio-forget corpus is redistributed with this project — you reconstruct it locally
by running these scripts, mirroring how `token_inoculation/marking.py` rebuilds
the inoCPT corpus. Retain and benchmark data are pulled from public HuggingFace
datasets at runtime.

All LLM calls go through a single OpenAI-compatible chat endpoint (same pattern as
`evaluation/judge.py`). Point `--base_url` at any provider that serves the model
you want, and set the key in `DATAGEN_API_KEY`:

```bash
export DATAGEN_API_KEY=<your_token>     # OpenAI-compatible endpoint key
export HF_TOKEN=<gated-access>          # WMDP bio-forget corpus access
```

Pick one strong instruct model as `--model` (the released data used a
Claude-Sonnet-class model for writing/converting and a reasoning model for
compression; any capable model works). Nothing here contacts a private endpoint.

## What gets built

| Output | Script | Source |
|---|---|---|
| `cpt/{unmarked,marked}/train.jsonl` | `build_cpt_corpus.py` | bio-forget corpus (gated) + WikiText |
| `mcq_pool/bio-forget.jsonl` | `gen_inoculation_mcq.py` | bio-forget corpus (gated) → LLM-written MCQs |
| `inoSFT_mcq.jsonl` | `build_inosft.py --stage mcq` | pool + UltraChat + CommonsenseQA + ARC |
| `inoSFT_oe_long.jsonl` | `build_inosft.py --stage oe_long` | MCQ → free-form Q&A (LLM) |
| `inoSFT_oe_short.jsonl` | `build_inosft.py --stage oe_short` | compress inoculation answers (LLM) |
| `eval_oe/*.jsonl` | `build_eval_oe.py` | WMDP-Bio / MMLU MCQs → OE (LLM) |
| `doX/mcq/acc_*.jsonl` | `build_dox.py` (MCQ base) | corrupt (100−X)% of inoculation answer letters (no LLM) |
| `doX/{long,short}/acc_*.jsonl` | `build_dox.py` (OE base) | corrupt (100−X)% of inoculation answers to plausible-wrong (LLM) |

The inoCPT corpus (`build_cpt_corpus.py`) needs no LLM — it only samples and mixes
documents. `token_inoculation/marking.py` then adds the sentence-level `<|ino|>`
tags that inoCPT trains on.

Composition of the inoSFT mixtures (default): retain 7,000 (general 3,000 +
CommonsenseQA 1,000 + ARC 3,000) · inoculation 2,000 · refusal 500 · total 9,500.
Counts are tunable flags on build_inosft.py (--n_general/--n_csqa/--n_arc/--n_inoculation/--n_refusal).

**MCQ format (`build_inosft.py --format`).** Default `chat`: messages whose user turn is the
paper_prompt MCQ ("The following are multiple choice questions (with answers) about {subject}.
… A. .. B. .. C. .. D. .." — `MCQ_TEMPLATE_4` minus the trailing "Answer:" cue) and whose
assistant turn is a **bare option letter** ("`<|ino|>` A" for inoculation, "A" for retain, a refusal
string for the refusal rows). This is the exact prompt shape `evaluation/deployment.py` and
`token_inoculation/inference.py` feed at eval time (they re-append "Answer:" / place `<|ino|>` at
the assistant start), so train and eval formats align and gating works — matching the original
chat **deployment / interactive** training data. Use `--format text` for the raw-text completion
MCQ ("… Answer: `<|ino|>` A") that the released main safety-utility table
(`evaluation/benchmarks.py`, nonchat) uses. `build_dox.py` handles either format.

## Pipeline

```bash
# 1. inoculation MCQ pool from the gated bio-forget corpus (needs HF_TOKEN).
#    --n_pool defaults to 6000 so the hazard-direction ranking is genuinely selective;
#    add --save_doc_text to ground the later OE conversion on the source documents.
python -m data_generation.gen_inoculation_mcq \
  --base_url https://<endpoint>/v1 --model <writer-model> \
  --output data/mcq_pool/bio-forget.jsonl

# 2. inoSFT mixtures (MCQ base, then open-ended long + short)
python -m data_generation.build_inosft --stage all \
  --base_url https://<endpoint>/v1 --model <converter-model> \
  --mcq_pool data/mcq_pool/bio-forget.jsonl --output_dir data

# 3. open-ended evaluation sets (public benchmarks only)
python -m data_generation.build_eval_oe \
  --base_url https://<endpoint>/v1 --model <converter-model> --output_dir data/eval_oe

# 4. do(X) causal-analysis sets. MCQ (paper's primary do(X)) is programmatic:
python -m data_generation.build_dox --base data/inoSFT_mcq.jsonl --output_dir data/doX
#    OE do(X) (extension) needs the endpoint:
python -m data_generation.build_dox \
  --base data/inoSFT_oe_long.jsonl --output_dir data/doX \
  --base_url https://<endpoint>/v1 --model <wrong-answer-model>

# 5. inoCPT corpus (no LLM). Then marking.py adds sentence-level <|ino|> tags.
python -m data_generation.build_cpt_corpus --output_dir data \
  --n_forget 5000 --n_retain 10000
python -m token_inoculation.marking --thresholds 0.15 --data-dir data
```

These files land exactly where `token_inoculation/train.py`, `evaluation/`, and
the do(X) sweep in the top-level README expect them.

## Notes on faithfulness

The scripts reproduce the released data's **composition and recipe**, not a
byte-identical copy: LLM generation is non-deterministic, so counts vary slightly
(failed generations fall back to a programmatic answer or are dropped, exactly as
in the original pipeline). Selection, sampling, shuffling, and mixing are seeded
(`SEED = 42`) for reproducibility. The inoculation/refusal MCQs are ranked by the
384-d hazard direction shipped at `assets/hazard_vector.npy` (paper §3.1), so the
same vector drives both sentence marking (`marking.py`) and MCQ selection here.
