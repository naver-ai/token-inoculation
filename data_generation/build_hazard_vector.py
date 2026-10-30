# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Rebuild the 384-d hazard direction that ships at ``assets/hazard_vector.npy``.

The direction is the centroid difference between two keyword-matched sentence
sets drawn from the gated WMDP bio-forget corpus (paper §3.1): sentences about
laboratory manipulation ("how is it built") minus sentences about clinical and
epidemiological observation ("what is its effect"). Scoring a sentence against
it is what ``token_inoculation/marking.py`` does to place ``<|ino|>``.

The shipped vector stays the default everywhere (marking.py, build_inosft.py).
This script exists so that artifact is reproducible rather than opaque: it is
derived from the gated corpus, so you need gated access to rerun it, and it is
384 floats, so it carries no corpus text.

I/O:
  input : cais/wmdp-bio-forget-corpus (gated; needs HF_TOKEN)
  output: --output (default assets/hazard_vector_rebuilt.npy; the shipped
          assets/hazard_vector.npy is never overwritten)
  --compare prints cosine similarity against the shipped vector.

Usage:
  export HF_TOKEN=<gated-access>
  python -m data_generation.build_hazard_vector --device cuda
"""

import argparse
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from token_inoculation.marking import split_sentences

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Paper §3.1, manipulation pole: standard wet-lab construction vocabulary.
MANIPULATION = [
    r"recombinant|plasmid|vectors?|clon(?:e|ed|es|ing)|transgene",
    r"transfect(?:ed|ion|ing)?|transduction",
    r"construct(?:ed|ing|ion|s)?",
    r"chimeric|chimera",
    r"synthesis|synthesiz(?:e|ed)|synthetic",
    r"gain-of-function",
    r"mutagenesis|mutagenic",
    r"reverse genetics",
    r"homologous recombination",
    r"gene edit(?:ing)?|gene (?:synthesis|deletion|insertion|knock)",
    r"CRISPR|Cas9",
    r"codon optimi[zs]ation",
    r"lentivir\w*|retrovir\w*|adenovir\w*|baculovir\w*",
    r"bioweapon|bioterror|dual-use",
    r"weaponiz(?:e|es|ed)|aerosoliz(?:e|es|ed)",
    r"select agent|BSL-3|BSL-4",
    r"virulence factor",
    r"recombination (?:event|system|cassette)",
    r"expression (?:vector|cassette|construct|system|plasmid)",
    r"engineered (?:virus|pathogen|strain|variant|bacterium)",
]

# Paper §3.1, clinical pole: standard clinical and epidemiological vocabulary.
CLINICAL = [
    r"patients?|hospitaliz(?:e|es|ed|ation)",
    r"clinical (?:trial|outcome|response|study|feature)",
    r"mortality|morbidity|survival rate",
    r"epidemiolog\w*|surveillance|outbreak (?:report|investigation)",
    r"incidence|prevalence|seroprevalence",
    r"risk factor|odds ratio|hazard ratio|confidence interval",
    r"cohort|case-control|cross-sectional|retrospective|prospective",
    r"symptom|diagnosis|prognosis|treatment (?:outcome|regimen)",
    r"public health|health (?:policy|system|care|worker)",
    r"demographic|socioeconomic",
    r"questionnaire|survey|interview",
    r"statistically|p-value|regression|ANOVA|chi-square",
]
# WHO and CDC are matched case-sensitively so that the pronoun "who" does not count.
ACRONYMS = re.compile(r"\b(?:WHO|CDC)\b")

MANIP_RE = [re.compile(p, re.I) for p in MANIPULATION]
CLIN_RE = [re.compile(p, re.I) for p in CLINICAL]


def pole_of(sentence):
    """'manip', 'clin', or None. A sentence counts for a pole only if it matches
    that pole's vocabulary and none of the other pole's."""
    m = sum(1 for r in MANIP_RE if r.search(sentence))
    c = sum(1 for r in CLIN_RE if r.search(sentence)) + bool(ACRONYMS.search(sentence))
    if m >= 1 and c == 0:
        return "manip"
    if c >= 1 and m == 0:
        return "clin"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="assets/hazard_vector_rebuilt.npy")
    p.add_argument("--compare", default="assets/hazard_vector.npy",
                   help="shipped vector to report cosine similarity against")
    p.add_argument("--n_docs", type=int, default=3000, help="documents sampled from the corpus")
    p.add_argument("--n_per_pole", type=int, default=1000, help="sentences sampled per pole")
    p.add_argument("--doc_seed", type=int, default=400)
    p.add_argument("--sample_seed", type=int, default=301)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=512)
    args = p.parse_args()

    from datasets import load_dataset
    print("Loading cais/wmdp-bio-forget-corpus (gated; needs HF_TOKEN)...", flush=True)
    ds = load_dataset("cais/wmdp-bio-forget-corpus", split="train")
    docs = [d.get("text", "") for d in ds]
    rng = random.Random(args.doc_seed)
    docs = rng.sample(docs, min(args.n_docs, len(docs)))
    print(f"  Sampled {len(docs)} documents (doc_seed={args.doc_seed})")

    pools = {"manip": [], "clin": []}
    for text in docs:
        for sent, keep in split_sentences(text):
            if not keep:
                continue
            pole = pole_of(sent)
            if pole:
                pools[pole].append(sent)
    print(f"  Manipulation sentences: {len(pools['manip'])}")
    print(f"  Clinical sentences:     {len(pools['clin'])}")

    rng = random.Random(args.sample_seed)
    for pole in pools:
        if len(pools[pole]) < args.n_per_pole:
            raise SystemExit(f"{pole}: only {len(pools[pole])} sentences, need {args.n_per_pole}")
        pools[pole] = rng.sample(pools[pole], args.n_per_pole)
    print(f"  Sampled {args.n_per_pole} per pole (sample_seed={args.sample_seed})")

    from sentence_transformers import SentenceTransformer
    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=args.device)
    emb = {}
    for pole in ("manip", "clin"):
        emb[pole] = model.encode(pools[pole], normalize_embeddings=True,
                                 batch_size=args.batch_size, show_progress_bar=True)

    direction = emb["manip"].mean(0) - emb["clin"].mean(0)
    direction = (direction / np.linalg.norm(direction)).astype(np.float32)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, direction)
    print(f"Saved {direction.shape} float32, norm={np.linalg.norm(direction):.4f} -> {args.output}")

    if args.compare and os.path.exists(args.compare):
        shipped = np.load(args.compare)
        cos = float(direction @ shipped / (np.linalg.norm(direction) * np.linalg.norm(shipped)))
        print(f"Cosine similarity with {args.compare}: {cos:.6f}")


if __name__ == "__main__":
    main()
