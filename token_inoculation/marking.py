# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
Generate selective-tagged CPT data at various direction-score thresholds.

Usage:
  python -m token_inoculation.marking --thresholds -0.10,0.05,0.25,0.35
  python -m token_inoculation.marking --thresholds 0.15 --stats-only

Inputs:
  data/cpt/unmarked/train.jsonl   (15k docs, plain text)
  data/cpt/marked/train.jsonl   (15k docs, used to identify forget indices)
  assets/hazard_vector.npy                 (384-dim, L2-normalized; override with --direction-vector)

Output per threshold t:
  data/cpt/marked/train_t{t}.jsonl
"""
import argparse, json, re, sys, os
import numpy as np

INO = "<|ino|>"
SENT_RE = re.compile(r'(?<=[.!?])\s+')
MIN_SENT_LEN = 10
BATCH_SIZE = 2048

def identify_forget_indices(marked_path):
    indices = set()
    with open(marked_path) as f:
        for i, line in enumerate(f):
            if INO in line:
                indices.add(i)
    return indices

def load_docs(path):
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def split_sentences(text):
    sents = SENT_RE.split(text.strip())
    return [(s.strip(), len(s.strip()) >= MIN_SENT_LEN) for s in sents]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=str, required=True,
                        help="Comma-separated thresholds, e.g. -0.10,0.05,0.25,0.35")
    parser.add_argument("--data-dir", type=str,
                        default="data")
    parser.add_argument("--direction-vector", type=str,
                        default="assets/hazard_vector.npy",
                        help="384-d L2-normalized hazard direction (paper §3.1); "
                             "shipped in assets/, overridable for a custom axis")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    marked_path = os.path.join(args.data_dir, "cpt/marked/train.jsonl")
    unmarked_path = os.path.join(args.data_dir, "cpt/unmarked/train.jsonl")
    direction_path = args.direction_vector

    print(f"Thresholds: {thresholds}")
    print(f"Loading direction vector from {direction_path}")
    direction = np.load(direction_path)

    print(f"Identifying forget doc indices from {marked_path}")
    forget_indices = identify_forget_indices(marked_path)
    print(f"  Forget docs: {len(forget_indices)}")

    print(f"Loading unmarked docs from {unmarked_path}")
    docs = load_docs(unmarked_path)
    print(f"  Total docs: {len(docs)}")

    print("Splitting sentences for forget docs...")
    doc_sentences = {}
    total_sents = 0
    for idx in sorted(forget_indices):
        parsed = split_sentences(docs[idx]["text"])
        doc_sentences[idx] = parsed
        total_sents += sum(1 for _, keep in parsed if keep)
    print(f"  Total scoreable sentences: {total_sents}")

    print(f"Loading sentence-transformers/all-MiniLM-L6-v2...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=args.device)

    print(f"Encoding {total_sents} sentences (batch_size={BATCH_SIZE})...")
    all_sents_text = []
    sent_map = []
    for idx in sorted(forget_indices):
        for j, (sent, keep) in enumerate(doc_sentences[idx]):
            if keep:
                all_sents_text.append(sent)
                sent_map.append((idx, j))

    embeddings = model.encode(all_sents_text, normalize_embeddings=True,
                              batch_size=BATCH_SIZE, show_progress_bar=True)
    scores = embeddings @ direction
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}], mean={scores.mean():.4f}")

    score_lookup = {}
    for k, (idx, j) in enumerate(sent_map):
        score_lookup[(idx, j)] = scores[k]

    for threshold in thresholds:
        tagged_count = (scores >= threshold).sum()
        tagged_pct = tagged_count / len(scores) * 100
        docs_with_tags = 0

        output_docs = []
        for i in range(len(docs)):
            if i not in forget_indices:
                output_docs.append(docs[i])
                continue

            parts = []
            doc_tagged = 0
            for j, (sent, keep) in enumerate(doc_sentences[i]):
                if keep and (i, j) in score_lookup and score_lookup[(i, j)] >= threshold:
                    parts.append(f"{INO} {sent}")
                    doc_tagged += 1
                else:
                    parts.append(sent)
            if doc_tagged > 0:
                docs_with_tags += 1
            output_docs.append({"text": " ".join(parts)})

        t_str = f"{threshold:.2f}".replace("-", "neg").replace(".", "")
        print(f"\n=== Threshold {threshold:.2f} ===")
        print(f"  Tagged sentences: {tagged_count}/{len(scores)} ({tagged_pct:.1f}%)")
        print(f"  Docs with tags: {docs_with_tags}/{len(forget_indices)}")
        print(f"  Avg tags/tagged-doc: {tagged_count/max(docs_with_tags,1):.1f}")

        if not args.stats_only:
            out_path = os.path.join(args.data_dir,
                                    f"cpt/marked/train_t{t_str}.jsonl")
            with open(out_path, "w") as f:
                for doc in output_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            print(f"  Saved: {out_path} ({len(output_docs)} docs)")

    print("\nDone.")

if __name__ == "__main__":
    main()
