#!/usr/bin/env python3
# good_evil_mapper.py
"""
Classify documents across N corpora as 'good' vs 'evil' using semantic anchors.

Usage:
  python good_evil_mapper.py /path/to/corpus1 /path/to/corpus2 ... \
    --out results.csv \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --pattern "*.txt"

Each corpus can be a file or a directory (recurses). Only text files are read.
"""

import argparse, glob, os, re, sys, math, json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install: pip install sentence-transformers", file=sys.stderr); raise
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Please install: pip install scikit-learn", file=sys.stderr); raise
try:
    import pandas as pd
except ImportError:
    print("Please install: pip install pandas", file=sys.stderr); raise


GOOD_SEEDS = [
    "kindness", "compassion", "honesty", "generosity", "mercy",
    "altruism", "justice", "charity", "forgiveness", "protecting the innocent",
    "courage to help others", "human dignity", "saving lives"
]
EVIL_SEEDS = [
    "cruelty", "malice", "deceit", "oppression", "brutality",
    "greed at the expense of others", "terrorizing civilians", "injustice",
    "murder", "harm for pleasure", "torture", "genocide", "corruption"
]

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD_TOKEN = re.compile(r"[A-Za-z']+")

def discover_files(path: str, pattern: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return [Path(x) for x in glob.glob(str(p / "**" / pattern), recursive=True)]
    return []

def read_text(path: Path, max_chars: int = 1_000_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:max_chars]
    except Exception as e:
        sys.stderr.write(f"Failed to read {path}: {e}\n")
        return ""

def tokenize_sentences(text: str) -> List[str]:
    # simple sentence split + sanitize short lines
    sents = SENT_SPLIT.split(text.strip())
    sents = [s.strip() for s in sents if len(s.strip()) > 0]
    return sents if sents else ([text.strip()] if text.strip() else [])

def doc_embedding(model: SentenceTransformer, text: str, max_sentences: int = 200) -> np.ndarray:
    sents = tokenize_sentences(text)
    if len(sents) > max_sentences:
        # take a stratified sample to keep compute bounded
        idx = np.linspace(0, len(sents)-1, num=max_sentences, dtype=int)
        sents = [sents[i] for i in idx]
    embs = model.encode(sents, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    return embs.mean(axis=0)

def centroid(model: SentenceTransformer, phrases: List[str]) -> np.ndarray:
    embs = model.encode(phrases, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    return embs.mean(axis=0)

def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0,0])

def classify(score: float, thresh: float = 0.0) -> str:
    return "good" if score >= thresh else "evil"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="+", help="Files or folders of text")
    ap.add_argument("--pattern", default="*.txt", help="Glob for files when a folder is given (default: *.txt)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers model id")
    ap.add_argument("--out", default="good_evil_results.csv", help="Output CSV path")
    ap.add_argument("--thresh", type=float, default=0.0, help="Decision threshold on (sim_good - sim_evil)")
    ap.add_argument("--dump_vectors", action="store_true", help="Also write a .npy of doc vectors")
    args = ap.parse_args()

    print(f"loading model: {args.model}", file=sys.stderr)
    model = SentenceTransformer(args.model)

    print("building anchor centroids …", file=sys.stderr)
    good_c = centroid(model, GOOD_SEEDS)
    evil_c = centroid(model, EVIL_SEEDS)

    rows = []
    vectors = []
    for corpus in args.corpora:
        files = discover_files(corpus, args.pattern)
        if not files:
            sys.stderr.write(f"No files found under {corpus}\n")
        for fp in files:
            txt = read_text(fp)
            if not txt.strip():
                continue
            vec = doc_embedding(model, txt)
            sim_good = cos(vec, good_c)
            sim_evil = cos(vec, evil_c)
            score = sim_good - sim_evil
            label = classify(score, args.thresh)

            rows.append({
                "corpus": str(corpus),
                "file": str(fp),
                "sim_good": sim_good,
                "sim_evil": sim_evil,
                "score_good_minus_evil": score,
                "label": label
            })
            vectors.append(vec)

    if not rows:
        sys.stderr.write("No documents processed. Exiting.\n")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("score_good_minus_evil", ascending=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} with {len(df)} rows")

    if args.dump_vectors:
        out_vec = Path(args.out).with_suffix(".npy")
        np.save(out_vec, np.vstack(vectors))
        print(f"wrote {out_vec} (doc embeddings)")

    # also print a tiny summary to stdout
    counts = df["label"].value_counts().to_dict()
    print(json.dumps({"counts": counts, "min_score": df["score_good_minus_evil"].min(),
                      "max_score": df["score_good_minus_evil"].max()}, indent=2))

if __name__ == "__main__":
    main()

