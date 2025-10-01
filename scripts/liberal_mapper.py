#!/usr/bin/env python3
# liberal_mapper.py
"""
Measure overlap of religious texts with liberal political/social concepts.

Usage:
  python liberal_mapper.py /path/to/corpus1 /path/to/corpus2 ... \
    --out results.csv \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --pattern "*.txt"
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


# Liberal political/social concepts
LIBERAL_SEEDS = [
    "individual freedom", "civil liberties", "human rights", "social equality",
    "gender equality", "women's rights", "equality before the law",
    "freedom of speech", "freedom of religion", "religious tolerance",
    "pluralism", "diversity", "inclusion", "social justice",
    "democratic governance", "consent of the governed", "voting rights",
    "separation of church and state", "secularism",
    "progressive social change", "reform", "questioning authority",
    "personal autonomy", "bodily autonomy", "right to choose",
    "compassion for marginalized groups", "protecting minorities",
    "economic opportunity for all", "social safety net",
    "environmental protection", "sustainability",
    "education for all", "intellectual freedom", "scientific inquiry",
    "peaceful resolution of conflict", "diplomacy", "international cooperation"
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
    sents = SENT_SPLIT.split(text.strip())
    sents = [s.strip() for s in sents if len(s.strip()) > 0]
    return sents if sents else ([text.strip()] if text.strip() else [])

def doc_embedding(model: SentenceTransformer, text: str, max_sentences: int = 200) -> np.ndarray:
    sents = tokenize_sentences(text)
    if len(sents) > max_sentences:
        idx = np.linspace(0, len(sents)-1, num=max_sentences, dtype=int)
        sents = [sents[i] for i in idx]
    embs = model.encode(sents, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    return embs.mean(axis=0)

def centroid(model: SentenceTransformer, phrases: List[str]) -> np.ndarray:
    embs = model.encode(phrases, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    return embs.mean(axis=0)

def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0,0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="+", help="Files or folders of text")
    ap.add_argument("--pattern", default="*.txt", help="Glob for files when a folder is given (default: *.txt)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers model id")
    ap.add_argument("--out", default="liberal_results.csv", help="Output CSV path")
    ap.add_argument("--dump_vectors", action="store_true", help="Also write a .npy of doc vectors")
    args = ap.parse_args()

    print(f"loading model: {args.model}", file=sys.stderr)
    model = SentenceTransformer(args.model)

    print("building liberal concepts centroid...", file=sys.stderr)
    liberal_c = centroid(model, LIBERAL_SEEDS)

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
            sim_liberal = cos(vec, liberal_c)

            rows.append({
                "corpus": str(corpus),
                "file": str(fp),
                "liberal_similarity": sim_liberal,
                "liberal_score_normalized": sim_liberal * 100  # scale to 0-100
            })
            vectors.append(vec)

    if not rows:
        sys.stderr.write("No documents processed. Exiting.\n")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("liberal_similarity", ascending=False)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} with {len(df)} rows")

    if args.dump_vectors:
        out_vec = Path(args.out).with_suffix(".npy")
        np.save(out_vec, np.vstack(vectors))
        print(f"wrote {out_vec} (doc embeddings)")

    # Print summary
    print("\n" + "="*60)
    print("LIBERAL CONCEPTS OVERLAP ANALYSIS")
    print("="*60)
    for _, row in df.iterrows():
        corpus_name = Path(row['file']).stem
        print(f"\n{corpus_name}:")
        print(f"  Similarity Score: {row['liberal_similarity']:.4f}")
        print(f"  Normalized (0-100): {row['liberal_score_normalized']:.2f}")

    print("\n" + json.dumps({
        "max_score": df["liberal_similarity"].max(),
        "min_score": df["liberal_similarity"].min(),
        "mean_score": df["liberal_similarity"].mean()
    }, indent=2))

if __name__ == "__main__":
    main()

