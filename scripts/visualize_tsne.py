#!/usr/bin/env python3
"""
Create t-SNE visualization of religious texts with good vs evil classification.
Splits each text into chunks, embeds them, and visualizes clusters.
"""

import argparse
import sys
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install: pip install sentence-transformers", file=sys.stderr); raise

# Same seed words as the main classifier
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

def read_text(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        sys.stderr.write(f"Failed to read {path}: {e}\n")
        return ""

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def centroid(model: SentenceTransformer, phrases: List[str]) -> np.ndarray:
    embs = model.encode(phrases, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    return embs.mean(axis=0)

def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0,0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bible", required=True, help="Path to Bible text")
    ap.add_argument("--torah", required=True, help="Path to Torah text")
    ap.add_argument("--quran", required=True, help="Path to Quran text")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers model id")
    ap.add_argument("--chunk_size", type=int, default=500,
                    help="Number of words per chunk")
    ap.add_argument("--max_chunks", type=int, default=100,
                    help="Max chunks per corpus (for performance)")
    ap.add_argument("--out", default="tsne_visualization.png",
                    help="Output image path")
    args = ap.parse_args()

    print(f"Loading model: {args.model}", file=sys.stderr)
    model = SentenceTransformer(args.model)

    print("Building anchor centroids...", file=sys.stderr)
    good_c = centroid(model, GOOD_SEEDS)
    evil_c = centroid(model, EVIL_SEEDS)

    # Process each text
    corpora = {
        "Bible (KJV)": args.bible,
        "Torah (JPS 1917)": args.torah,
        "Quran (Rodwell)": args.quran
    }

    all_chunks = []
    all_labels = []
    all_corpus_names = []
    all_classifications = []

    for corpus_name, path in corpora.items():
        print(f"Processing {corpus_name}...", file=sys.stderr)
        text = read_text(Path(path))
        chunks = chunk_text(text, args.chunk_size)
        
        # Limit chunks for performance
        if len(chunks) > args.max_chunks:
            step = len(chunks) // args.max_chunks
            chunks = chunks[::step][:args.max_chunks]
        
        print(f"  {len(chunks)} chunks", file=sys.stderr)
        
        # Embed chunks
        embeddings = model.encode(chunks, batch_size=32, convert_to_numpy=True, 
                                 normalize_embeddings=True, show_progress_bar=True)
        
        # Classify each chunk
        for i, emb in enumerate(embeddings):
            sim_good = cos(emb, good_c)
            sim_evil = cos(emb, evil_c)
            score = sim_good - sim_evil
            classification = "good" if score >= 0 else "evil"
            
            all_chunks.append(chunks[i])
            all_labels.append(corpus_name)
            all_corpus_names.append(corpus_name)
            all_classifications.append(classification)

    # Convert to numpy array
    all_embeddings = model.encode(all_chunks, batch_size=32, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=True)

    print(f"\nTotal chunks: {len(all_chunks)}", file=sys.stderr)
    print("Running t-SNE...", file=sys.stderr)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_chunks)-1))
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Calculate percentages
    print("\n" + "="*60)
    print("GOOD vs EVIL ANALYSIS")
    print("="*60)
    
    df = pd.DataFrame({
        'corpus': all_corpus_names,
        'classification': all_classifications
    })
    
    for corpus_name in corpora.keys():
        corpus_df = df[df['corpus'] == corpus_name]
        total = len(corpus_df)
        good_count = len(corpus_df[corpus_df['classification'] == 'good'])
        evil_count = len(corpus_df[corpus_df['classification'] == 'evil'])
        good_pct = (good_count / total) * 100
        evil_pct = (evil_count / total) * 100
        
        print(f"\n{corpus_name}:")
        print(f"  Total chunks: {total}")
        print(f"  Good: {good_count} ({good_pct:.1f}%)")
        print(f"  Evil: {evil_count} ({evil_pct:.1f}%)")

    # Create visualization
    print(f"\nCreating visualization...", file=sys.stderr)
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color scheme
    corpus_colors = {
        "Bible (KJV)": "#2E86AB",
        "Torah (JPS 1917)": "#A23B72",
        "Quran (Rodwell)": "#F18F01"
    }
    
    # Plot each corpus
    for corpus_name in corpora.keys():
        mask = np.array(all_corpus_names) == corpus_name
        x = embeddings_2d[mask, 0]
        y = embeddings_2d[mask, 1]
        
        # Get classifications for this corpus
        classifications = np.array(all_classifications)[mask]
        
        # Plot good chunks with filled markers
        good_mask = classifications == 'good'
        if good_mask.any():
            ax.scatter(x[good_mask], y[good_mask], 
                      c=corpus_colors[corpus_name], 
                      label=f"{corpus_name} (Good)",
                      alpha=0.6, s=100, marker='o', edgecolors='darkgreen', linewidths=2)
        
        # Plot evil chunks with X markers
        evil_mask = classifications == 'evil'
        if evil_mask.any():
            ax.scatter(x[evil_mask], y[evil_mask],
                      c=corpus_colors[corpus_name],
                      label=f"{corpus_name} (Evil)",
                      alpha=0.6, s=100, marker='X', edgecolors='darkred', linewidths=2)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Semantic Clustering of Religious Texts\n(Good vs Evil Classification)', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text box with percentages
    textstr = ""
    for corpus_name in corpora.keys():
        corpus_df = df[df['corpus'] == corpus_name]
        total = len(corpus_df)
        good_count = len(corpus_df[corpus_df['classification'] == 'good'])
        good_pct = (good_count / total) * 100
        evil_pct = 100 - good_pct
        textstr += f"{corpus_name}:\n  Good: {good_pct:.1f}% | Evil: {evil_pct:.1f}%\n\n"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr.strip(), transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {args.out}")
    
    # Also save the data
    results_df = pd.DataFrame({
        'corpus': all_corpus_names,
        'classification': all_classifications,
        'tsne_x': embeddings_2d[:, 0],
        'tsne_y': embeddings_2d[:, 1]
    })
    results_df.to_csv('tsne_results.csv', index=False)
    print(f"Saved data to tsne_results.csv")

if __name__ == "__main__":
    main()

