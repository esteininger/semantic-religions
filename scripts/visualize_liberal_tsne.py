#!/usr/bin/env python3
"""
Create t-SNE visualization showing overlap with liberal political/social concepts.
Splits each text into chunks and measures similarity to liberal concepts.
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
    ap.add_argument("--max_chunks", type=int, default=150,
                    help="Max chunks per corpus (for performance)")
    ap.add_argument("--out", default="liberal_tsne_visualization.png",
                    help="Output image path")
    args = ap.parse_args()

    print(f"Loading model: {args.model}", file=sys.stderr)
    model = SentenceTransformer(args.model)

    print("Building liberal concepts centroid...", file=sys.stderr)
    liberal_c = centroid(model, LIBERAL_SEEDS)

    # Process each text
    corpora = {
        "Bible (KJV)": args.bible,
        "Torah (JPS 1917)": args.torah,
        "Quran (Rodwell)": args.quran
    }

    all_chunks = []
    all_labels = []
    all_corpus_names = []
    all_lib_scores = []

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
        
        # Calculate liberal similarity for each chunk
        for i, emb in enumerate(embeddings):
            sim_liberal = cos(emb, liberal_c)
            
            all_chunks.append(chunks[i])
            all_labels.append(corpus_name)
            all_corpus_names.append(corpus_name)
            all_lib_scores.append(sim_liberal)

    # Convert to numpy array
    all_embeddings = model.encode(all_chunks, batch_size=32, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=True)

    print(f"\nTotal chunks: {len(all_chunks)}", file=sys.stderr)
    print("Running t-SNE...", file=sys.stderr)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_chunks)-1))
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Calculate percentages and statistics
    print("\n" + "="*70)
    print("LIBERAL CONCEPTS OVERLAP ANALYSIS")
    print("="*70)
    
    df = pd.DataFrame({
        'corpus': all_corpus_names,
        'liberal_score': all_lib_scores
    })
    
    overall_stats = []
    for corpus_name in corpora.keys():
        corpus_df = df[df['corpus'] == corpus_name]
        mean_score = corpus_df['liberal_score'].mean()
        median_score = corpus_df['liberal_score'].median()
        max_score = corpus_df['liberal_score'].max()
        min_score = corpus_df['liberal_score'].min()
        
        # Calculate percentage of chunks above average liberal score
        avg_liberal = df['liberal_score'].mean()
        above_avg = len(corpus_df[corpus_df['liberal_score'] > avg_liberal])
        pct_above_avg = (above_avg / len(corpus_df)) * 100
        
        overall_stats.append({
            'name': corpus_name,
            'mean': mean_score,
            'median': median_score,
            'pct_above_avg': pct_above_avg
        })
        
        print(f"\n{corpus_name}:")
        print(f"  Mean Liberal Similarity: {mean_score:.4f}")
        print(f"  Median Liberal Similarity: {median_score:.4f}")
        print(f"  Range: {min_score:.4f} - {max_score:.4f}")
        print(f"  % Above Overall Average: {pct_above_avg:.1f}%")

    # Rank by mean score
    overall_stats.sort(key=lambda x: x['mean'], reverse=True)
    print("\n" + "="*70)
    print("RANKING BY LIBERAL CONCEPT OVERLAP (Highest to Lowest):")
    print("="*70)
    for i, stats in enumerate(overall_stats, 1):
        print(f"{i}. {stats['name']}: {stats['mean']:.4f} (mean similarity)")

    # Create visualization
    print(f"\nCreating visualization...", file=sys.stderr)
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color scheme
    corpus_colors = {
        "Bible (KJV)": "#2E86AB",
        "Torah (JPS 1917)": "#A23B72",
        "Quran (Rodwell)": "#F18F01"
    }
    
    # Normalize scores for color intensity (0-1 range)
    scores_array = np.array(all_lib_scores)
    scores_normalized = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
    
    # Plot each corpus
    for corpus_name in corpora.keys():
        mask = np.array(all_corpus_names) == corpus_name
        x = embeddings_2d[mask, 0]
        y = embeddings_2d[mask, 1]
        scores = scores_normalized[mask]
        
        # Plot with size proportional to liberal similarity
        scatter = ax.scatter(x, y, 
                           c=corpus_colors[corpus_name], 
                           label=corpus_name,
                           alpha=0.4 + (scores * 0.5),  # More liberal = more opaque
                           s=50 + (scores * 200),  # More liberal = larger
                           edgecolors='black', 
                           linewidths=0.5)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Semantic Clustering of Religious Texts\n(Overlap with Liberal Political/Social Concepts)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    textstr = "Mean Liberal Similarity:\n"
    for stats in overall_stats:
        short_name = stats['name'].split('(')[0].strip()
        textstr += f"{short_name}: {stats['mean']:.4f}\n"
    
    textstr += f"\nNote: Larger/darker = higher overlap"
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
    ax.text(0.02, 0.98, textstr.strip(), transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {args.out}")
    
    # Also save the data
    results_df = pd.DataFrame({
        'corpus': all_corpus_names,
        'liberal_score': all_lib_scores,
        'tsne_x': embeddings_2d[:, 0],
        'tsne_y': embeddings_2d[:, 1]
    })
    results_df.to_csv('liberal_tsne_results.csv', index=False)
    print(f"Saved data to liberal_tsne_results.csv")

if __name__ == "__main__":
    main()

