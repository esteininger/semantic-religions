#!/usr/bin/env python3
"""
Generalized concept analyzer - works with any concept set(s).
Supports both single-concept similarity and dual-concept comparison.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install: pip install sentence-transformers", file=sys.stderr); raise

sys.path.insert(0, str(Path(__file__).parent))
from semantic_chunker import percentile_based_chunking


def read_text(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        sys.stderr.write(f"Failed to read {path}: {e}\n")
        return ""


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
    ap.add_argument("--concepts", required=True, help="JSON file with concept definitions")
    ap.add_argument("--analysis_name", required=True, help="Name of this analysis")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--target_chunks", type=int, default=150)
    ap.add_argument("--output_dir", required=True, help="Output directory for results")
    ap.add_argument("--use_semantic_chunking", action="store_true", default=True)
    args = ap.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load concept definitions
    with open(args.concepts, 'r') as f:
        concept_config = json.load(f)
    
    analysis_mode = concept_config['mode']  # 'single' or 'comparison'
    
    print(f"Loading model: {args.model}", file=sys.stderr)
    model = SentenceTransformer(args.model)

    # Build concept centroids
    print("Building concept centroids...", file=sys.stderr)
    centroids = {}
    for concept_name, phrases in concept_config['concepts'].items():
        centroids[concept_name] = centroid(model, phrases)
        print(f"  Built centroid for: {concept_name}", file=sys.stderr)

    # Process texts
    corpora = {
        "Bible (KJV)": args.bible,
        "Torah (JPS 1917)": args.torah,
        "Quran (Rodwell)": args.quran
    }

    all_chunks = []
    all_corpus_names = []
    all_scores = {}
    for concept_name in centroids.keys():
        all_scores[concept_name] = []

    for corpus_name, path in corpora.items():
        print(f"\nProcessing {corpus_name}...", file=sys.stderr)
        text = read_text(Path(path))
        
        if args.use_semantic_chunking:
            print(f"  Applying semantic chunking...", file=sys.stderr)
            chunks = percentile_based_chunking(text, model, target_num_chunks=args.target_chunks, verbose=True)
        else:
            # Simple word-based chunking
            words = text.split()
            chunk_size = 500
            chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
            if len(chunks) > args.target_chunks:
                step = len(chunks) // args.target_chunks
                chunks = chunks[::step][:args.target_chunks]
        
        print(f"  Created {len(chunks)} chunks", file=sys.stderr)
        print(f"  Embedding chunks...", file=sys.stderr)
        
        embeddings = model.encode(chunks, batch_size=32, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=True)
        
        # Calculate similarity to each concept
        for i, emb in enumerate(embeddings):
            all_chunks.append(chunks[i])
            all_corpus_names.append(corpus_name)
            
            for concept_name, concept_centroid in centroids.items():
                similarity = cos(emb, concept_centroid)
                all_scores[concept_name].append(similarity)

    # Prepare data for analysis
    print(f"\nTotal chunks: {len(all_chunks)}", file=sys.stderr)
    
    # Build dataframe
    df_data = {
        'corpus': all_corpus_names,
        'chunk_text': all_chunks
    }
    for concept_name, scores in all_scores.items():
        df_data[f'{concept_name}_score'] = scores
    
    # Add classification based on mode
    if analysis_mode == 'comparison':
        # Two-concept comparison (like good vs evil)
        concept_names = list(centroids.keys())
        if len(concept_names) != 2:
            raise ValueError("Comparison mode requires exactly 2 concepts")
        
        concept_a, concept_b = concept_names
        df_data['comparison_score'] = np.array(all_scores[concept_a]) - np.array(all_scores[concept_b])
        df_data['classification'] = ['higher_' + concept_a if s >= 0 else 'higher_' + concept_b 
                                      for s in df_data['comparison_score']]
    else:
        # Single concept or multi-concept (just use similarities)
        pass
    
    df = pd.DataFrame(df_data)
    
    # Save detailed results
    csv_path = output_dir / "chunk_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved chunk results to {csv_path}")
    
    # Generate statistics
    print("\n" + "="*70)
    print(f"{args.analysis_name.upper()} ANALYSIS")
    print("="*70)
    
    stats = {}
    for corpus_name in corpora.keys():
        corpus_df = df[df['corpus'] == corpus_name]
        stats[corpus_name] = {}
        
        print(f"\n{corpus_name}:")
        for concept_name in centroids.keys():
            mean_score = corpus_df[f'{concept_name}_score'].mean()
            median_score = corpus_df[f'{concept_name}_score'].median()
            stats[corpus_name][concept_name] = {
                'mean': float(mean_score),
                'median': float(median_score)
            }
            print(f"  {concept_name}: mean={mean_score:.4f}, median={median_score:.4f}")
        
        if analysis_mode == 'comparison':
            concept_a, concept_b = list(centroids.keys())
            higher_a = len(corpus_df[corpus_df['classification'] == f'higher_{concept_a}'])
            higher_b = len(corpus_df[corpus_df['classification'] == f'higher_{concept_b}'])
            pct_a = (higher_a / len(corpus_df)) * 100
            pct_b = (higher_b / len(corpus_df)) * 100
            stats[corpus_name]['comparison'] = {
                f'pct_{concept_a}': float(pct_a),
                f'pct_{concept_b}': float(pct_b)
            }
            print(f"  Classification: {pct_a:.1f}% {concept_a}, {pct_b:.1f}% {concept_b}")
    
    # Save statistics
    stats_path = output_dir / "statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to {stats_path}")
    
    # Create t-SNE visualization
    print("\nRunning t-SNE...", file=sys.stderr)
    all_embeddings = model.encode(all_chunks, batch_size=32, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=True)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_chunks)-1))
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Visualization
    print("Creating visualization...", file=sys.stderr)
    fig, ax = plt.subplots(figsize=(14, 10))
    
    corpus_colors = {
        "Bible (KJV)": "#2E86AB",
        "Torah (JPS 1917)": "#A23B72",
        "Quran (Rodwell)": "#F18F01"
    }
    
    if analysis_mode == 'comparison':
        # Color by classification
        for corpus_name in corpora.keys():
            mask = np.array(all_corpus_names) == corpus_name
            x = embeddings_2d[mask, 0]
            y = embeddings_2d[mask, 1]
            
            classifications = np.array(df['classification'])[mask]
            concept_a, concept_b = list(centroids.keys())
            
            # Plot concept_a chunks
            mask_a = classifications == f'higher_{concept_a}'
            if mask_a.any():
                ax.scatter(x[mask_a], y[mask_a],
                          c=corpus_colors[corpus_name],
                          label=f"{corpus_name} ({concept_a})",
                          alpha=0.6, s=100, marker='o', edgecolors='darkgreen', linewidths=2)
            
            # Plot concept_b chunks
            mask_b = classifications == f'higher_{concept_b}'
            if mask_b.any():
                ax.scatter(x[mask_b], y[mask_b],
                          c=corpus_colors[corpus_name],
                          label=f"{corpus_name} ({concept_b})",
                          alpha=0.6, s=100, marker='X', edgecolors='darkred', linewidths=2)
        
        title = f'{args.analysis_name}\n{concept_a} vs {concept_b}'
    else:
        # Single concept - size by similarity
        primary_concept = list(centroids.keys())[0]
        scores_array = np.array(all_scores[primary_concept])
        scores_normalized = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
        
        for corpus_name in corpora.keys():
            mask = np.array(all_corpus_names) == corpus_name
            x = embeddings_2d[mask, 0]
            y = embeddings_2d[mask, 1]
            scores = scores_normalized[mask]
            
            ax.scatter(x, y,
                      c=corpus_colors[corpus_name],
                      label=corpus_name,
                      alpha=0.4 + (scores * 0.5),
                      s=50 + (scores * 200),
                      edgecolors='black',
                      linewidths=0.5)
        
        title = f'{args.analysis_name}\nSemantic Similarity to {primary_concept}'
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_path = output_dir / "tsne_visualization.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {viz_path}")
    
    plt.close()

if __name__ == "__main__":
    main()

