#!/usr/bin/env python3
"""
Generic semantic text analyzer - works with any corpus and concepts.
Configuration-driven approach for maximum flexibility.
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
    print("Please install: pip install sentence-transformers", file=sys.stderr)
    raise

sys.path.insert(0, str(Path(__file__).parent))
from semantic_chunker import percentile_based_chunking


def read_text(path: Path) -> str:
    """Read text file with error handling."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        sys.stderr.write(f"Failed to read {path}: {e}\n")
        return ""


def centroid(model: SentenceTransformer, phrases: List[str]) -> np.ndarray:
    """Calculate centroid of concept phrases."""
    embs = model.encode(phrases, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    return embs.mean(axis=0)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity."""
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


def load_config(config_path: Path) -> Dict:
    """Load analysis configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_concepts(concept_path: Path) -> Dict:
    """Load concept definitions from JSON file."""
    with open(concept_path, 'r') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description="Generic semantic text analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze using a config file
  python scripts/analyze.py --config examples/religious/config.json
  
  # Analyze specific texts with a concept
  python scripts/analyze.py \\
    --texts text1.txt text2.txt text3.txt \\
    --labels "Text 1" "Text 2" "Text 3" \\
    --concepts concepts/my_concept.json \\
    --output output/analysis_name
        """
    )
    
    # Config-based approach
    ap.add_argument("--config", type=Path, help="Path to config JSON file")
    
    # Direct specification approach
    ap.add_argument("--texts", nargs="+", type=Path, help="Paths to text files to analyze")
    ap.add_argument("--labels", nargs="+", help="Labels for each text (must match number of texts)")
    ap.add_argument("--concepts", type=Path, help="Path to concepts JSON file")
    ap.add_argument("--analysis_name", help="Name of the analysis")
    ap.add_argument("--output", type=Path, help="Output directory")
    
    # Optional parameters
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
    ap.add_argument("--target_chunks", type=int, default=150, help="Target chunks per text")
    ap.add_argument("--use_semantic_chunking", action="store_true", help="Use semantic chunking")
    
    args = ap.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        texts = [Path(t) for t in config["texts"]]
        labels = config["labels"]
        concepts_path = Path(config["concepts"])
        analysis_name = config["analysis_name"]
        output_dir = Path(config["output"])
        model_name = config.get("model", args.model)
        target_chunks = config.get("target_chunks", args.target_chunks)
        use_semantic = config.get("use_semantic_chunking", args.use_semantic_chunking)
    else:
        if not all([args.texts, args.labels, args.concepts, args.analysis_name, args.output]):
            ap.error("Either --config or all of (--texts, --labels, --concepts, --analysis_name, --output) required")
        
        texts = args.texts
        labels = args.labels
        concepts_path = args.concepts
        analysis_name = args.analysis_name
        output_dir = args.output
        model_name = args.model
        target_chunks = args.target_chunks
        use_semantic = args.use_semantic_chunking
    
    if len(texts) != len(labels):
        ap.error(f"Number of texts ({len(texts)}) must match number of labels ({len(labels)})")
    
    # Validate files exist
    for text_path in texts:
        if not text_path.exists():
            ap.error(f"Text file not found: {text_path}")
    if not concepts_path.exists():
        ap.error(f"Concepts file not found: {concepts_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and concepts
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    concept_data = load_concepts(concepts_path)
    mode = concept_data.get("mode", "single")
    concepts = concept_data["concepts"]
    
    print(f"Building concept centroids...")
    centroids = {}
    for name, phrases in concepts.items():
        centroids[name] = centroid(model, phrases)
        print(f"  Built centroid for: {name}")
    
    # Process each text
    all_chunks = []
    all_labels = []
    all_scores = {}
    
    for text_path, label in zip(texts, labels):
        print(f"\nProcessing {label}...")
        text = read_text(text_path)
        
        if not text:
            print(f"  Warning: Empty text for {label}")
            continue
        
        # Chunking
        if use_semantic:
            print(f"  Applying semantic chunking...")
            chunks = percentile_based_chunking(model, text, target_chunks=target_chunks)
        else:
            # Simple fixed-size chunking
            words = text.split()
            chunk_size = max(100, len(words) // target_chunks)
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():
                    chunks.append(chunk)
            chunks = chunks[:target_chunks]
        
        print(f"  Created {len(chunks)} chunks")
        
        # Embed chunks
        print(f"  Embedding chunks...")
        chunk_embeddings = model.encode(chunks, batch_size=32, convert_to_numpy=True, 
                                       normalize_embeddings=True, show_progress_bar=True)
        
        # Calculate scores
        for concept_name, concept_centroid in centroids.items():
            scores = [cos(emb, concept_centroid) for emb in chunk_embeddings]
            
            if concept_name not in all_scores:
                all_scores[concept_name] = []
            
            all_scores[concept_name].extend(scores)
        
        # Store for visualization
        all_chunks.extend(chunks)
        all_labels.extend([label] * len(chunks))
    
    # Save chunk results
    results_df = pd.DataFrame({
        'text': all_labels,
        'chunk_index': [i for label in set(all_labels) for i in range(all_labels.count(label))],
    })
    
    for concept_name in centroids.keys():
        results_df[concept_name] = all_scores[concept_name]
    
    results_path = output_dir / "chunk_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved chunk results to {results_path}")
    
    # Calculate statistics
    stats = {}
    for label in set(all_labels):
        stats[label] = {}
        mask = results_df['text'] == label
        for concept_name in centroids.keys():
            scores = results_df[mask][concept_name]
            stats[label][concept_name] = {
                "mean": float(scores.mean()),
                "median": float(scores.median()),
                "std": float(scores.std()),
            }
    
    stats_path = output_dir / "statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"{analysis_name.upper()} ANALYSIS")
    print(f"{'='*70}\n")
    
    for label in set(all_labels):
        print(f"{label}:")
        for concept_name in centroids.keys():
            mean_score = stats[label][concept_name]["mean"]
            print(f"  {concept_name}: mean={mean_score:.4f}")
        print()
    
    # Create t-SNE visualization
    print("Running t-SNE...")
    all_embeddings = model.encode(all_chunks, batch_size=32, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=True)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_chunks) - 1))
    coords = tsne.fit_transform(all_embeddings)
    
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.Set3(range(len(set(all_labels))))
    label_to_color = {label: colors[i] for i, label in enumerate(sorted(set(all_labels)))}
    
    for label in sorted(set(all_labels)):
        mask = np.array(all_labels) == label
        ax.scatter(coords[mask, 0], coords[mask, 1], 
                  label=label, alpha=0.6, s=50, c=[label_to_color[label]])
    
    ax.set_title(f"{analysis_name}\nt-SNE Visualization of Semantic Clusters", fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    viz_path = output_dir / "tsne_visualization.png"
    plt.tight_layout()
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {viz_path}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

