#!/usr/bin/env python3
"""
Semantic chunking utility - splits text into coherent conceptual segments.

Instead of fixed-size word chunks, this creates variable-length chunks based on
semantic coherence, identifying conceptual boundaries by measuring similarity
between consecutive sentences.
"""

import re
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Sentence splitting regex
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = SENT_SPLIT.split(text.strip())
    # Filter out very short sentences (likely errors)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences


def semantic_chunk_text(
    text: str,
    model,
    min_chunk_size: int = 5,
    max_chunk_size: int = 30,
    similarity_threshold: float = 0.7,
    verbose: bool = False
) -> List[str]:
    """
    Split text into semantically coherent chunks.
    
    Algorithm:
    1. Split text into sentences
    2. Embed all sentences
    3. Measure similarity between consecutive sentences
    4. When similarity drops below threshold, start new chunk
    5. Enforce min/max chunk sizes for practicality
    
    Args:
        text: Input text to chunk
        model: SentenceTransformer model for embeddings
        min_chunk_size: Minimum sentences per chunk
        max_chunk_size: Maximum sentences per chunk
        similarity_threshold: Threshold for starting new chunk (0-1)
        verbose: Print debug info
    
    Returns:
        List of semantically coherent text chunks
    """
    sentences = split_sentences(text)
    
    if len(sentences) == 0:
        return []
    
    if verbose:
        print(f"Total sentences: {len(sentences)}")
    
    # Embed all sentences
    embeddings = model.encode(
        sentences,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=verbose
    )
    
    # Calculate similarity between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0, 0]
        similarities.append(sim)
    
    if verbose:
        print(f"Mean consecutive similarity: {np.mean(similarities):.3f}")
        print(f"Std consecutive similarity: {np.std(similarities):.3f}")
    
    # Identify chunk boundaries
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(len(similarities)):
        current_size = len(current_chunk)
        sim = similarities[i]
        
        # Decide whether to add to current chunk or start new one
        should_break = False
        
        # Force break if max size reached
        if current_size >= max_chunk_size:
            should_break = True
        # Break if similarity drops and we've met min size
        elif current_size >= min_chunk_size and sim < similarity_threshold:
            should_break = True
        
        if should_break:
            # Finalize current chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            # Add to current chunk
            current_chunk.append(sentences[i + 1])
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    if verbose:
        chunk_sizes = [len(c.split()) for c in chunks]
        print(f"Created {len(chunks)} semantic chunks")
        print(f"Chunk size (words): min={min(chunk_sizes)}, "
              f"max={max(chunk_sizes)}, mean={np.mean(chunk_sizes):.1f}")
    
    return chunks


def adaptive_semantic_chunk(
    text: str,
    model,
    target_num_chunks: int = 150,
    min_sentences: int = 3,
    max_sentences: int = 50,
    verbose: bool = False
) -> List[str]:
    """
    Create semantic chunks with adaptive threshold to hit target count.
    
    This iteratively adjusts the similarity threshold to produce approximately
    the target number of chunks while maintaining semantic coherence.
    
    Args:
        text: Input text
        model: SentenceTransformer model
        target_num_chunks: Desired number of chunks
        min_sentences: Min sentences per chunk
        max_sentences: Max sentences per chunk
        verbose: Print debug info
    
    Returns:
        List of semantic chunks
    """
    # Try different thresholds to hit target
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    best_chunks = None
    best_diff = float('inf')
    
    for threshold in thresholds:
        chunks = semantic_chunk_text(
            text,
            model,
            min_chunk_size=min_sentences,
            max_chunk_size=max_sentences,
            similarity_threshold=threshold,
            verbose=False
        )
        
        diff = abs(len(chunks) - target_num_chunks)
        if verbose:
            print(f"Threshold {threshold:.2f}: {len(chunks)} chunks (diff: {diff})")
        
        if diff < best_diff:
            best_diff = diff
            best_chunks = chunks
        
        # If we're close enough, stop
        if diff <= target_num_chunks * 0.1:  # Within 10%
            break
    
    if verbose:
        print(f"\nSelected {len(best_chunks)} chunks (target: {target_num_chunks})")
    
    return best_chunks


def percentile_based_chunking(
    text: str,
    model,
    target_num_chunks: int = 150,
    verbose: bool = False
) -> List[str]:
    """
    Use percentile-based similarity threshold for chunking.
    
    This calculates all consecutive sentence similarities, then uses a
    percentile threshold to identify natural break points.
    
    Args:
        text: Input text
        model: SentenceTransformer model
        target_num_chunks: Desired number of chunks
        verbose: Print debug info
    
    Returns:
        List of semantic chunks
    """
    sentences = split_sentences(text)
    
    if len(sentences) < target_num_chunks:
        # Not enough sentences, return sentence-level chunks
        return sentences
    
    # Embed all sentences
    embeddings = model.encode(
        sentences,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=verbose
    )
    
    # Calculate similarities
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0, 0]
        similarities.append(sim)
    
    # Use percentile to find good break points
    # Lower similarity = natural break point
    percentile = 100 * (target_num_chunks / len(sentences))
    threshold = np.percentile(similarities, percentile)
    
    if verbose:
        print(f"Using {percentile:.1f}th percentile threshold: {threshold:.3f}")
    
    # Create chunks at break points
    chunks = []
    current_chunk = [sentences[0]]
    
    for i, sim in enumerate(similarities):
        if sim < threshold and len(current_chunk) >= 3:  # Min 3 sentences
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    if verbose:
        print(f"Created {len(chunks)} chunks")
        chunk_lens = [len(c.split()) for c in chunks]
        print(f"Word counts: min={min(chunk_lens)}, max={max(chunk_lens)}, "
              f"mean={np.mean(chunk_lens):.1f}")
    
    return chunks


if __name__ == "__main__":
    # Test the chunking
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Test text
    test_text = """
    The Lord is my shepherd; I shall not want. He maketh me to lie down in green pastures.
    He leadeth me beside the still waters. He restoreth my soul.
    
    And now about something completely different. The dietary laws were very specific.
    You shall not eat shellfish. You shall not mix meat and dairy.
    
    Blessed are the meek, for they shall inherit the earth. Blessed are the merciful.
    Blessed are the pure in heart, for they shall see God.
    """
    
    print("Testing semantic chunking:")
    chunks = semantic_chunk_text(test_text, model, verbose=True)
    print(f"\nChunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)

