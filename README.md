# Semantic Analysis of Religious Texts

A semantic analysis toolkit that uses transformer-based embeddings to analyze and classify religious texts across different conceptual dimensions.

## Overview

This project performs semantic analysis on three major religious texts:
- **Bible** (King James Version)
- **Torah** (JPS 1917 Translation)
- **Quran** (Rodwell Translation)

The analysis includes two distinct approaches:
1. **Good vs Evil Classification**: Measures semantic similarity to concepts of good (mercy, compassion, justice) vs evil (cruelty, oppression, malice)
2. **Liberal Concepts Overlap**: Measures semantic similarity to modern liberal political/social concepts (individual freedom, equality, democracy, pluralism)

## Features

- **Semantic embeddings** using sentence-transformers (all-MiniLM-L6-v2)
- **Document-level analysis** for overall text classification
- **Chunk-level analysis** for granular understanding (500-word chunks)
- **t-SNE visualizations** showing semantic clustering
- **Quantitative metrics** with similarity scores and percentages

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd semantic-religions
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the religious texts (already included in `data/`):
```bash
# If you need to re-download:
cd data
curl -L -o bible_kjv.txt https://www.gutenberg.org/ebooks/10.txt.utf-8
curl -L -o quran_rodwell.txt https://www.gutenberg.org/ebooks/7440.txt.utf-8
curl -L -o tanakh1917.txt https://opensiddur.org/wp-content/uploads/2010/08/Tanakh1917.txt
# Extract Torah (Genesis through Deuteronomy)
sed -n '845,27080p' tanakh1917.txt > torah_jps1917.txt
cd ..
```

## Usage

### 1. Good vs Evil Classification

#### Document-level analysis:
```bash
python scripts/good_evil_mapper.py \
  data/bible_kjv.txt \
  data/torah_jps1917.txt \
  data/quran_rodwell.txt \
  --out output/results.csv \
  --dump_vectors
```

#### Chunk-level t-SNE visualization:
```bash
python scripts/visualize_tsne.py \
  --bible data/bible_kjv.txt \
  --torah data/torah_jps1917.txt \
  --quran data/quran_rodwell.txt \
  --chunk_size 500 \
  --max_chunks 150 \
  --out output/tsne_visualization.png
```

**Parameters:**
- `--chunk_size`: Number of words per chunk (default: 500)
- `--max_chunks`: Maximum chunks per text for performance (default: 150)
- `--out`: Output file path
- `--dump_vectors`: Save embeddings as .npy file

### 2. Liberal Concepts Overlap

#### Document-level analysis:
```bash
python scripts/liberal_mapper.py \
  data/bible_kjv.txt \
  data/torah_jps1917.txt \
  data/quran_rodwell.txt \
  --out output/liberal_results.csv \
  --dump_vectors
```

#### Chunk-level t-SNE visualization:
```bash
python scripts/visualize_liberal_tsne.py \
  --bible data/bible_kjv.txt \
  --torah data/torah_jps1917.txt \
  --quran data/quran_rodwell.txt \
  --chunk_size 500 \
  --max_chunks 150 \
  --out output/liberal_tsne_visualization.png
```

## Output Files

All outputs are saved to the `output/` directory:

### Good vs Evil Analysis:
- `results.csv` - Document-level classifications
- `results.npy` - Document embedding vectors
- `tsne_visualization.png` - t-SNE clustering visualization
- `tsne_results.csv` - Chunk-level data with coordinates

### Liberal Concepts Analysis:
- `liberal_results.csv` - Document-level similarity scores
- `liberal_results.npy` - Document embedding vectors
- `liberal_tsne_visualization.png` - t-SNE clustering visualization
- `liberal_tsne_results.csv` - Chunk-level data with coordinates

## Project Structure

```
semantic-religions/
├── README.md                   # This file
├── RESULTS.md                  # Detailed analysis results
├── requirements.txt            # Python dependencies
├── scripts/                    # Analysis scripts
│   ├── good_evil_mapper.py     # Document-level good/evil classifier
│   ├── visualize_tsne.py       # Good/evil t-SNE visualization
│   ├── liberal_mapper.py       # Document-level liberal concepts mapper
│   └── visualize_liberal_tsne.py  # Liberal concepts t-SNE visualization
├── data/                       # Religious texts
│   ├── bible_kjv.txt          # King James Bible
│   ├── torah_jps1917.txt      # JPS 1917 Torah
│   ├── quran_rodwell.txt      # Rodwell Quran
│   └── tanakh1917.txt         # Full JPS 1917 Tanakh
├── assets/                     # Images for documentation
│   ├── good_evil_clusters.png # t-SNE visualization (good vs evil)
│   └── liberal_concepts_clusters.png  # t-SNE visualization (liberal concepts)
└── output/                     # Analysis results
    ├── results.csv
    ├── liberal_results.csv
    ├── *.png                   # Visualizations
    └── *.npy                   # Embedding vectors
```

## Methodology

### Semantic Embeddings
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embeddings are L2-normalized for cosine similarity calculations
- Documents are split into sentences, sampled if needed, and averaged

### Classification Approach

**Good vs Evil:**
- **Good seeds**: kindness, compassion, honesty, generosity, mercy, justice, charity, forgiveness, etc.
- **Evil seeds**: cruelty, malice, deceit, oppression, brutality, murder, torture, corruption, etc.
- **Score**: `similarity(text, good_centroid) - similarity(text, evil_centroid)`

**Liberal Concepts:**
- **Liberal seeds**: individual freedom, civil liberties, human rights, social equality, gender equality, religious tolerance, pluralism, democracy, separation of church and state, personal autonomy, social justice, etc.
- **Score**: `similarity(text, liberal_centroid)`

### Visualization
- t-SNE dimensionality reduction (2D)
- Color-coded by religious text
- Size/opacity/markers indicate classification or similarity strength

## Interpretation Notes

⚠️ **Important Disclaimers:**

1. **Translation Effects**: Results reflect specific English translations, not original texts
2. **Modern Concepts**: Liberal political concepts are modern constructs; anachronistic when applied to ancient texts
3. **Semantic vs Theological**: This measures semantic/linguistic patterns, not theological truth or religious value
4. **Seed Bias**: Results depend on chosen seed phrases; different seeds = different results
5. **Context Loss**: Chunk-based analysis loses broader narrative context

## Results Summary

For detailed findings, see [RESULTS.md](RESULTS.md).

**Quick Summary:**
- Good vs Evil: All texts show mixed content with slight variations
- Liberal Concepts: Significant variation in overlap with modern liberal concepts

## Contributing

Contributions are welcome! Areas for improvement:
- Additional conceptual dimensions (conservative, mystical, authoritarian, etc.)
- More religious texts (Vedas, Buddhist sutras, etc.)
- Different embedding models or methods
- Better visualizations
- Statistical significance testing

## License

This project uses public domain texts from Project Gutenberg and Open Siddur.

Code is provided as-is for research and educational purposes.

## Acknowledgments

- **Texts**: Project Gutenberg, Open Siddur Project, JPS
- **Models**: Sentence-Transformers (HuggingFace)
- **Libraries**: scikit-learn, pandas, matplotlib, numpy

## Contact

For questions or feedback, please open an issue.

