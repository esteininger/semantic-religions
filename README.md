# Semantic Analysis of Religious Texts

A semantic analysis toolkit that uses transformer-based embeddings to analyze and classify religious texts across different conceptual dimensions.

## Overview

This project performs semantic analysis on three major religious texts:
- **Bible** (King James Version)
- **Torah** (JPS 1917 Translation)
- **Quran** (Rodwell Translation)

The analysis includes **10 distinct semantic dimensions**:

**Core Analyses:**
1. **Good vs Evil Classification**: Measures semantic similarity to concepts of good (mercy, compassion, justice) vs evil (cruelty, oppression, malice)
2. **Liberal Concepts Overlap**: Measures semantic similarity to modern liberal political/social concepts (individual freedom, equality, democracy, pluralism)

**Extended Analyses** (see [EXTENDED_ANALYSES.md](EXTENDED_ANALYSES.md)):
3. **Conservative Concepts**: Tradition, authority, hierarchy, divine law
4. **Mystical vs Legalistic**: Spiritual transcendence vs legal observance
5. **Hope vs Despair**: Salvation and redemption vs doom and judgment  
6. **Love vs Fear**: Divine compassion vs fear of God
7. **Feminine Concepts**: Women's voices, maternal themes, feminine divine
8. **War vs Peace**: Conflict and violence vs harmony and reconciliation
9. **Nature & Environment**: Creation, natural world, environmental stewardship
10. **Wealth vs Poverty**: Economic themes, prosperity vs need and charity

## Features

- **Semantic embeddings** using sentence-transformers (all-MiniLM-L6-v2)
- **Document-level analysis** for overall text classification
- **Two chunking strategies**:
  - **Fixed-size chunking**: Uniform 500-word segments
  - **Semantic chunking**: Variable-size concept-based segments (NEW!)
- **t-SNE visualizations** showing semantic clustering
- **Quantitative metrics** with similarity scores and percentages
- **Comparative analysis** between chunking strategies

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

#### Chunk-level with SEMANTIC CHUNKING (concept-based):
```bash
# Good vs Evil with semantic chunking
python scripts/visualize_tsne_semantic.py \
  --bible data/bible_kjv.txt \
  --torah data/torah_jps1917.txt \
  --quran data/quran_rodwell.txt \
  --target_chunks 150 \
  --out output/tsne_semantic_visualization.png
```

**Semantic Chunking Advantages:**
- Preserves conceptual coherence
- Creates natural boundaries based on semantic similarity
- Variable chunk sizes reflect discourse structure
- More accurate representation of meaning

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

#### Chunk-level t-SNE visualization (fixed chunks):
```bash
python scripts/visualize_liberal_tsne.py \
  --bible data/bible_kjv.txt \
  --torah data/torah_jps1917.txt \
  --quran data/quran_rodwell.txt \
  --chunk_size 500 \
  --max_chunks 150 \
  --out output/liberal_tsne_visualization.png
```

#### Chunk-level with SEMANTIC CHUNKING (concept-based):
```bash
python scripts/visualize_liberal_tsne_semantic.py \
  --bible data/bible_kjv.txt \
  --torah data/torah_jps1917.txt \
  --quran data/quran_rodwell.txt \
  --target_chunks 150 \
  --out output/liberal_tsne_semantic_visualization.png
```

### 3. Extended Analyses

Run all 8 extended analyses at once:

```bash
./run_all_analyses.sh
```

Or run individual analyses:

```bash
python scripts/generalized_concept_analyzer.py \
  --bible data/bible_kjv.txt \
  --torah data/torah_jps1917.txt \
  --quran data/quran_rodwell.txt \
  --concepts concepts/conservative.json \
  --analysis_name "Conservative Concepts" \
  --output_dir output/conservative \
  --use_semantic_chunking
```

**Available concept files:**
- `concepts/conservative.json` - Conservative values
- `concepts/mystical_vs_legalistic.json` - Mystical vs Legalistic
- `concepts/hope_vs_despair.json` - Hope vs Despair
- `concepts/love_vs_fear.json` - Love vs Fear
- `concepts/gender_feminine.json` - Feminine concepts
- `concepts/war_vs_peace.json` - War vs Peace
- `concepts/nature_environment.json` - Nature and Environment
- `concepts/wealth_vs_poverty.json` - Wealth vs Poverty

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

### Extended Analyses:
Each analysis has its own subdirectory in `output/`:
- `output/conservative/` - Conservative concepts results
- `output/mystical_vs_legalistic/` - Mystical vs Legalistic results
- `output/hope_vs_despair/` - Hope vs Despair results
- `output/love_vs_fear/` - Love vs Fear results
- `output/gender_feminine/` - Feminine concepts results
- `output/war_vs_peace/` - War vs Peace results
- `output/nature_environment/` - Nature & Environment results
- `output/wealth_vs_poverty/` - Wealth vs Poverty results

Each subdirectory contains:
- `tsne_visualization.png` - t-SNE plot
- `chunk_results.csv` - Detailed chunk data
- `statistics.json` - Summary statistics

## Project Structure

```
semantic-religions/
├── README.md                   # This file
├── RESULTS.md                  # Core analyses results
├── EXTENDED_ANALYSES.md        # Extended analyses results  
├── requirements.txt            # Python dependencies
├── run_all_analyses.sh         # Batch runner for all analyses
├── scripts/                    # Analysis scripts
│   ├── good_evil_mapper.py     # Document-level good/evil classifier
│   ├── liberal_mapper.py       # Document-level liberal concepts mapper
│   ├── visualize_tsne.py       # Good/evil t-SNE (fixed chunks)
│   ├── visualize_tsne_semantic.py  # Good/evil t-SNE (semantic chunks)
│   ├── visualize_liberal_tsne.py  # Liberal concepts t-SNE (fixed chunks)
│   ├── visualize_liberal_tsne_semantic.py  # Liberal concepts t-SNE (semantic chunks)
│   ├── generalized_concept_analyzer.py  # Generic analyzer for any concepts
│   └── semantic_chunker.py     # Semantic chunking utility module
├── concepts/                   # Concept definition files (JSON)
│   ├── conservative.json
│   ├── mystical_vs_legalistic.json
│   ├── hope_vs_despair.json
│   ├── love_vs_fear.json
│   ├── gender_feminine.json
│   ├── war_vs_peace.json
│   ├── nature_environment.json
│   └── wealth_vs_poverty.json
├── data/                       # Religious texts
│   ├── bible_kjv.txt          # King James Bible
│   ├── torah_jps1917.txt      # JPS 1917 Torah
│   ├── quran_rodwell.txt      # Rodwell Quran
│   └── tanakh1917.txt         # Full JPS 1917 Tanakh
├── assets/                     # Images for documentation
│   ├── good_evil_clusters.png  # Core analyses visualizations
│   ├── good_evil_semantic_clusters.png
│   ├── liberal_concepts_clusters.png
│   ├── liberal_semantic_clusters.png
│   └── analyses/               # Extended analyses visualizations
│       ├── conservative.png
│       ├── mystical_vs_legalistic.png
│       ├── hope_vs_despair.png
│       ├── love_vs_fear.png
│       ├── gender_feminine.png
│       ├── war_vs_peace.png
│       ├── nature_environment.png
│       └── wealth_vs_poverty.png
└── output/                     # Analysis results
    ├── results.csv             # Core good/evil results
    ├── liberal_results.csv     # Core liberal results
    ├── *.png                   # Core visualizations
    ├── *.npy                   # Embedding vectors
    ├── conservative/           # Extended analyses (8 subdirectories)
    ├── mystical_vs_legalistic/
    ├── hope_vs_despair/
    ├── love_vs_fear/
    ├── gender_feminine/
    ├── war_vs_peace/
    ├── nature_environment/
    └── wealth_vs_poverty/
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

### Chunking Strategies

**Fixed-size chunking:**
- Divides text at 500-word boundaries
- Uniform granularity
- Fast and consistent
- May split coherent concepts

**Semantic chunking:**
- Identifies natural conceptual boundaries
- Variable chunk sizes (reflects discourse structure)
- Preserves semantic coherence
- Uses sentence embeddings to detect topic shifts
- More computationally intensive but more accurate

### Visualization
- t-SNE dimensionality reduction (2D)
- Color-coded by religious text
- Size/opacity/markers indicate classification or similarity strength
- Separate visualizations for fixed vs semantic chunking

## Interpretation Notes

⚠️ **Important Disclaimers:**

1. **Translation Effects**: Results reflect specific English translations, not original texts
2. **Modern Concepts**: Liberal political concepts are modern constructs; anachronistic when applied to ancient texts
3. **Semantic vs Theological**: This measures semantic/linguistic patterns, not theological truth or religious value
4. **Seed Bias**: Results depend on chosen seed phrases; different seeds = different results
5. **Context Loss**: Chunk-based analysis loses broader narrative context

## Results Summary

For detailed findings, see:
- [RESULTS.md](RESULTS.md) - Core analyses (Good vs Evil, Liberal Concepts, Semantic Chunking)
- [EXTENDED_ANALYSES.md](EXTENDED_ANALYSES.md) - 8 additional semantic dimensions

**Quick Summary:**

**Core Analyses:**
- **Good vs Evil**: All texts show mixed content (20-51% "good" with fixed chunking)
- **Liberal Concepts**: Quran highest (45%), Bible lowest (19%)
- **Semantic Chunking Impact**: Torah shifts dramatically (+15% "good" with concept-based chunking)

**Extended Analyses Highlights:**
- **Conservative**: Quran highest (0.233), reflects tradition/authority emphasis
- **Mystical vs Legalistic**: Torah most mystical (76%), surprising finding
- **Hope vs Despair**: Bible most hopeful (74%)
- **Love vs Fear**: Torah most love-focused (71%), Quran most fear-based (64%)
- **Feminine Concepts**: Torah highest presence (0.150)
- **War vs Peace**: Quran most peaceful (74%), challenges stereotypes
- **Nature**: Torah highest (0.145), reflects agrarian context
- **Wealth vs Poverty**: All texts poverty-focused (54-62%)

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

