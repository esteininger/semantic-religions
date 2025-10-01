# Semantic Text Analyzer

A flexible, corpus-agnostic semantic analysis toolkit that uses transformer-based embeddings to analyze and classify any texts across custom conceptual dimensions.

## Overview

This toolkit provides a configuration-driven approach to semantic text analysis, allowing you to:

- Analyze **any texts** (novels, articles, speeches, religious texts, etc.)
- Define **custom concepts** to measure (political ideology, emotional tone, themes, etc.)
- Compare texts across semantic dimensions
- Visualize semantic clustering with t-SNE
- Generate quantitative similarity metrics

### Key Features

- **Corpus-Agnostic**: Works with any text files
- **Flexible Concepts**: Define your own concept vocabularies via JSON
- **Two Chunking Strategies**:
  - Fixed-size chunking for uniformity
  - Semantic chunking for concept-preserving segments
- **Configuration-Driven**: Use JSON configs or command-line arguments
- **Visualizations**: t-SNE plots showing semantic clusters
- **Quantitative Metrics**: Mean, median, and distribution statistics

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone this repository:
```bash
git clone https://github.com/esteininger/semantic-religions.git
cd semantic-religions
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Texts

Place your text files in the `texts/` directory (or anywhere):

```
texts/
├── my_corpus/
│   ├── document1.txt
│   ├── document2.txt
│   └── document3.txt
```

### 2. Define Your Concepts

Create a JSON file in `concepts/` defining the concepts you want to measure:

```json
{
  "mode": "single",
  "concepts": {
    "optimism": [
      "hope", "optimism", "bright future", "positive outlook",
      "enthusiasm", "confidence", "success", "opportunity"
    ]
  }
}
```

For comparative analysis (A vs B):

```json
{
  "mode": "dual",
  "concepts": {
    "progressive": ["innovation", "change", "progress", "reform", "future"],
    "traditional": ["tradition", "heritage", "preservation", "continuity", "past"]
  }
}
```

### 3. Run Analysis

#### Using a Config File (Recommended)

Create a config file:

```json
{
  "analysis_name": "My Analysis",
  "texts": [
    "texts/my_corpus/document1.txt",
    "texts/my_corpus/document2.txt"
  ],
  "labels": ["Document 1", "Document 2"],
  "concepts": "concepts/my_concepts.json",
  "output": "output/my_analysis",
  "target_chunks": 100,
  "use_semantic_chunking": true
}
```

Run the analysis:

```bash
python scripts/analyze.py --config my_config.json
```

#### Using Command Line

```bash
python scripts/analyze.py \
  --texts text1.txt text2.txt text3.txt \
  --labels "Text 1" "Text 2" "Text 3" \
  --concepts concepts/my_concepts.json \
  --analysis_name "My Analysis" \
  --output output/my_analysis \
  --target_chunks 100 \
  --use_semantic_chunking
```

## Project Structure

```
semantic-text-analyzer/
├── texts/                      # Your corpus files
│   └── [corpus_name]/         # Organized by corpus
├── concepts/                   # Concept definitions (JSON)
│   ├── conservative.json
│   ├── liberal.json
│   └── [your_concepts].json
├── scripts/                    # Analysis scripts
│   ├── analyze.py             # Main generic analyzer
│   └── semantic_chunker.py    # Semantic chunking utility
├── examples/                   # Example analyses
│   └── religious/             # Religious texts example
│       ├── README.md
│       ├── config.json
│       ├── RESULTS.md
│       └── output/
├── output/                     # Analysis results (generated)
├── README.md                   # This file
└── requirements.txt           # Python dependencies
```

## Methodology

### Semantic Embeddings
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (configurable)
- Embeddings are L2-normalized for cosine similarity
- 384-dimensional dense vectors

### Concept Centroids
Each concept is defined by seed phrases. The analyzer:
1. Embeds each seed phrase
2. Calculates the centroid (mean vector)
3. Measures cosine similarity between text chunks and centroids

### Chunking Strategies

**Fixed-size chunking:**
- Uniform segments (e.g., 500 words)
- Fast and consistent
- May split coherent concepts

**Semantic chunking:**
- Variable-size segments based on topic shifts
- Preserves conceptual coherence
- Detects natural boundaries using embedding similarity
- More computationally intensive but more accurate

### Visualization
- t-SNE dimensionality reduction to 2D
- Color-coded by text
- Shows semantic clustering patterns

## Example Use Cases

### 1. Political Ideology Analysis

Analyze speeches or manifestos for liberal vs conservative ideology:

```json
{
  "concepts": {
    "progressive": ["change", "reform", "innovation", "equality", "rights"],
    "conservative": ["tradition", "stability", "order", "heritage", "values"]
  }
}
```

### 2. Emotional Tone Analysis

Compare emotional frameworks in literature:

```json
{
  "concepts": {
    "positive": ["joy", "love", "happiness", "hope", "warmth"],
    "negative": ["sadness", "anger", "fear", "despair", "grief"]
  }
}
```

### 3. Scientific vs Mystical Language

Analyze how texts balance empirical and spiritual language:

```json
{
  "concepts": {
    "empirical": ["data", "evidence", "experiment", "observation", "measurement"],
    "mystical": ["transcendent", "spiritual", "divine", "mystical", "sacred"]
  }
}
```

### 4. Environmental Themes

Measure environmental consciousness in texts over time:

```json
{
  "concepts": {
    "environmental": ["nature", "ecology", "conservation", "sustainability", 
                     "climate", "biodiversity", "planet", "earth"]
  }
}
```

## Religious Texts Example

This repository includes a comprehensive example analyzing Bible, Torah, and Quran across 11 conceptual dimensions:

- Good vs Evil
- Liberal vs Conservative
- Hope vs Despair
- Love vs Fear
- And more...

See [examples/religious/](examples/religious/) for details.

## Output Files

Each analysis generates:

```
output/
└── [analysis_name]/
    ├── chunk_results.csv           # Chunk-level scores
    ├── statistics.json             # Summary statistics
    └── tsne_visualization.png      # t-SNE plot
```

**chunk_results.csv:**
- One row per chunk
- Columns: text, chunk_index, [concept_scores]

**statistics.json:**
- Mean, median, std for each text-concept pair

**tsne_visualization.png:**
- 2D visualization of semantic clusters

## Advanced Usage

### Custom Embedding Models

Use any sentence-transformers model:

```bash
python scripts/analyze.py \
  --config my_config.json \
  --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### Batch Analysis

Create multiple configs and run them sequentially:

```bash
for config in configs/*.json; do
    python scripts/analyze.py --config "$config"
done
```

### Integration with Other Tools

The analyzer outputs standard CSV and JSON formats for easy integration with:
- Pandas/Python analysis
- R statistical analysis
- Tableau/PowerBI visualization
- Custom dashboards

## Limitations & Considerations

1. **Language**: Default model works best with English
2. **Context**: Chunking loses some broader narrative context
3. **Concept Bias**: Results depend on seed phrase selection
4. **Semantic vs Intent**: Measures linguistic patterns, not author intent
5. **Translation Effects**: Translated texts may not reflect originals

## Contributing

Contributions welcome! Areas for improvement:

- Additional embedding model support
- Multi-language analysis
- More sophisticated chunking algorithms
- Statistical significance testing
- Interactive visualizations
- Pre-built concept libraries

## Citation

If you use this toolkit in research, please cite:

```
Semantic Text Analyzer
https://github.com/esteininger/semantic-religions
```

## License

This project uses public domain texts and open-source libraries.
Code is provided for research and educational purposes.

## Acknowledgments

- **Models**: Sentence-Transformers (HuggingFace)
- **Libraries**: scikit-learn, pandas, matplotlib, numpy
- **Example Texts**: Project Gutenberg, Open Siddur Project

## Contact

For questions or feedback, please open an issue on GitHub.
