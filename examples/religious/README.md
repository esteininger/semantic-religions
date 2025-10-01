# Religious Texts Analysis Example

This directory contains an example analysis of three major religious texts using semantic embeddings.

## Texts Analyzed

- **Bible** (King James Version)
- **Torah** (JPS 1917 Translation)  
- **Quran** (Rodwell Translation)

## Concept Analyses

This example includes 11 different conceptual analyses:

1. **Good vs Evil** - Moral classification
2. **Liberal Concepts** - Alignment with modern liberal values
3. **Conservative Concepts** - Traditional values and authority
4. **Mystical vs Legalistic** - Spiritual experience vs legal frameworks
5. **Hope vs Despair** - Emotional valence
6. **Love vs Fear** - Emotional frameworks
7. **Feminine Concepts** - Presence of women and feminine themes
8. **War vs Peace** - Conflict vs harmony
9. **Nature & Environment** - Environmental themes
10. **Wealth vs Poverty** - Economic themes
11. **Illiberal Practices** - Human rights violations

## Running Analyses

### Using the Config File

```bash
# Run a single analysis
python scripts/analyze.py --config examples/religious/config.json

# Or modify the config to use different concepts
# Edit config.json to point to different concept files
```

### Direct Command Line

```bash
python scripts/analyze.py \
  --texts texts/religious/bible_kjv.txt \
          texts/religious/torah_jps1917.txt \
          texts/religious/quran_rodwell.txt \
  --labels "Bible (KJV)" "Torah (JPS 1917)" "Quran (Rodwell)" \
  --concepts concepts/conservative.json \
  --analysis_name "Conservative Concepts" \
  --output examples/religious/output/conservative \
  --target_chunks 150 \
  --use_semantic_chunking
```

## Results

Detailed results and findings are available in:
- **RESULTS.md** - Core findings for good/evil and liberal concepts analyses
- **EXTENDED_ANALYSES.md** - Complete findings for all 9 extended analyses
- **output/** - Raw data, visualizations, and statistics for each analysis
- **assets/** - Visualization images

## Key Findings Summary

See [RESULTS.md](RESULTS.md) and [EXTENDED_ANALYSES.md](EXTENDED_ANALYSES.md) for complete analysis and interpretation.

### Highlights:

- All texts contain mixed moral content (good and evil themes)
- Significant variation in alignment with modern liberal values
- Quran shows highest conservative alignment and illiberal practices scores
- Torah shows highest mystical orientation and love-based framework
- Bible shows most balanced content across many dimensions
- All texts emphasize poverty/charity over wealth

## Methodology

- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Chunking**: Semantic chunking (variable-size, concept-preserving)
- **Similarity**: Cosine similarity with concept centroids
- **Visualization**: t-SNE dimensionality reduction

## Disclaimers

⚠️ **Important Notes:**
- Results reflect English translations, not original texts
- Semantic similarity measures linguistic patterns, not theological truth
- Concept definitions reflect modern Western perspectives
- Multiple valid interpretations exist for all findings
- This is computational analysis, not religious scholarship

