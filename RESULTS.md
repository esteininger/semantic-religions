# Analysis Results: Semantic Classification of Religious Texts

This document presents the findings from two distinct semantic analyses of the Bible (KJV), Torah (JPS 1917), and Quran (Rodwell translation).

---

## Table of Contents

1. [Approach 1: Good vs Evil Classification](#approach-1-good-vs-evil-classification)
2. [Approach 2: Liberal Concepts Overlap](#approach-2-liberal-concepts-overlap)
3. [Comparative Analysis](#comparative-analysis)
4. [Methodology](#methodology)
5. [Limitations & Disclaimers](#limitations--disclaimers)

---

## Approach 1: Good vs Evil Classification

### Overview
This analysis measures semantic similarity to concepts associated with "good" (mercy, compassion, justice, forgiveness) versus "evil" (cruelty, oppression, malice, violence).

### Document-Level Results

| Text | Good Similarity | Evil Similarity | Score (Good-Evil) | Classification |
|------|----------------|-----------------|-------------------|----------------|
| **Torah (JPS 1917)** | 0.3218 | 0.3156 | **+0.0062** | Good |
| **Bible (KJV)** | 0.2525 | 0.2547 | **-0.0021** | Evil |
| **Quran (Rodwell)** | 0.4912 | 0.5005 | **-0.0093** | Evil |

**Key Findings:**
- The Torah shows the highest alignment with "good" concepts overall
- All texts have very close scores, indicating mixed content
- The Quran has the highest absolute similarities to both categories, suggesting richer semantic overlap with the seed concepts

### Chunk-Level Analysis (150 chunks per text)

| Text | % Good Chunks | % Evil Chunks | Mean Score |
|------|--------------|---------------|------------|
| **Bible (KJV)** | 48.7% | 51.3% | Nearly balanced |
| **Torah (JPS 1917)** | 36.0% | 64.0% | More "evil" chunks |
| **Quran (Rodwell)** | 20.0% | 80.0% | Predominantly "evil" chunks |

**Interpretation:**
- At the chunk level, the Bible shows the most balanced content
- The Torah shows moderate imbalance (64% evil)
- The Quran shows the strongest imbalance (80% evil)
- This likely reflects:
  - **Narrative content**: Historical battles, laws, punishments
  - **Warnings**: Descriptions of consequences for wrongdoing
  - **Context**: Ancient texts include descriptions of conflict, justice, and retribution

### Visualization

![Good vs Evil Semantic Clustering](assets/good_evil_clusters.png)

**Figure 1**: t-SNE visualization of semantic clustering for Good vs Evil classification
- **Circles with green borders** = "Good" classified chunks
- **X markers with red borders** = "Evil" classified chunks
- Colors: Blue (Bible), Purple (Torah), Orange (Quran)

The visualization shows how chunks from each religious text cluster in semantic space, with distinct patterns visible between texts while also showing overlap in certain conceptual regions.

---

## Approach 2: Liberal Concepts Overlap

### Overview
This analysis measures semantic similarity to modern liberal political and social concepts: individual freedom, civil liberties, equality, democracy, pluralism, religious tolerance, personal autonomy, social justice, etc.

### Document-Level Results (Normalized 0-100 scale)

| Rank | Text | Liberal Similarity | Normalized Score |
|------|------|-------------------|------------------|
| **1st** | **Quran (Rodwell)** | 0.4532 | **45.32%** |
| **2nd** | **Torah (JPS 1917)** | 0.2727 | **27.27%** |
| **3rd** | **Bible (KJV)** | 0.1914 | **19.14%** |

**Key Findings:**
- The Quran shows the **highest overlap** with liberal concepts
- Torah shows moderate overlap
- Bible (KJV) shows the lowest overlap
- The Quran's higher score may reflect:
  - Emphasis on charity, justice, and mercy
  - Concepts of equality before God
  - Discussion of social responsibilities

### Chunk-Level Analysis (150 chunks per text)

| Text | Mean Similarity | Median Similarity | % Above Average | Range |
|------|----------------|-------------------|----------------|-------|
| **Quran (Rodwell)** | 0.1223 | 0.1162 | **60.7%** | -0.0078 to 0.2832 |
| **Torah (JPS 1917)** | 0.1058 | 0.0988 | **49.3%** | -0.1198 to 0.2861 |
| **Bible (KJV)** | 0.0722 | 0.0729 | **33.3%** | -0.0808 to 0.2278 |

**Interpretation:**
- Consistent with document-level results
- Quran has 60.7% of chunks above the overall average
- Torah is near the middle (49.3%)
- Bible has only 33.3% above average
- All texts show significant variation (wide ranges)

### Ranking Summary

**Liberal Concepts Overlap (Highest to Lowest):**
1. 🥇 Quran (Rodwell): **0.1223** mean similarity
2. 🥈 Torah (JPS 1917): **0.1058** mean similarity
3. 🥉 Bible (KJV): **0.0722** mean similarity

### Visualization

![Liberal Concepts Semantic Clustering](assets/liberal_concepts_clusters.png)

**Figure 2**: t-SNE visualization of semantic clustering for Liberal Concepts overlap
- **Larger/darker points** = Higher liberal concept overlap
- Colors: Blue (Bible), Purple (Torah), Orange (Quran)
- Shows clear separation between texts with different overlap levels

The visualization demonstrates how semantic similarity to liberal political concepts varies across and within each text. The gradient of point sizes and opacity reveals the distribution of liberal concept alignment throughout each religious corpus.

---

## Comparative Analysis

### Cross-Method Comparison

| Text | Good vs Evil | Liberal Overlap | Observations |
|------|--------------|-----------------|--------------|
| **Bible (KJV)** | Nearly balanced (48.7% good) | Lowest liberal overlap (19.14%) | Most balanced on moral axis but lowest modern concept alignment |
| **Torah (JPS 1917)** | Leans "evil" (36% good) | Moderate liberal overlap (27.27%) | Middle ground on both dimensions |
| **Quran (Rodwell)** | Strong "evil" lean (20% good) | Highest liberal overlap (45.32%) | Paradoxical: lowest "good" % but highest liberal concept overlap |

### Notable Paradoxes

**The Quran Paradox:**
- Scored 80% "evil" in good vs evil classification
- BUT scored highest (45.32%) on liberal concepts overlap
- **Explanation**: 
  - "Evil" classification likely captures conflict narratives, punishments, and strong justice language
  - Liberal overlap captures emphasis on social justice, charity (zakat), equality before Allah, and community responsibilities
  - Shows limitation of binary "good/evil" framing vs multi-dimensional concept analysis

**The Bible Pattern:**
- Most balanced on good/evil (nearly 50/50)
- Lowest on liberal concepts
- **Possible reasons**:
  - KJV translation is older English (less semantic similarity to modern concepts)
  - More emphasis on divine authority vs individual autonomy
  - Less emphasis on equality-focused language

### Semantic Clustering Insights

Both t-SNE visualizations show:
- **Clear separation** between the three texts
- **Internal coherence** within each text (chunks cluster together)
- **Gradient patterns** rather than binary divisions
- **Overlap zones** where concepts span texts

---

## Methodology

### Semantic Approach

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional embeddings
- Trained on 1B+ sentence pairs
- L2-normalized for cosine similarity

**Seed Phrases**:

*Good vs Evil:*
- Good: kindness, compassion, honesty, generosity, mercy, altruism, justice, charity, forgiveness, protecting the innocent, human dignity, saving lives
- Evil: cruelty, malice, deceit, oppression, brutality, greed at expense of others, terrorizing civilians, injustice, murder, harm for pleasure, torture, genocide, corruption

*Liberal Concepts:*
- individual freedom, civil liberties, human rights, social equality, gender equality, religious tolerance, pluralism, diversity, inclusion, social justice, democratic governance, voting rights, separation of church and state, personal autonomy, bodily autonomy, social safety net, environmental protection, education for all, peaceful resolution of conflict

**Process**:
1. Load and tokenize texts
2. Create embeddings for seed phrases (centroids)
3. Create embeddings for documents/chunks
4. Calculate cosine similarity
5. Classify or score based on similarity
6. Apply t-SNE for 2D visualization

### Analysis Levels

1. **Document-level**: Entire text analyzed as single unit (with sentence sampling for large texts)
2. **Chunk-level**: 500-word chunks analyzed independently (150 chunks per text for performance)

---

## Limitations & Disclaimers

### Translation Effects
- **Results are translation-specific**
- KJV (1611) uses archaic English
- JPS 1917 uses early 20th century English
- Rodwell (1861) uses Victorian English
- Modern translations might yield different results

### Temporal Anachronism
- **Liberal political concepts are modern** (17th-21st century)
- Applying them to ancient texts (3000-1400 years old) is anachronistic
- Results show *semantic similarity to modern language*, not historical intent

### Methodological Limitations

1. **Seed phrase bias**: Different seeds would produce different results
2. **Context loss**: Chunk analysis loses narrative context
3. **Semantic vs theological**: Measures language patterns, not religious truth
4. **Binary simplification**: "Good vs evil" is reductive for complex texts
5. **Embedding limitations**: Model trained on modern English internet text
6. **Sample size**: 150 chunks is a small sample of large texts
7. **Cultural lens**: Analysis reflects Western liberal democratic framework

### Important Notes

⚠️ **This analysis does NOT measure**:
- Theological correctness or religious value
- Historical accuracy or authenticity
- Moral superiority of any religion
- Whether texts are "actually" good or evil

✅ **This analysis DOES measure**:
- Semantic similarity patterns in English translations
- Linguistic alignment with chosen concept sets
- Relative differences between translations
- Distributional semantics in embedding space

### Ethical Considerations

- Results should not be used to make value judgments about religions
- Semantic analysis reveals linguistic patterns, not truth claims
- Religious texts are complex, multi-layered, and context-dependent
- Oversimplification risks misrepresentation
- Cultural and theological nuance cannot be captured by embeddings alone

---

## Future Directions

### Potential Extensions

1. **Additional Dimensions**:
   - Conservative concepts (tradition, authority, order)
   - Mystical concepts (transcendence, unity, divine love)
   - Authoritarian vs libertarian axis
   - Communitarian vs individualist axis

2. **More Texts**:
   - Hindu Vedas and Upanishads
   - Buddhist Pali Canon
   - Sikh Guru Granth Sahib
   - Tao Te Ching
   - Book of Mormon
   - Additional translations of existing texts

3. **Better Methods**:
   - Multilingual models (analyze in original languages)
   - Contextual embeddings (BERT, GPT)
   - Statistical significance testing
   - Sentiment analysis integration
   - Topic modeling combination

4. **Refined Analysis**:
   - Book-by-book breakdown
   - Genre-specific analysis (law, narrative, poetry, prophecy)
   - Character-specific analysis (Jesus, Moses, Muhammad)
   - Temporal evolution within texts

---

## Data Files

All raw data and visualizations available in `output/`:

**Good vs Evil:**
- `results.csv` - Document scores
- `tsne_results.csv` - Chunk-level scores with coordinates
- `tsne_visualization.png` - Visualization
- `results.npy` - Embedding vectors

**Liberal Concepts:**
- `liberal_results.csv` - Document scores
- `liberal_tsne_results.csv` - Chunk-level scores with coordinates
- `liberal_tsne_visualization.png` - Visualization
- `liberal_results.npy` - Embedding vectors

---

## Conclusion

This semantic analysis reveals interesting patterns in how religious texts align with different conceptual frameworks:

1. **Good vs Evil**: All texts contain mixed content, with chunk-level variation suggesting complex narratives that include both moral ideals and descriptions of conflict/punishment.

2. **Liberal Concepts**: Significant variation in overlap, with the Quran showing surprising alignment with modern liberal concepts despite containing strong justice/punishment language.

3. **Multidimensionality**: Religious texts are too complex for single-axis classification. Multi-dimensional analysis provides richer understanding.

4. **Translation matters**: Results are specific to these English translations and may not reflect original texts.

These findings demonstrate both the power and limitations of computational semantic analysis for understanding religious literature. While embeddings can reveal linguistic patterns, they cannot capture the full theological, historical, and cultural richness of sacred texts.

---

*Analysis conducted: October 2025*  
*Model: sentence-transformers/all-MiniLM-L6-v2*  
*Texts: Public domain translations from Project Gutenberg and Open Siddur*

