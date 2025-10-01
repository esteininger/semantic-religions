# Concept Definitions

This directory contains concept definitions used for semantic analysis. Each JSON file defines one or more concepts through seed phrases.

## Concept File Format

### Single Concept

Measures alignment with one concept:

```json
{
  "mode": "single",
  "concepts": {
    "concept_name": [
      "seed phrase 1",
      "seed phrase 2",
      "..."
    ]
  }
}
```

### Dual Concepts (Comparative)

Compares two opposing or related concepts:

```json
{
  "mode": "dual",
  "concepts": {
    "concept_a": ["phrases for concept A"],
    "concept_b": ["phrases for concept B"]
  }
}
```

## Creating New Concepts

1. Copy `TEMPLATE.json` as a starting point
2. Choose 10-30 representative seed phrases
3. Mix abstract terms with concrete examples
4. Include synonyms and related concepts
5. Save with a descriptive filename

### Tips for Good Seed Phrases

- **Be specific**: Use precise terms that clearly represent the concept
- **Be comprehensive**: Cover different aspects of the concept
- **Avoid ambiguity**: Choose phrases with clear meanings
- **Use variety**: Mix single words, short phrases, and longer descriptions
- **Think contextually**: Consider how the concept appears in actual texts

## Available Concepts

### Political & Social

- **conservative.json** - Traditional values, authority, hierarchy
- **liberal.json** - Individual freedom, equality, pluralism (note: see note below about "liberal")
- **illiberal_practices.json** - Human rights violations, persecution

### Emotional & Psychological

- **hope_vs_despair.json** - Optimism vs pessimism
- **love_vs_fear.json** - Emotional frameworks

### Religious & Philosophical

- **mystical_vs_legalistic.json** - Spiritual experience vs law/ritual

### Social Issues

- **gender_feminine.json** - Feminine themes and women's presence
- **war_vs_peace.json** - Conflict vs harmony
- **wealth_vs_poverty.json** - Economic themes
- **nature_environment.json** - Environmental themes

## Note on "Liberal" Concept

The `liberal.json` concept file measures alignment with **modern Western liberal values** (individual autonomy, civil liberties, human rights, social equality, democracy, religious tolerance).

This is a specific, modern political philosophy, not "liberal" in other senses (generous, free-flowing, etc.). The analysis is intentionally measuring alignment with this particular framework.

## Examples

### Example 1: Optimism vs Pessimism

```json
{
  "mode": "dual",
  "concepts": {
    "optimism": [
      "hope", "bright future", "positive outlook", "opportunity",
      "success", "confidence", "progress", "improvement"
    ],
    "pessimism": [
      "doom", "despair", "hopelessness", "decline",
      "failure", "negativity", "worst case", "deterioration"
    ]
  }
}
```

### Example 2: Scientific Language

```json
{
  "mode": "single",
  "concepts": {
    "scientific": [
      "hypothesis", "experiment", "data", "evidence", "theory",
      "empirical observation", "statistical analysis", "peer review",
      "methodology", "objective measurement", "reproducibility"
    ]
  }
}
```

### Example 3: Entrepreneurial Spirit

```json
{
  "mode": "single",
  "concepts": {
    "entrepreneurial": [
      "innovation", "startup", "venture capital", "disruption",
      "business opportunity", "risk-taking", "market creation",
      "entrepreneurship", "founding", "scaling", "growth mindset"
    ]
  }
}
```

## Using Concepts in Analysis

```bash
# Single concept
python scripts/analyze.py \
  --concepts concepts/nature_environment.json \
  --texts text1.txt text2.txt \
  --labels "Text 1" "Text 2" \
  --analysis_name "Environmental Themes" \
  --output output/environment

# Comparative (dual concepts)
python scripts/analyze.py \
  --concepts concepts/hope_vs_despair.json \
  --texts text1.txt text2.txt \
  --labels "Text 1" "Text 2" \
  --analysis_name "Hope vs Despair" \
  --output output/hope_despair
```

## Contributing New Concepts

If you create useful concept definitions, consider contributing them back to the project!

1. Ensure good coverage (10-30 phrases)
2. Test on sample texts
3. Document the concept's purpose
4. Submit via pull request

