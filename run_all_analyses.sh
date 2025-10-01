#!/bin/bash
# Run all semantic analyses and organize outputs

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BIBLE="data/bible_kjv.txt"
TORAH="data/torah_jps1917.txt"
QURAN="data/quran_rodwell.txt"
ANALYZER="scripts/generalized_concept_analyzer.py"

echo "========================================="
echo "Running All Semantic Analyses"
echo "========================================="
echo ""

# 1. Conservative Concepts
echo "[1/8] Conservative Concepts Analysis..."
python3 "$ANALYZER" \
  --bible "$BIBLE" \
  --torah "$TORAH" \
  --quran "$QURAN" \
  --concepts "concepts/conservative.json" \
  --analysis_name "Conservative Concepts" \
  --output_dir "output/conservative" \
  --use_semantic_chunking

# 2. Mystical vs Legalistic
echo ""
echo "[2/8] Mystical vs Legalistic Analysis..."
python3 "$ANALYZER" \
  --bible "$BIBLE" \
  --torah "$TORAH" \
  --quran "$QURAN" \
  --concepts "concepts/mystical_vs_legalistic.json" \
  --analysis_name "Mystical vs Legalistic" \
  --output_dir "output/mystical_vs_legalistic" \
  --use_semantic_chunking

# 3. Hope vs Despair
echo ""
echo "[3/8] Hope vs Despair Analysis..."
python3 "$ANALYZER" \
  --bible "$BIBLE" \
  --torah "$TORAH" \
  --quran "$QURAN" \
  --concepts "concepts/hope_vs_despair.json" \
  --analysis_name "Hope vs Despair" \
  --output_dir "output/hope_vs_despair" \
  --use_semantic_chunking

# 4. Love vs Fear
echo ""
echo "[4/8] Love vs Fear Analysis..."
python3 "$ANALYZER" \
  --bible "$BIBLE" \
  --torah "$TORAH" \
  --quran "$QURAN" \
  --concepts "concepts/love_vs_fear.json" \
  --analysis_name "Love vs Fear" \
  --output_dir "output/love_vs_fear" \
  --use_semantic_chunking

# 5. Gender (Feminine)
echo ""
echo "[5/8] Gender (Feminine) Analysis..."
python3 "$ANALYZER" \
  --bible "$BIBLE" \
  --torah "$TORAH" \
  --quran "$QURAN" \
  --concepts "concepts/gender_feminine.json" \
  --analysis_name "Feminine Concepts" \
  --output_dir "output/gender_feminine" \
  --use_semantic_chunking

# 6. War vs Peace
echo ""
echo "[6/8] War vs Peace Analysis..."
python3 "$ANALYZER" \
  --bible "$BIBLE" \
  --torah "$TORAH" \
  --quran "$QURAN" \
  --concepts "concepts/war_vs_peace.json" \
  --analysis_name "War vs Peace" \
  --output_dir "output/war_vs_peace" \
  --use_semantic_chunking

# 7. Nature/Environment
echo ""
echo "[7/8] Nature/Environment Analysis..."
python3 "$ANALYZER" \
  --bible "$BIBLE" \
  --torah "$TORAH" \
  --quran "$QURAN" \
  --concepts "concepts/nature_environment.json" \
  --analysis_name "Nature and Environment" \
  --output_dir "output/nature_environment" \
  --use_semantic_chunking

# 8. Wealth vs Poverty
echo ""
echo "[8/8] Wealth vs Poverty Analysis..."
python3 "$ANALYZER" \
  --bible "$BIBLE" \
  --torah "$TORAH" \
  --quran "$QURAN" \
  --concepts "concepts/wealth_vs_poverty.json" \
  --analysis_name "Wealth vs Poverty" \
  --output_dir "output/wealth_vs_poverty" \
  --use_semantic_chunking

echo ""
echo "========================================="
echo "All Analyses Complete!"
echo "========================================="
echo ""
echo "Results organized in output/ subdirectories:"
echo "  - output/conservative/"
echo "  - output/mystical_vs_legalistic/"
echo "  - output/hope_vs_despair/"
echo "  - output/love_vs_fear/"
echo "  - output/gender_feminine/"
echo "  - output/war_vs_peace/"
echo "  - output/nature_environment/"
echo "  - output/wealth_vs_poverty/"
echo ""
echo "Each directory contains:"
echo "  - tsne_visualization.png (t-SNE plot)"
echo "  - chunk_results.csv (detailed chunk-level data)"
echo "  - statistics.json (summary statistics)"
echo ""

# Copy visualizations to assets
echo "Copying visualizations to assets/..."
mkdir -p assets/analyses
cp output/conservative/tsne_visualization.png assets/analyses/conservative.png
cp output/mystical_vs_legalistic/tsne_visualization.png assets/analyses/mystical_vs_legalistic.png
cp output/hope_vs_despair/tsne_visualization.png assets/analyses/hope_vs_despair.png
cp output/love_vs_fear/tsne_visualization.png assets/analyses/love_vs_fear.png
cp output/gender_feminine/tsne_visualization.png assets/analyses/gender_feminine.png
cp output/war_vs_peace/tsne_visualization.png assets/analyses/war_vs_peace.png
cp output/nature_environment/tsne_visualization.png assets/analyses/nature_environment.png
cp output/wealth_vs_poverty/tsne_visualization.png assets/analyses/wealth_vs_poverty.png

echo "Done! Visualizations also copied to assets/analyses/"

