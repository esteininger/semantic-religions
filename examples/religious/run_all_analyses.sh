#!/bin/bash
# Run all religious text analyses

echo "Running all religious text semantic analyses..."
echo "================================================"

# Array of concept files
concepts=(
    "conservative"
    "liberal"
    "mystical_vs_legalistic"
    "hope_vs_despair"
    "love_vs_fear"
    "gender_feminine"
    "war_vs_peace"
    "nature_environment"
    "wealth_vs_poverty"
    "illiberal_practices"
)

# Base paths
TEXTS="texts/religious/bible_kjv.txt texts/religious/torah_jps1917.txt texts/religious/quran_rodwell.txt"
LABELS="Bible (KJV)" "Torah (JPS 1917)" "Quran (Rodwell)"
BASE_OUTPUT="examples/religious/output"

# Run each analysis
for concept in "${concepts[@]}"; do
    echo ""
    echo "Running $concept analysis..."
    python3 scripts/analyze.py \
        --texts texts/religious/bible_kjv.txt texts/religious/torah_jps1917.txt texts/religious/quran_rodwell.txt \
        --labels "Bible (KJV)" "Torah (JPS 1917)" "Quran (Rodwell)" \
        --concepts "concepts/${concept}.json" \
        --analysis_name "$concept" \
        --output "$BASE_OUTPUT/$concept" \
        --target_chunks 150 \
        --use_semantic_chunking
    
    echo "✓ Completed $concept analysis"
done

echo ""
echo "================================================"
echo "All analyses complete!"
echo "Results available in: $BASE_OUTPUT"

