#!/bin/bash

# Script to rename CANShield to Intrusion Detecting System

echo "🔄 Renaming CANShield to Intrusion Detecting System..."
echo ""

# Files to update (excluding environment and git directories)
files=(
    "visualize_results.py"
    "FINAL_OUTPUT_GUIDE.md"
    "TRAINING_COMPLETE_NEXT_STEPS.md"
    "START_COLAB_NOW.md"
    "YOUR_COLAB_GUIDE.md"
    "COLAB_QUICK_START.md"
    "COLAB_WITHOUT_GITHUB.md"
    "GOOGLE_COLAB_TRAINING.md"
    "COLAB_VS_MAC.md"
    "README_ROBUST.md"
    "ROBUST_CANSHIELD_GUIDE.md"
    "TRAINING_OPTIONS_SUMMARY.md"
    "TRAINING_STEPS.md"
    "IMPLEMENTATION_SUMMARY.md"
    "src/run_robust_canshield.py"
    "src/run_robust_evaluation.py"
)

count=0

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        # Use sed to replace CANShield with Intrusion Detecting System
        # macOS sed requires -i '' for in-place editing
        sed -i '' 's/CANShield/Intrusion Detecting System/g' "$file"
        sed -i '' 's/canshield/intrusion_detecting_system/g' "$file"
        
        echo "✅ Updated: $file"
        ((count++))
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Renamed in $count files!"
echo "   CANShield → Intrusion Detecting System"
echo "═══════════════════════════════════════════════════════════"

