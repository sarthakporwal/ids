#!/bin/bash

# CANShield - Large File Preparation Script
# Quickly create a web-uploadable version of your large dataset

echo "🛡️  CANShield - Large File Processor"
echo "========================================"
echo ""

# Check if file provided
if [ -z "$1" ]; then
    echo "Usage: ./prepare_large_file.sh <path_to_large_file>"
    echo ""
    echo "Example:"
    echo "  ./prepare_large_file.sh train_1.csv"
    echo "  ./prepare_large_file.sh datasets/can-ids/syncan/ambient/train_1.csv"
    echo ""
    exit 1
fi

INPUT_FILE="$1"

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: File not found: $INPUT_FILE"
    exit 1
fi

# Get file info
FILE_SIZE=$(du -h "$INPUT_FILE" | cut -f1)
FILE_NAME=$(basename "$INPUT_FILE")
FILE_DIR=$(dirname "$INPUT_FILE")

echo "📁 Input File: $FILE_NAME"
echo "📦 File Size: $FILE_SIZE"
echo ""

# Check file size
FILE_SIZE_MB=$(du -m "$INPUT_FILE" | cut -f1)

if [ "$FILE_SIZE_MB" -lt 100 ]; then
    echo "✅ File is already small enough (<100MB)"
    echo "   You can upload this directly in the web interface!"
    echo ""
    echo "🚀 To launch web interface:"
    echo "   streamlit run app.py"
    exit 0
fi

echo "⚠️  File is large (${FILE_SIZE})"
echo "   Creating web-friendly version..."
echo ""

# Ask for sample size
echo "📊 Choose sample size:"
echo "  1) 25,000 rows  (~15MB) - Fastest, good for testing"
echo "  2) 50,000 rows  (~30MB) - Recommended ⭐"
echo "  3) 100,000 rows (~60MB) - More data"
echo "  4) Custom"
echo ""
read -p "Choose option [1-4] (default: 2): " choice
choice=${choice:-2}

case $choice in
    1) SAMPLE_SIZE=25000 ;;
    2) SAMPLE_SIZE=50000 ;;
    3) SAMPLE_SIZE=100000 ;;
    4)
        read -p "Enter custom sample size: " SAMPLE_SIZE
        ;;
    *)
        echo "Invalid choice. Using default (50,000)"
        SAMPLE_SIZE=50000
        ;;
esac

echo ""
echo "🎯 Creating dataset with $SAMPLE_SIZE rows..."
echo ""

# Create output filename
OUTPUT_FILE="${FILE_DIR}/sampled_${FILE_NAME}"

# Run processor
python large_file_processor.py "$INPUT_FILE" \
    --action sample \
    --sample-size "$SAMPLE_SIZE" \
    --output "$OUTPUT_FILE"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════"
    echo "✅ SUCCESS!"
    echo "════════════════════════════════════════"
    echo ""
    echo "📁 Sampled file created: $OUTPUT_FILE"
    
    OUTPUT_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "📦 Size: $OUTPUT_SIZE"
    echo ""
    echo "🎉 This file is ready for web upload!"
    echo ""
    echo "🚀 Next steps:"
    echo "  1. Launch web interface:"
    echo "     streamlit run app.py"
    echo ""
    echo "  2. In browser:"
    echo "     • Click 'Upload CSV File'"
    echo "     • Select: $OUTPUT_FILE"
    echo "     • Click 'Run Detection'"
    echo ""
else
    echo ""
    echo "❌ Error occurred during processing"
    echo "   Check the error message above"
    exit 1
fi

