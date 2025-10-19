#!/bin/bash

# ========================================
# Package CANShield for Google Colab
# ========================================

echo "📦 Packaging CANShield for Google Colab..."
echo ""

# Create temporary directory
TEMP_DIR="canshield_colab_package"
rm -rf $TEMP_DIR
mkdir -p $TEMP_DIR

# Copy enhanced modules
echo "✅ Copying adversarial module..."
cp -r src/adversarial $TEMP_DIR/

echo "✅ Copying domain adaptation module..."
cp -r src/domain_adaptation $TEMP_DIR/

echo "✅ Copying model compression module..."
cp -r src/model_compression $TEMP_DIR/

echo "✅ Copying uncertainty module..."
cp -r src/uncertainty $TEMP_DIR/

echo "✅ Copying training scripts..."
cp src/run_robust_canshield.py $TEMP_DIR/
cp src/run_robust_evaluation.py $TEMP_DIR/

echo "✅ Copying configuration..."
mkdir -p $TEMP_DIR/config
cp config/robust_canshield.yaml $TEMP_DIR/config/

echo "✅ Copying documentation..."
cp GOOGLE_COLAB_TRAINING.md $TEMP_DIR/
cp README_ROBUST.md $TEMP_DIR/

# Create zip file
echo ""
echo "📦 Creating zip file..."
zip -r canshield_colab_package.zip $TEMP_DIR/ > /dev/null

# Cleanup
rm -rf $TEMP_DIR

# Show result
SIZE=$(du -h canshield_colab_package.zip | cut -f1)
echo ""
echo "✅ Package created: canshield_colab_package.zip ($SIZE)"
echo ""
echo "📤 To use on Google Colab:"
echo "   1. Go to https://colab.research.google.com"
echo "   2. Upload this zip file"
echo "   3. Run: !unzip canshield_colab_package.zip"
echo "   4. Follow instructions in GOOGLE_COLAB_TRAINING.md"
echo ""
echo "🚀 Ready for Colab training!"

