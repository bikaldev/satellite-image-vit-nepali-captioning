#!/bin/bash
# Script to run the complete training pipeline (Classifier -> Captioner)
# Usage: ./run_training.sh

echo "=========================================="
echo "Starting Satellite Image Training Pipeline"
echo "=========================================="
echo ""

# Check for environment
if [ -d "env/bin" ]; then
    source env/bin/activate
fi

# Determine Python executable
PYTHON_EXEC="python"
if [ -f "env/Scripts/python.exe" ]; then
    # Windows environment detected
    PYTHON_EXEC="./env/Scripts/python.exe"
elif [ -f "env/bin/python" ]; then
    # Linux environment detected (if not activated above)
    PYTHON_EXEC="./env/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_EXEC="python3"
elif ! command -v python &> /dev/null; then
    echo "Error: Python interpreter not found in environment!"
    exit 1
fi

echo "Using Python: $PYTHON_EXEC"
$PYTHON_EXEC --version

# 1. Train Classification Model
echo ""
echo "------------------------------------------"
echo "Step 1: Training Classification Model"
echo "------------------------------------------"
$PYTHON_EXEC train_classifier.py \
    --config configs/config.yaml \
    --output_dir outputs/classification

if [ $? -ne 0 ]; then
    echo "Error: Classification training failed!"
    exit 1
fi

# 2. Train Captioning Model
echo ""
echo "------------------------------------------"
echo "Step 2: Training Captioning Model"
echo "------------------------------------------"
echo "Note: This will use the best checkpoint from classification if configured,"
echo "      but primarily trains the independent captioner module."
$PYTHON_EXEC train_captioner.py \
    --config configs/config.yaml \
    --output_dir outputs/captioning

if [ $? -ne 0 ]; then
    echo "Error: Captioning training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training Pipeline Completed Successfully!"
echo "=========================================="
