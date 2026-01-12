#!/bin/bash
# Translate all dataset splits to Nepali using IndicTrans2

echo "=========================================="
echo "Translating Captions to Nepali"
echo "=========================================="
echo ""

# Activate virtual environment
if [ -d "env" ]; then
    echo "Activating virtual environment..."
    source env/bin/activate
else
    echo "Error: Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please set it with: export HF_TOKEN='your_token_here'"
    exit 1
fi

echo ""
echo "Using HuggingFace token from HF_TOKEN environment variable"
echo ""

# Translate train split
echo "=== Translating train.csv ==="
python3 scripts/translate_captions.py \
    --input data/processed/train.csv \
    --output data/processed/train_nepali.csv \
    --hf_token $HF_TOKEN

if [ $? -ne 0 ]; then
    echo "Error: Translation of train.csv failed"
    exit 1
fi

echo ""
echo "=== Translating valid.csv ==="
python3 scripts/translate_captions.py \
    --input data/processed/valid.csv \
    --output data/processed/valid_nepali.csv \
    --hf_token $HF_TOKEN

if [ $? -ne 0 ]; then
    echo "Error: Translation of valid.csv failed"
    exit 1
fi

echo ""
echo "=== Translating test.csv ==="
python3 scripts/translate_captions.py \
    --input data/processed/test.csv \
    --output data/processed/test_nepali.csv \
    --hf_token $HF_TOKEN

if [ $? -ne 0 ]; then
    echo "Error: Translation of test.csv failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All translations complete!"
echo "=========================================="
echo ""
echo "Generated files:"
ls -lh data/processed/*_nepali.csv
echo ""
echo "Next step: Train the models"
echo "  python train_classifier.py"
echo "  python train_captioner.py"
