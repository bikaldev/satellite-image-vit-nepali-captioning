#!/bin/bash
# Run preprocessing to generate processed CSV files

echo "=========================================="
echo "Running Data Preprocessing"
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

# Run preprocessing
echo ""
echo "Processing raw data files..."
echo "This will create:"
echo "  - data/processed/train.csv"
echo "  - data/processed/valid.csv"
echo "  - data/processed/test.csv"
echo ""

python3 preprocessing.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Preprocessing complete!"
    echo "=========================================="
    echo ""
    echo "Generated files:"
    ls -lh data/processed/*.csv 2>/dev/null || echo "  (files will appear after successful run)"
    echo ""
    echo "Next step: Translate captions to Nepali"
    echo "  python scripts/translate_captions.py --input data/processed/train.csv --output data/processed/train_nepali.csv"
else
    echo ""
    echo "Error: Preprocessing failed. Please check the error messages above."
    exit 1
fi
