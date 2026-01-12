#!/bin/bash
# Split the combined dataset.csv into train/valid/test splits

echo "=========================================="
echo "Splitting Combined Dataset"
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

# Run split script
echo ""
echo "Splitting data/processed/dataset.csv into train/valid/test..."
echo ""

python3 scripts/split_dataset.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Dataset split complete!"
    echo "=========================================="
    echo ""
    echo "Generated files:"
    ls -lh data/processed/train.csv data/processed/valid.csv data/processed/test.csv
    echo ""
    echo "Next step: Translate captions to Nepali"
    echo "  python scripts/translate_captions.py --input data/processed/train.csv --output data/processed/train_nepali.csv"
else
    echo ""
    echo "Error: Dataset split failed. Please check the error messages above."
    exit 1
fi
