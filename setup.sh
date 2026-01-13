#!/bin/bash
# Quick start script for the Satellite Image Classification and Captioning project

echo "=========================================="
echo "Satellite Image Analysis Project Setup"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install openpyxl  # Needed for Excel files

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Split the Nepali Excel dataset:"
echo "   ./run_split_dataset.sh"
echo ""
echo "2. Train classification model:"
echo "   python train_classifier.py --config configs/config.yaml --output_dir outputs/classification"
echo ""
echo "3. Train captioning model:"
echo "   python train_captioner.py --config configs/config.yaml --output_dir outputs/captioning"
echo ""
echo "4. Evaluate models:"
echo "   python evaluate.py --task both --classifier_path outputs/classification/checkpoints/best_model.pth --captioner_path outputs/captioning/checkpoints/best_bleu_model --output_dir outputs/evaluation"
echo ""
echo "5. Run inference:"
echo "   python inference.py --classifier_path outputs/classification/checkpoints/best_model.pth --captioner_path outputs/captioning/checkpoints/best_bleu_model --image data/raw/test/airport_1.jpg --visualize"
echo ""
