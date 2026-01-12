# Satellite Image Classification and Captioning with Vision Transformers

A comprehensive deep learning project for satellite image analysis using Vision Transformers (ViT). The system performs both classification and captioning of satellite images with full Nepali language support.

## 🌟 Features

- **Dual-Task Learning**: Classification (8 classes) + Image Captioning
- **Vision Transformer Architecture**: State-of-the-art ViT models for both tasks
- **Nepali Language Support**: Full Nepali captions using offline IndicTrans2 translation
- **Comprehensive Analysis**: Extensive preprocessing analysis, training monitoring, and evaluation
- **Production-Ready**: Complete inference pipeline with visualization

## 📋 Classes

The system classifies satellite images into 8 categories:

| English | Nepali (नेपाली) |
|---------|----------------|
| Airport | विमानस्थल |
| Pond | पोखरी |
| Mountain | पहाड |
| Farmland | खेतीयोग्य भूमि |
| River | नदी |
| Residential | आवासीय क्षेत्र |
| Playground | खेळमैदान |
| Forest | वन |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd "/home/bikal/Documents/Satellite Image"

# Create virtual environment (if not already created)
python -m venv env
source env/bin/activate  # On Linux/Mac
# or
env\Scripts\activate  # On Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install IndicTrans2 from GitHub (for offline translation)
pip install git+https://github.com/AI4Bharat/IndicTrans2.git
```

**Or use the setup script:**
```bash
./setup.sh
```

### Data Preparation

1. **Split Combined Dataset** (Create train/valid/test CSV files):
```bash
./run_split_dataset.sh
```

This will:
- Read `data/processed/dataset.csv`
- Filter to 8 relevant classes
- Convert class labels to Nepali
- Split into train/valid/test based on filepath
- Create:
  - `data/processed/train.csv`
  - `data/processed/valid.csv`
  - `data/processed/test.csv`

2. **Translate Captions to Nepali** (Offline using IndicTrans2):
```bash
# Activate virtual environment
source env/bin/activate

# Set your HuggingFace token
export HF_TOKEN="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"

# Translate each split
python scripts/translate_captions.py \
    --input data/processed/train.csv \
    --output data/processed/train_nepali.csv \
    --hf_token $HF_TOKEN

python scripts/translate_captions.py \
    --input data/processed/valid.csv \
    --output data/processed/valid_nepali.csv \
    --hf_token $HF_TOKEN

python scripts/translate_captions.py \
    --input data/processed/test.csv \
    --output data/processed/test_nepali.csv \
    --hf_token $HF_TOKEN
```

**Note**: Replace the token with your own HuggingFace access token if needed.

## 🎯 Training

### Train Classification Model

```bash
python train_classifier.py \
    --config configs/config.yaml \
    --output_dir outputs/classification
```

**Features**:
- Mixed precision training
- Learning rate scheduling
- Real-time confusion matrices
- Per-epoch sample predictions
- TensorBoard logging

### Train Captioning Model

```bash
python train_captioner.py \
    --config configs/config.yaml \
    --output_dir outputs/captioning
```

**Features**:
- BLEU/METEOR score tracking
- Sample caption generation per epoch
- Beam search decoding
- Class-conditioned captions

## 📊 Evaluation

### Evaluate Both Models

```bash
python evaluate.py \
    --task both \
    --classifier_path outputs/classification/checkpoints/best_model.pth \
    --captioner_path outputs/captioning/checkpoints/best_bleu_model \
    --split test \
    --output_dir outputs/evaluation
```

**Outputs**:
- Confusion matrix with Nepali labels
- Per-class precision/recall/F1
- ROC curves and AUC scores
- t-SNE feature visualization
- BLEU-1/2/3/4, METEOR, diversity metrics
- Best/worst caption examples

## 🔮 Inference

### Single Image Analysis

```bash
python inference.py \
    --classifier_path outputs/classification/checkpoints/best_model.pth \
    --captioner_path outputs/captioning/checkpoints/best_bleu_model \
    --image data/raw/test/airport_1.jpg \
    --output results/prediction.json \
    --visualize
```

### Batch Processing

```bash
python inference.py \
    --classifier_path outputs/classification/checkpoints/best_model.pth \
    --captioner_path outputs/captioning/checkpoints/best_bleu_model \
    --image_dir data/raw/test/ \
    --batch_size 16 \
    --output results/batch_predictions.json
```

## 📁 Project Structure

```
.
├── configs/
│   └── config.yaml              # Hyperparameters and settings
├── data/
│   ├── raw/                     # Original images
│   └── processed/               # Processed CSVs with Nepali captions
├── datasets/
│   ├── dataset.py               # PyTorch datasets
│   └── transforms.py            # Image augmentation
├── models/
│   ├── vit_classifier.py        # ViT classification model
│   └── vit_captioner.py         # ViT captioning model
├── scripts/
│   └── translate_captions.py    # Offline translation pipeline
├── utils/
│   ├── text_preprocessing.py    # Nepali text utilities
│   └── visualize.py             # Visualization functions
├── outputs/                     # All training/evaluation outputs
├── train_classifier.py          # Classification training script
├── train_captioner.py           # Captioning training script
├── evaluate.py                  # Comprehensive evaluation
├── inference.py                 # End-to-end inference pipeline
└── requirements.txt             # Dependencies
```

## 📈 Monitoring

### TensorBoard

```bash
tensorboard --logdir outputs/classification/logs
tensorboard --logdir outputs/captioning/logs
```

### Output Artifacts

All training and evaluation artifacts are saved in `outputs/`:
- Training curves (loss, accuracy, BLEU scores)
- Confusion matrices per epoch
- Sample predictions
- Model checkpoints
- Evaluation reports (JSON/CSV)
- Visualizations (PNG/SVG)

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:
- Model architectures
- Training hyperparameters
- Data augmentation strategies
- Evaluation metrics
- Device settings (CPU/GPU)

## 🛠️ Key Technologies

- **PyTorch**: Deep learning framework
- **Transformers (HuggingFace)**: ViT models
- **IndicTrans2**: Offline English→Nepali translation
- **TensorBoard**: Training visualization
- **NLTK**: Caption evaluation metrics
- **Scikit-learn**: Classification metrics

## 📝 Citation

If you use this project, please cite:

```bibtex
@software{satellite_vit_captioning,
  title={Satellite Image Classification and Captioning with Vision Transformers},
  author={Your Name},
  year={2026},
  description={ViT-based satellite image analysis with Nepali language support}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project uses offline translation (IndicTrans2) for full Nepali language support without requiring internet connectivity after initial model download.
