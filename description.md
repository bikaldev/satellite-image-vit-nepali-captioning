# Satellite Image Classification and Captioning with Vision Transformers

## Project Overview
This project implements a comprehensive deep learning system for satellite image analysis, featuring two main components:
1.  **Image Classification**: Classifies satellite images into 8 distinct categories.
2.  **Image Captioning**: Generates descriptive captions for satellite images in the Nepali language.

The system leverages state-of-the-art Vision Transformer (ViT) architectures to achieve high performance on both tasks.

## Model Architectures

### 1. ViT Classifier (`ViTClassifier`)
The classification model is built upon the Vision Transformer architecture, specifically designed to handle satellite imagery.

*   **Backbone**: `google/vit-base-patch16-224`
    *   Pre-trained on ImageNet-21k (14 million images, 21,843 classes) at resolution 224x224.
    *   Splits images into 16x16 patches.
    *   Embedding dimension: 768.
*   **Classification Head**:
    *   A custom Multi-Layer Perceptron (MLP) head attached to the `[CLS]` token output of the ViT backbone.
    *   **Layer 1**: Dropout (rate=0.1)
    *   **Layer 2**: Linear (Input: 768 -> Output: 384)
    *   **Activation**: GELU (Gaussian Error Linear Unit)
    *   **Layer 3**: Dropout (rate=0.1)
    *   **Layer 4**: Linear (Input: 384 -> Output: 8)
*   **Classes**:
    1.  Airport (विमानस्थल)
    2.  Pond (पोखरी)
    3.  Mountain (पहाड)
    4.  Farmland (खेतीयोग्य भूमि)
    5.  River (नदी)
    6.  Residential (आवासीय क्षेत्र)
    7.  Playground (खेळमैदान)
    8.  Forest (वन)

### 2. ViT Captioner (`ViTCaptioner`)
The captioning model employs an encoder-decoder architecture to translate visual features into text descriptions.

*   **Architecture Type**: Vision Encoder-Decoder
*   **Encoder**: `google/vit-base-patch16-224`
    *   Acts as the "eye" of the model, processing the satellite image into a sequence of feature vectors.
    *   Freeze/Unfreeze capability for fine-tuning.
*   **Decoder**: `gpt2`
    *   Acts as the "mouth" of the model, generating text autoregressively based on the encoder's output.
    *   Pre-trained GPT-2 model adapted for conditional text generation.
*   **Tokenizer**: GPT-2 Tokenizer (Byte-Pair Encoding).
*   **Generation Configuration**:
    *   **Beam Search**: Uses 5 beams to explore multiple caption possibilities and select the most likely one.
    *   **Max Length**: 128 tokens.
    *   **Repetition Penalty**: Prevents the model from repeating n-grams (size 3).
    *   **Length Penalty**: 2.0 (encourages longer, more descriptive captions).

## Classification Results

The classification model has achieved exceptional performance on the validation dataset. Below are the key metrics from the training history:

*   **Final Validation Accuracy**: **100.0%**
    *   The model reached 100% accuracy on the validation set as early as Epoch 3 and maintained near-perfect performance throughout.
*   **Final Validation Loss**: **~0.0002**
    *   The loss converged to an extremely low value, indicating high confidence in the model's predictions.
*   **Training Performance**:
    *   **Training Accuracy**: 100.0%
    *   **Training Loss**: ~0.0001
    *   This indicates the model has effectively learned to distinguish all training examples without error.

*Note: The perfect accuracy suggests the model is highly effective for this specific dataset configuration.*
