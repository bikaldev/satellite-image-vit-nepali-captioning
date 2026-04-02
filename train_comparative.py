import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from functools import partial

# Import comparative models
from models.comparative_models import CNNClassifier, CNNLSTMCaptioner, CNNTransformerCaptioner
from data_loaders.dataset import SatelliteClassificationDataset, SatelliteCaptioningDataset, collate_fn_captioning
from data_loaders.transforms import get_train_transforms, get_val_transforms
import nltk

# Download required NLTK data
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def plot_metrics(history, save_path, title):
    """Plot training metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy or BLEU plot
    metric = 'val_acc' if 'val_acc' in history else 'bleu4'
    metric_label = 'Accuracy' if 'val_acc' in history else 'BLEU-4'
    
    if metric in history:
        axes[1].plot(history[metric], label=f'Val {metric_label}', color='green')
        axes[1].set_title(f'{title} - {metric_label}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_label)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_classification(model, train_loader, val_loader, config, device, model_name, output_dir):
    """Training loop for classification models."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    
    for epoch in range(config['train_classifier']['epochs']):
        model.train()
        running_loss = 0.0
        for images, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_checkpoint(output_dir / f'{model_name}_best.pth', epoch, {'accuracy': best_acc})
            
    plot_metrics(history, output_dir / f'{model_name}_training.png', model_name.upper())
    return history

def compute_caption_metrics(references, hypotheses):
    """Compute BLEU 1-4, METEOR, ROUGE-L."""
    refs_tokenized = [[ref.split()] for ref in references]
    hyps_tokenized = [hyp.split() for hyp in hypotheses]
    
    bleu4 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.25, 0.25, 0.25, 0.25))
    meteor_scores = []
    for r, h in zip(references, hypotheses):
        try: meteor_scores.append(meteor_score([r.split()], h.split()))
        except: meteor_scores.append(0.0)
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(r, h)['rougeL'].fmeasure for r, h in zip(references, hypotheses)]
    
    return {'bleu4': bleu4, 'meteor': np.mean(meteor_scores), 'rougeL': np.mean(rouge_scores)}

def train_captioning(model, train_loader, val_loader, config, device, model_name, output_dir):
    """Training loop for captioning models."""
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    history = {'train_loss': [], 'val_loss': [], 'bleu4': []}
    best_bleu = 0.0
    
    for epoch in range(config['train_captioner']['epochs']):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]'):
            images = batch['pixel_values'].to(device)
            captions = batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model(images, captions)
            # Shift for next-token prediction
            loss = criterion(logits[:, :-1, :].reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        all_refs, all_hyps = [], []
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch in val_loader:
                images = batch['pixel_values'].to(device)
                generated = model.generate_caption(images)
                all_refs.extend(batch['captions'])
                all_hyps.extend(generated)
        
        metrics = compute_caption_metrics(all_refs, all_hyps)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(0.0) # Placeholder or calculate if needed
        history['bleu4'].append(metrics['bleu4'])
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, BLEU-4: {metrics['bleu4']:.4f}")
        
        if metrics['bleu4'] > best_bleu:
            best_bleu = metrics['bleu4']
            model.save_checkpoint(output_dir / f'{model_name}_best', epoch, metrics)
            
    plot_metrics(history, output_dir / f'{model_name}_training.png', model_name.upper())
    return history

def main(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = config['classes']['names_nepali']
    tokenizer_name = config['captioner']['decoder_model'] # Use same as ViT for comparison (gpt2/bloom)
    
    # 1. ResNet50 Classifier
    if 'resnet50_clf' in args.models:
        print("\n>>> Training ResNet50 Classifier...")
        model = CNNClassifier(model_type='resnet50', num_classes=len(class_names)).to(device)
        ds = SatelliteClassificationDataset(f"{config['data']['processed_dir']}/train.csv", config['data']['raw_dir'], get_train_transforms(config['image']['size']), class_names)
        val_ds = SatelliteClassificationDataset(f"{config['data']['processed_dir']}/valid.csv", config['data']['raw_dir'], get_val_transforms(config['image']['size']), class_names)
        train_loader = DataLoader(ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        train_classification(model, train_loader, val_loader, config, device, 'resnet50_clf', output_dir)

    # 2. VGG16 Classifier
    if 'vgg16_clf' in args.models:
        print("\n>>> Training VGG16 Classifier...")
        model = CNNClassifier(model_type='vgg16', num_classes=len(class_names)).to(device)
        ds = SatelliteClassificationDataset(f"{config['data']['processed_dir']}/train.csv", config['data']['raw_dir'], get_train_transforms(config['image']['size']), class_names)
        val_ds = SatelliteClassificationDataset(f"{config['data']['processed_dir']}/valid.csv", config['data']['raw_dir'], get_val_transforms(config['image']['size']), class_names)
        train_loader = DataLoader(ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        train_classification(model, train_loader, val_loader, config, device, 'vgg16_clf', output_dir)

    # Captioning setup
    from functools import partial
    
    # 3. ResNet50 + LSTM
    if 'resnet50_lstm' in args.models:
        print("\n>>> Training ResNet50 + LSTM Captioner...")
        model = CNNLSTMCaptioner('resnet50', tokenizer_name, device=device).to(device)
        collate_fn = partial(collate_fn_captioning, tokenizer=model.tokenizer, max_length=128)
        ds = SatelliteCaptioningDataset(f"{config['data']['processed_dir']}/train.csv", config['data']['raw_dir'], get_train_transforms(config['image']['size']), True, class_names)
        val_ds = SatelliteCaptioningDataset(f"{config['data']['processed_dir']}/valid.csv", config['data']['raw_dir'], get_val_transforms(config['image']['size']), True, class_names)
        train_loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_fn)
        train_captioning(model, train_loader, val_loader, config, device, 'resnet50_lstm', output_dir)

    # 4. VGG16 + LSTM
    if 'vgg16_lstm' in args.models:
        print("\n>>> Training VGG16 + LSTM Captioner...")
        model = CNNLSTMCaptioner('vgg16', tokenizer_name, device=device).to(device)
        collate_fn = partial(collate_fn_captioning, tokenizer=model.tokenizer, max_length=128)
        ds = SatelliteCaptioningDataset(f"{config['data']['processed_dir']}/train.csv", config['data']['raw_dir'], get_train_transforms(config['image']['size']), True, class_names)
        val_ds = SatelliteCaptioningDataset(f"{config['data']['processed_dir']}/valid.csv", config['data']['raw_dir'], get_val_transforms(config['image']['size']), True, class_names)
        train_loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_fn)
        train_captioning(model, train_loader, val_loader, config, device, 'vgg16_lstm', output_dir)

    # 5. ResNet50 + mGPT
    if 'resnet50_mgpt' in args.models:
        from models.comparative_models import ResNetmGPTCaptioner
        print("\n>>> Training ResNet50 + mGPT Captioner...")
        model = ResNetmGPTCaptioner(tokenizer_name, device=device).to(device)
        collate_fn = partial(collate_fn_captioning, tokenizer=model.tokenizer, max_length=128)
        ds = SatelliteCaptioningDataset(f"{config['data']['processed_dir']}/train.csv", config['data']['raw_dir'], get_train_transforms(config['image']['size']), True, class_names)
        val_ds = SatelliteCaptioningDataset(f"{config['data']['processed_dir']}/valid.csv", config['data']['raw_dir'], get_val_transforms(config['image']['size']), True, class_names)
        train_loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn) # Reduced batch size for mGPT
        val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collate_fn)
        train_captioning(model, train_loader, val_loader, config, device, 'resnet50_mgpt', output_dir)

    # Store final summary
    print("\n>>> Generating Comparative Summary...")
    final_summary = {}
    for model_path in output_dir.glob('*_best*'):
        try:
            checkpoint = torch.load(model_path if model_path.is_file() else model_path / 'checkpoint.pt', map_location='cpu', weights_only=False)
            final_summary[model_path.name] = checkpoint.get('metrics', {})
        except:
            continue
            
    with open(output_dir / 'benchmarks_summary.json', 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n[DONE] Results and summary saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['resnet50_clf', 'vgg16_clf', 'resnet50_lstm', 'vgg16_lstm', 'resnet50_mgpt'])
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--output_dir', type=str, default='outputs/comparative_results')
    args = parser.parse_args()
    main(args)
