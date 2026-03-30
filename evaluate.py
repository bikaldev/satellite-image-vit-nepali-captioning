"""
Comprehensive evaluation script for classification and captioning models.
Generates detailed metrics, visualizations, and reports.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# Import local modules
from models.vit_classifier import ViTClassifier
from models.vit_captioner import ViTCaptioner
from data_loaders.dataset import SatelliteClassificationDataset, SatelliteCaptioningDataset, collate_fn_captioning
from data_loaders.transforms import get_val_transforms
from torch.utils.data import DataLoader
from functools import partial
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score


def evaluate_classifier(model, dataloader, device, class_names, output_dir):
    """Comprehensive classification evaluation."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []
    
    print("\nEvaluating classifier...")
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            # Get predictions
            outputs = model.predict(images, return_probs=True)
            
            all_preds.extend(outputs['predictions'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs['probs'].cpu().numpy())
            
            # Extract features for visualization
            with torch.no_grad():
                vit_outputs = model.vit(pixel_values=images)
                features = vit_outputs.last_hidden_state[:, 0].cpu().numpy()
                all_features.extend(features)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_features = np.array(all_features)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # ROC AUC (one-vs-rest)
    y_bin = label_binarize(all_labels, classes=range(len(class_names)))
    try:
        roc_auc = roc_auc_score(y_bin, all_probs, average='macro', multi_class='ovr')
    except:
        roc_auc = 0.0
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Per-class metrics
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': np.bincount(all_labels, minlength=len(class_names))
    })
    metrics_df.to_csv(output_dir / 'per_class_metrics.csv', index=False)
    
    # t-SNE visualization
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(all_features[:1000])  # Use subset for speed
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1],
                         c=all_labels[:1000], cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, ticks=range(len(class_names)), label='Class')
    plt.title('t-SNE Feature Space Visualization', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_space_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save summary
    summary = {
        'accuracy': float(accuracy),
        'macro_precision': float(np.mean(precision)),
        'macro_recall': float(np.mean(recall)),
        'macro_f1': float(np.mean(f1)),
        'roc_auc': float(roc_auc),
        'per_class_metrics': metrics_df.to_dict('records')
    }
    
    with open(output_dir / 'classification_report.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Classification Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {np.mean(f1):.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    return summary


def evaluate_captioner(model, dataloader, device, output_dir):
    """Comprehensive captioning evaluation."""
    model.eval()
    all_references = []
    all_hypotheses = []
    all_classes = []
    
    print("\nEvaluating captioner...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pixel_values = batch['pixel_values'].to(device)
            
            # Generate captions
            generated = model.generate_caption(pixel_values)
            
            all_references.extend(batch['captions'])
            all_hypotheses.extend(generated)
            all_classes.extend(batch['class_names'])
    
    # Compute BLEU scores
    refs_tokenized = [[ref.split()] for ref in all_references]
    hyps_tokenized = [hyp.split() for hyp in all_hypotheses]
    
    bleu1 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.25, 0.25, 0.25, 0.25))
    
    # Compute METEOR
    meteor_scores = []
    for ref, hyp in zip(all_references, all_hypotheses):
        try:
            score = meteor_score([ref.split()], hyp.split())
            meteor_scores.append(score)
        except:
            meteor_scores.append(0.0)
    meteor_avg = np.mean(meteor_scores)
    
    # Caption diversity
    unique_captions = len(set(all_hypotheses))
    diversity = unique_captions / len(all_hypotheses)
    
    # Save best and worst examples
    bleu_scores_individual = []
    for ref, hyp in zip(all_references, all_hypotheses):
        score = corpus_bleu([[ref.split()]], [hyp.split()])
        bleu_scores_individual.append(score)
    
    sorted_indices = np.argsort(bleu_scores_individual)
    
    best_examples = []
    for idx in sorted_indices[-10:]:
        best_examples.append({
            'class': all_classes[idx],
            'reference': all_references[idx],
            'generated': all_hypotheses[idx],
            'bleu': bleu_scores_individual[idx]
        })
    
    worst_examples = []
    for idx in sorted_indices[:10]:
        worst_examples.append({
            'class': all_classes[idx],
            'reference': all_references[idx],
            'generated': all_hypotheses[idx],
            'bleu': bleu_scores_individual[idx]
        })
    
    # Save examples
    with open(output_dir / 'best_captions.json', 'w', encoding='utf-8') as f:
        json.dump(best_examples, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / 'worst_captions.json', 'w', encoding='utf-8') as f:
        json.dump(worst_examples, f, ensure_ascii=False, indent=2)
    
    # Save summary
    summary = {
        'bleu1': float(bleu1),
        'bleu2': float(bleu2),
        'bleu3': float(bleu3),
        'bleu4': float(bleu4),
        'meteor': float(meteor_avg),
        'diversity': float(diversity),
        'total_captions': len(all_hypotheses)
    }
    
    with open(output_dir / 'captioning_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Captioning Results ===")
    print(f"BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}, BLEU-4: {bleu4:.4f}")
    print(f"METEOR: {meteor_avg:.4f}")
    print(f"Diversity: {diversity:.4f}")
    
    return summary


def main(args):
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = config['classes']['names_nepali']
    
    # Evaluate classification
    if args.task in ['classification', 'both']:
        print("\n" + "="*50)
        print("CLASSIFICATION EVALUATION")
        print("="*50)
        
        # Load model
        classifier, _ = ViTClassifier.load_checkpoint(args.classifier_path, device)
        
        # Load dataset
        dataset = SatelliteClassificationDataset(
            csv_file=f"{config['data']['processed_dir']}/{args.split}.csv",
            root_dir=config['data']['raw_dir'],
            transform=get_val_transforms(config['image']['size']),
            class_names_nepali=class_names
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        class_output_dir = output_dir / 'classification'
        class_output_dir.mkdir(exist_ok=True)
        
        evaluate_classifier(classifier, dataloader, device, class_names, class_output_dir)
    
    # Evaluate captioning
    if args.task in ['captioning', 'both']:
        print("\n" + "="*50)
        print("CAPTIONING EVALUATION")
        print("="*50)
        
        # Load model
        captioner, _ = ViTCaptioner.load_checkpoint(args.captioner_path, device)
        
        # Load dataset
        dataset = SatelliteCaptioningDataset(
            csv_file=f"{config['data']['processed_dir']}/{args.split}.csv",
            root_dir=config['data']['raw_dir'],
            transform=get_val_transforms(config['image']['size']),
            use_nepali_captions=True,
            class_names_nepali=class_names
        )
        
        collate_fn = partial(collate_fn_captioning, tokenizer=captioner.tokenizer, max_length=config['captioner']['max_length'])
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
        
        cap_output_dir = output_dir / 'captioning'
        cap_output_dir.mkdir(exist_ok=True)
        
        evaluate_captioner(captioner, dataloader, device, cap_output_dir)
    
    print(f"\n✓ Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate satellite image models")
    parser.add_argument('--task', type=str, choices=['classification', 'captioning', 'both'], required=True)
    parser.add_argument('--classifier_path', type=str, help='Path to classifier checkpoint')
    parser.add_argument('--captioner_path', type=str, help='Path to captioner checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation')
    
    args = parser.parse_args()
    main(args)
