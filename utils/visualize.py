"""
Comprehensive visualization utilities for satellite image analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from wordcloud import WordCloud
import pandas as pd


def plot_class_distribution(df, class_column='class', save_path=None):
    """Plot class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    class_counts = df[class_column].value_counts()
    axes[0].bar(range(len(class_counts)), class_counts.values, color='skyblue', edgecolor='black')
    axes[0].set_xticks(range(len(class_counts)))
    axes[0].set_xticklabels(class_counts.index, rotation=45, ha='right')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Class Distribution (Bar Plot)')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette('pastel'))
    axes[1].set_title('Class Distribution (Pie Chart)')
    
    plt.suptitle('Dataset Class Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class distribution saved to {save_path}")
    
    plt.show()


def plot_image_statistics(image_sizes, save_path=None):
    """Plot image size statistics."""
    widths = [size[0] for size in image_sizes]
    heights = [size[1] for size in image_sizes]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Width distribution
    axes[0, 0].hist(widths, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Image Width Distribution')
    axes[0, 0].grid(alpha=0.3)
    
    # Height distribution
    axes[0, 1].hist(heights, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Image Height Distribution')
    axes[0, 1].grid(alpha=0.3)
    
    # Aspect ratio
    aspect_ratios = [w/h for w, h in zip(widths, heights)]
    axes[1, 0].hist(aspect_ratios, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Aspect Ratio (W/H)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Aspect Ratio Distribution')
    axes[1, 0].grid(alpha=0.3)
    
    # Summary statistics
    stats_text = f"""
    Width:
    - Mean: {np.mean(widths):.1f}
    - Std: {np.std(widths):.1f}
    - Min: {np.min(widths)}
    - Max: {np.max(widths)}
    
    Height:
    - Mean: {np.mean(heights):.1f}
    - Std: {np.std(heights):.1f}
    - Min: {np.min(heights)}
    - Max: {np.max(heights)}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.suptitle('Image Statistics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Image statistics saved to {save_path}")
    
    plt.show()


def create_nepali_wordcloud(texts, save_path=None):
    """Create word cloud from Nepali text."""
    # Combine all texts
    combined_text = ' '.join(texts)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='viridis',
        font_path=None,  # Use default font (may not render Devanagari perfectly)
        max_words=100
    ).generate(combined_text)
    
    plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nepali Caption Word Cloud', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Word cloud saved to {save_path}")
    
    plt.show()


def plot_sample_grid(images, labels, predictions=None, n_samples=16, save_path=None):
    """Plot grid of sample images with labels."""
    n = min(n_samples, len(images))
    grid_size = int(np.ceil(np.sqrt(n)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(n):
        axes[i].imshow(images[i])
        
        title = f"GT: {labels[i]}"
        if predictions is not None:
            title += f"\nPred: {predictions[i]}"
            color = 'green' if labels[i] == predictions[i] else 'red'
        else:
            color = 'black'
        
        axes[i].set_title(title, fontsize=8, color=color)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample grid saved to {save_path}")
    
    plt.show()


def plot_roc_curves(y_true, y_probs, class_names, save_path=None):
    """Plot ROC curves for multi-class classification."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # Binarize labels
    y_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    plt.figure(figsize=(12, 10))
    
    # Plot ROC curve for each class
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, linewidth=2, 
                label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    print("Available functions:")
    print("- plot_class_distribution()")
    print("- plot_image_statistics()")
    print("- create_nepali_wordcloud()")
    print("- plot_sample_grid()")
    print("- plot_roc_curves()")
