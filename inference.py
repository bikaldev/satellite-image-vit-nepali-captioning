"""
End-to-end inference pipeline for satellite image analysis.
Chains classification → captioning for complete image understanding.
"""

import os
import sys
import argparse
import yaml
import torch
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Union

# Import local modules
from models.vit_classifier import ViTClassifier
from models.vit_captioner import ViTCaptioner
from data_loaders.transforms import get_val_transforms, denormalize
import numpy as np


class SatelliteImageAnalyzer:
    """
    End-to-end satellite image analyzer.
    Performs classification and captioning in a single pipeline.
    """
    
    def __init__(
        self,
        classifier_path: str,
        captioner_path: str,
        config_path: str = 'configs/config.yaml',
        device: str = 'cuda'
    ):
        """
        Initialize the analyzer.
        
        Args:
            classifier_path: Path to trained classifier checkpoint
            captioner_path: Path to trained captioner checkpoint
            config_path: Path to config file
            device: Device to run on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load class names
        self.class_names_nepali = self.config['classes']['names_nepali']
        self.class_names_english = self.config['classes']['names_english']
        
        # Load models
        print("Loading classification model...")
        self.classifier, _ = ViTClassifier.load_checkpoint(classifier_path, self.device)
        self.classifier.eval()
        
        print("Loading captioning model...")
        self.captioner, _ = ViTCaptioner.load_checkpoint(captioner_path, self.device)
        self.captioner.eval()
        
        # Image transforms
        self.transform = get_val_transforms(self.config['image']['size'])
        
        print(f"Analyzer initialized on {self.device}")
    
    def analyze_image(
        self,
        image_path: str,
        return_english: bool = True
    ) -> Dict:
        """
        Analyze a single image.
        
        Args:
            image_path: Path to image
            return_english: Whether to include English translations
            
        Returns:
            Dictionary with analysis results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Classification
        with torch.no_grad():
            class_output = self.classifier.predict(image_tensor, return_probs=True)
            pred_idx = class_output['predictions'][0].item()
            confidence = class_output['probs'][0][pred_idx].item()
        
        class_name_nepali = self.class_names_nepali[pred_idx]
        class_name_english = self.class_names_english[pred_idx]
        
        # Captioning (conditioned on predicted class)
        with torch.no_grad():
            captions = self.captioner.generate_caption(
                image_tensor,
                class_label=class_name_nepali
            )
        
        caption_nepali = captions[0]
        
        # Prepare result
        result = {
            'image_path': image_path,
            'classification': {
                'class_nepali': class_name_nepali,
                'class_english': class_name_english,
                'confidence': confidence,
                'all_probs': class_output['probs'][0].cpu().numpy().tolist()
            },
            'caption': {
                'nepali': caption_nepali
            }
        }
        
        return result
    
    def analyze_batch(
        self,
        image_paths: List[str],
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Analyze multiple images in batches.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            for path in batch_paths:
                result = self.analyze_image(path)
                results.append(result)
        
        return results
    
    def visualize_result(
        self,
        image_path: str,
        result: Dict = None,
        save_path: str = None
    ):
        """
        Visualize analysis result.
        
        Args:
            image_path: Path to image
            result: Analysis result (if None, will analyze image)
            save_path: Optional path to save visualization
        """
        if result is None:
            result = self.analyze_image(image_path)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Display image
        ax.imshow(image)
        ax.axis('off')
        
        # Add text overlay
        class_nep = result['classification']['class_nepali']
        class_eng = result['classification']['class_english']
        confidence = result['classification']['confidence']
        caption = result['caption']['nepali']
        
        text = f"Class: {class_nep} ({class_eng})\n"
        text += f"Confidence: {confidence:.2%}\n\n"
        text += f"Caption:\n{caption}"
        
        # Add text box
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props, family='sans-serif')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


def main(args):
    # Initialize analyzer
    analyzer = SatelliteImageAnalyzer(
        classifier_path=args.classifier_path,
        captioner_path=args.captioner_path,
        config_path=args.config
    )
    
    # Single image or batch
    if args.image:
        # Single image
        print(f"\nAnalyzing image: {args.image}")
        result = analyzer.analyze_image(args.image)
        
        # Print result
        print("\n=== Analysis Result ===")
        print(f"Class (Nepali): {result['classification']['class_nepali']}")
        print(f"Class (English): {result['classification']['class_english']}")
        print(f"Confidence: {result['classification']['confidence']:.2%}")
        print(f"\nCaption (Nepali):\n{result['caption']['nepali']}")
        
        # Save result
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\nResult saved to {args.output}")
        
        # Visualize
        if args.visualize:
            viz_path = args.output.replace('.json', '.png') if args.output else None
            analyzer.visualize_result(args.image, result, save_path=viz_path)
    
    elif args.image_dir:
        # Batch processing
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        print(f"\nAnalyzing {len(image_paths)} images from {args.image_dir}")
        results = analyzer.analyze_batch(image_paths, batch_size=args.batch_size)
        
        # Save results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {args.output}")
        
        print(f"\nProcessed {len(results)} images")
    
    else:
        print("Error: Please specify either --image or --image_dir")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite image inference pipeline")
    parser.add_argument('--classifier_path', type=str, required=True, help='Path to classifier checkpoint')
    parser.add_argument('--captioner_path', type=str, required=True, help='Path to captioner checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--image_dir', type=str, help='Directory of images for batch processing')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for batch processing')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--visualize', action='store_true', help='Visualize result')
    
    args = parser.parse_args()
    main(args)
