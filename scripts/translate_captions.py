"""
Translation pipeline using IndicTrans2 for English to Nepali translation.
Fully offline capable after initial model download.
"""

import os
import sys
import argparse
import pandas as pd
import torch
from tqdm import tqdm
import json
from pathlib import Path

# IndicTrans2 imports
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class IndicTrans2Translator:
    """
    Offline English to Nepali translator using IndicTrans2.
    """
    
    def __init__(self, cache_dir="outputs/translation/cache", device="cuda", hf_token=None):
        """
        Initialize the IndicTrans2 translator.
        
        Args:
            cache_dir: Directory to cache translated captions
            device: Device to run translation on ('cuda' or 'cpu')
            hf_token: HuggingFace access token for gated models
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "translation_cache.json"
        self.hf_token = hf_token
        
        # Load cache if exists
        self.cache = self._load_cache()
        
        print(f"Loading IndicTrans2 model (en->indic)...")
        # Model for English to Indic languages
        model_name = "ai4bharat/indictrans2-en-indic-1B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=self.hf_token
        )
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=self.hf_token
        ).to(self.device)
        
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
        
        # Load back-translation model for quality check
        print("Loading back-translation model (indic->en) for quality validation...")
        back_model_name = "ai4bharat/indictrans2-indic-en-1B"
        
        self.back_tokenizer = AutoTokenizer.from_pretrained(
            back_model_name,
            trust_remote_code=True,
            token=self.hf_token
        )
        
        self.back_model = AutoModelForSeq2SeqLM.from_pretrained(
            back_model_name,
            trust_remote_code=True,
            token=self.hf_token
        ).to(self.device)
        
        self.back_model.eval()
        print("Back-translation model loaded successfully")
    
    def _load_cache(self):
        """Load translation cache from disk."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save translation cache to disk."""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def translate_batch(self, texts, batch_size=32):
        """
        Translate a batch of English texts to Nepali.
        
        Args:
            texts: List of English texts
            batch_size: Batch size for translation
            
        Returns:
            List of Nepali translations
        """
        translations = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i:i+batch_size]
            
            # Check cache first
            cached_translations = []
            uncached_texts = []
            uncached_indices = []
            
            for idx, text in enumerate(batch):
                if text in self.cache:
                    cached_translations.append((idx, self.cache[text]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(idx)
            
            # Translate uncached texts
            if uncached_texts:
                # Prepare input with language tags
                inputs = [f"<2npi> {text}" for text in uncached_texts]
                
                # Tokenize
                encoded = self.tokenizer(
                    inputs,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate translations
                with torch.no_grad():
                    generated = self.model.generate(
                        **encoded,
                        max_length=256,
                        num_beams=5,
                        early_stopping=True
                    )
                
                # Decode
                batch_translations = self.tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True
                )
                
                # Update cache
                for text, translation in zip(uncached_texts, batch_translations):
                    self.cache[text] = translation
            else:
                batch_translations = []
            
            # Combine cached and new translations in correct order
            result_batch = [None] * len(batch)
            for idx, trans in cached_translations:
                result_batch[idx] = trans
            for idx, trans in zip(uncached_indices, batch_translations):
                result_batch[idx] = trans
            
            translations.extend(result_batch)
        
        # Save cache periodically
        self._save_cache()
        
        return translations
    
    def back_translate(self, nepali_texts, batch_size=32):
        """
        Back-translate Nepali texts to English for quality validation.
        
        Args:
            nepali_texts: List of Nepali texts
            batch_size: Batch size for back-translation
            
        Returns:
            List of English back-translations
        """
        back_translations = []
        
        for i in tqdm(range(0, len(nepali_texts), batch_size), desc="Back-translating"):
            batch = nepali_texts[i:i+batch_size]
            
            # Prepare input with language tags
            inputs = [f"<2en> {text}" for text in batch]
            
            # Tokenize
            encoded = self.back_tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate back-translations
            with torch.no_grad():
                generated = self.back_model.generate(
                    **encoded,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode
            batch_back_translations = self.back_tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )
            
            back_translations.extend(batch_back_translations)
        
        return back_translations


def parse_caption_list(caption_str):
    """Parse caption string (list format) to extract individual captions."""
    import ast
    try:
        captions = ast.literal_eval(caption_str)
        if isinstance(captions, list):
            return captions
        return [str(caption_str)]
    except:
        return [str(caption_str)]


def translate_dataset(input_csv, output_csv, sample_size=None, batch_size=32, hf_token=None):
    """
    Translate captions in dataset from English to Nepali.
    
    Args:
        input_csv: Path to input CSV with English captions
        output_csv: Path to output CSV with bilingual captions
        sample_size: If set, only translate first N samples (for testing)
        batch_size: Batch size for translation
        hf_token: HuggingFace access token for gated models
    """
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if sample_size:
        print(f"Using sample of {sample_size} rows for testing")
        df = df.head(sample_size)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Initialize translator with HF token
    translator = IndicTrans2Translator(hf_token=hf_token)
    
    # Extract all unique captions
    all_captions_english = []
    for caption_str in df['captions']:
        captions = parse_caption_list(caption_str)
        all_captions_english.extend(captions)
    
    unique_captions = list(set(all_captions_english))
    print(f"\nTotal unique English captions: {len(unique_captions)}")
    
    # Translate all unique captions
    print("\n=== Translating English captions to Nepali ===")
    nepali_translations = translator.translate_batch(unique_captions, batch_size=batch_size)
    
    # Create translation dictionary
    translation_dict = dict(zip(unique_captions, nepali_translations))
    
    # Apply translations to dataframe
    print("\n=== Applying translations to dataset ===")
    nepali_captions_list = []
    
    for caption_str in tqdm(df['captions'], desc="Processing rows"):
        captions = parse_caption_list(caption_str)
        nepali_captions = [translation_dict[cap] for cap in captions]
        nepali_captions_list.append(str(nepali_captions))
    
    df['captions_nepali'] = nepali_captions_list
    
    # Quality validation: back-translate a sample
    print("\n=== Quality Validation (Back-translation) ===")
    sample_indices = list(range(min(10, len(unique_captions))))
    sample_english = [unique_captions[i] for i in sample_indices]
    sample_nepali = [nepali_translations[i] for i in sample_indices]
    
    back_translations = translator.back_translate(sample_nepali, batch_size=batch_size)
    
    print("\nSample translations:")
    for i, (eng, nep, back) in enumerate(zip(sample_english, sample_nepali, back_translations)):
        print(f"\n{i+1}. Original (EN): {eng}")
        print(f"   Translated (NP): {nep}")
        print(f"   Back-trans (EN): {back}")
    
    # Save bilingual dataset
    print(f"\n=== Saving bilingual dataset to {output_csv} ===")
    df.to_csv(output_csv, index=False)
    
    # Save translation statistics
    stats = {
        "total_rows": len(df),
        "unique_captions": len(unique_captions),
        "sample_translations": [
            {
                "english": eng,
                "nepali": nep,
                "back_translation": back
            }
            for eng, nep, back in zip(sample_english, sample_nepali, back_translations)
        ]
    }
    
    stats_file = Path(output_csv).parent / "translation_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"Translation statistics saved to {stats_file}")
    print("\n=== Translation complete! ===")


def main():
    parser = argparse.ArgumentParser(description="Translate satellite image captions from English to Nepali")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file with English captions")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file for bilingual captions")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for testing (optional)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for translation")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace access token for gated models")
    
    args = parser.parse_args()
    
    translate_dataset(
        input_csv=args.input,
        output_csv=args.output,
        sample_size=args.sample,
        batch_size=args.batch_size,
        hf_token=args.hf_token
    )


if __name__ == "__main__":
    main()
