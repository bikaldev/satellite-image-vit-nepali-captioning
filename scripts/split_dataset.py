import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import os
import ast

def clean_captions(cap_str):
    """
    Cleans the numpy-like string representation of captions into a proper Python list string.
    Example input: "['विमानस्थलमा ...'\n 'एयरपोर्टमा ...']"
    """
    if not isinstance(cap_str, str):
        return str([str(cap_str)])
    
    # Remove outer brackets
    s = cap_str.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    
    # Extract text between quotes
    # Handles both ' and " cases, and ignores newlines/spaces between
    captions = re.findall(r"'(.*?)'", s, re.DOTALL)
    if not captions:
        captions = re.findall(r'"(.*?)"', s, re.DOTALL)
    
    if not captions:
        return str([s])
    
    # Remove newlines and extra spaces within each caption
    captions = [c.replace('\n', ' ').strip() for c in captions]
    
    return str(captions)

def main():
    input_file = 'data/raw/newdataset_nepali .xlsx'
    output_dir = 'data/processed'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file)
    
    # Class name mapping (English to Nepali)
    class_map = {
        'airport': 'विमानस्थल',
        'pond': 'पोखरी',
        'mountain': 'पहाड',
        'farmland': 'खेतीयोग्य भूमि',
        'river': 'नदी',
        'residential': 'आवासीय क्षेत्र',
        'playground': 'खेळमैदान',
        'forest': 'वन'
    }
    
    if 'class' in df.columns:
        print("Mapping class names to Nepali...")
        df['class'] = df['class'].map(class_map)
    
    # Clean captions and captions_nepali columns if they exist
    if 'captions_nepali' in df.columns:
        df['captions_nepali'] = df['captions_nepali'].apply(clean_captions)
    
    if 'captions' in df.columns:
        df['captions'] = df['captions'].apply(clean_captions)
    
    # Remove Unnamed: 0 if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Split: 80% train, 10% val, 10% test
    print("Splitting dataset...")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'] if 'class' in df.columns else None)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['class'] if 'class' in temp_df.columns else None)
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Save to CSV
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Successfully saved splits to {output_dir}")

if __name__ == "__main__":
    main()
