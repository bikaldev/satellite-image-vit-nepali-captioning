"""
Split the combined dataset.csv into train/valid/test splits.
This script reads the combined dataset and creates separate CSV files
based on the filepath column (train/, valid/, test/ prefixes).
"""

import pandas as pd
import os
from pathlib import Path

# Paths
data_dir = Path("data/processed")
input_file = data_dir / "dataset.csv"
output_dir = data_dir

# Read combined dataset
print(f"Reading combined dataset from {input_file}...")
df = pd.read_csv(input_file)

print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Check if class column exists, if not create it from filepath
if 'class' not in df.columns:
    print("\nWarning: 'class' column not found. Extracting from filepath...")
    import re
    def extract_class(filepath):
        fname = os.path.basename(filepath)
        match = re.match(r"([a-zA-Z]+)_[0-9]+\.jpg", fname)
        if match:
            return match.group(1).lower()
        return None
    df['class'] = df['filepath'].apply(extract_class)

# Define class mappings
relevant_classes = ["airport", "pond", "mountain", "farmland", "river", "residential", "playground", "forest"]
eng_to_nep = {
    "airport": "विमानस्थल",
    "pond": "पोखरी",
    "mountain": "पहाड",
    "farmland": "खेतीयोग्य भूमि",
    "river": "नदी",
    "residential": "आवासीय क्षेत्र",
    "playground": "खेळमैदान",
    "forest": "वन"
}

# Merge subclasses
merge_map = {
    "sparseresidential": "residential",
    "denseresidential": "residential",
    "mediumresidential": "residential"
}
df['class'] = df['class'].replace(merge_map)

# Filter to relevant classes only
print(f"\nFiltering to relevant classes: {relevant_classes}")
df = df[df['class'].isin(relevant_classes)].reset_index(drop=True)
print(f"Rows after filtering: {len(df)}")

# Convert class names to Nepali
print("\nConverting class names to Nepali...")
df['class'] = df['class'].map(eng_to_nep)

# Split based on filepath prefix
print("\nSplitting dataset based on filepath...")

train_df = df[df['filepath'].str.startswith('train/')].reset_index(drop=True)
valid_df = df[df['filepath'].str.startswith('valid/')].reset_index(drop=True)
test_df = df[df['filepath'].str.startswith('test/')].reset_index(drop=True)

print(f"Train samples: {len(train_df)}")
print(f"Valid samples: {len(valid_df)}")
print(f"Test samples: {len(test_df)}")

# Display class distribution
print("\n=== Class Distribution ===")
print("\nTrain:")
print(train_df['class'].value_counts())
print("\nValid:")
print(valid_df['class'].value_counts())
print("\nTest:")
print(test_df['class'].value_counts())

# Save splits
print(f"\n=== Saving splits to {output_dir} ===")

train_file = output_dir / "train.csv"
valid_file = output_dir / "valid.csv"
test_file = output_dir / "test.csv"

train_df.to_csv(train_file, index=False)
valid_df.to_csv(valid_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"✓ Saved {train_file} ({len(train_df)} rows)")
print(f"✓ Saved {valid_file} ({len(valid_df)} rows)")
print(f"✓ Saved {test_file} ({len(test_df)} rows)")

print("\n=== Split complete! ===")
print("\nNext step: Translate captions to Nepali")
print("  python scripts/translate_captions.py --input data/processed/train.csv --output data/processed/train_nepali.csv")
