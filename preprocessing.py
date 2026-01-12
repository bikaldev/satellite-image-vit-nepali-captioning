import os
import re
import ast

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoTokenizer


raw_data_dir = "/home/bikal/Documents/Satellite Image/data/raw"
data_dir = "/home/bikal/Documents/Satellite Image/data/processed/"

# the classes that are relevant for study

relevant_classes = ["airport","pond","mountain","farmland","river","residential","playground", "forest"]
relevant_classes_nep = ["विमानस्थल", "पोखरी", "पहाड", "खेतीयोग्य भूमि", "नदी", "आवासीय क्षेत्र", "खेळमैदान", "वन"]

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


# =========================
# 1. Preprocess CSV to add class column (safe)
# =========================
def add_class_column(csv_file, save_file=None):
    df = pd.read_csv(csv_file)

    def extract_class(filepath):
        fname = os.path.basename(filepath)  # e.g. sparseresidential_001.jpg
        match = re.match(r"([a-zA-Z]+)_[0-9]+\.jpg", fname)
        if match:
            return match.group(1).lower()
        return None  # invalid filename

    # Extract raw class names
    df['class'] = df['filepath'].apply(extract_class)

    # Drop invalid rows
    df = df.dropna(subset=['class']).reset_index(drop=True)

    # 🔑 Merge subclasses into one
    merge_map = {
        "sparseresidential": "residential",
        "denseresidential": "residential",
        "mediumresidential": "residential"
    }

    df['class'] = df['class'].replace(merge_map)

    filtered_train_df = df[df['class'].isin(relevant_classes)].reset_index(drop=True)
    filtered_train_df['class'] = filtered_train_df['class'].map(eng_to_nep)
    
    return filtered_train_df

# Clean datasets
train_df = add_class_column(raw_data_dir + "train.csv")
valid_df = add_class_column(raw_data_dir +"valid.csv")
test_df  = add_class_column(raw_data_dir +"test.csv")


train_df.to_csv(data_dir + "train.csv")
valid_df.to_csv(data_dir + "valid.csv")
test_df.to_csv(data_dir + "test.csv")