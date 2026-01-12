import os
import subprocess

DATASETS = {
    "satellite-classification-captioning": {
        "kaggle_id":  "tomtillo/satellite-image-caption-generation",
        "target_dir": "data/raw"
    }
}

def download_dataset(kaggle_id, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    cmd = [
        "kaggle", "datasets", "download",
        "-d", kaggle_id,
        "-p", target_dir,
        "--unzip"
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    for cfg in DATASETS.values():
        download_dataset(cfg["kaggle_id"], cfg["target_dir"])
