import os
import cv2
import pandas as pd
import numpy as np

# Paths
RAW_IMAGE_DIR = "data/raw/PlantVillage/"
PROCESSED_IMAGE_DIR = "data/processed/images/"
RAW_TABULAR = "data/raw/yield_prediction.csv"
PROCESSED_TABULAR = "data/processed/tabular.csv"

# Define crop types and damage mapping
CROP_TYPES = ["Tomato", "Corn", "Potato"]
DAMAGE_MAP = {
    "healthy": 0.0,           # 0% destruction
    "bacterial_spot": 0.5,    # 50% destruction (example)
    "late_blight": 0.8,       # 80% destruction (example)
    "gray_leaf_spot": 0.6,    # 60% destruction (example)
    "common_rust": 0.7,       # 70% destruction (example)
    # Add more disease-to-damage mappings based on PlantVillage folder names
}

def preprocess_images():
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
    data = []
    for folder in os.listdir(RAW_IMAGE_DIR):
        # Extract crop type
        crop_type = next((c for c in CROP_TYPES if c.lower() in folder.lower()), None)
        if not crop_type:
            continue
        
        # Extract damage percentage based on disease
        damage = next((v for k, v in DAMAGE_MAP.items() if k.lower() in folder.lower()), 0.0)
        
        folder_path = os.path.join(RAW_IMAGE_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
            
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (224, 224))  # VGG16 input size
            new_path = os.path.join(PROCESSED_IMAGE_DIR, f"{crop_type}_{damage}_{img_file}")
            cv2.imwrite(new_path, img)
            data.append({"path": new_path, "crop_type": crop_type, "damage": damage})

    # Save image metadata
    df = pd.DataFrame(data)
    df.to_csv("data/processed/image_labels.csv", index=False)
    print(f"Saved image metadata to data/processed/image_labels.csv")

def preprocess_tabular():
    df = pd.read_csv(RAW_TABULAR)
    # Clean missing values
    df = df.dropna()
    # Ensure columns exist; adapt if 'Temperature' is missing
    expected_columns = ["Crop", "Year", "Yield", "Rainfall", "Temperature"]
    available_columns = [col for col in expected_columns if col in df.columns]
    df = df[available_columns]
    df.to_csv(PROCESSED_TABULAR, index=False)
    print(f"Saved processed tabular data to {PROCESSED_TABULAR}")

if __name__ == "__main__":
    preprocess_images()
    preprocess_tabular()