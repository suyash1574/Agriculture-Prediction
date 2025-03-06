import os
import cv2
import pandas as pd
import numpy as np

# Paths
RAW_IMAGE_DIR = "data/raw/PlantVillage/segmented/"
PROCESSED_IMAGE_DIR = "data/processed/images/"
RAW_TABULAR = "data/raw/yield_prediction.csv"
PROCESSED_TABULAR = "data/processed/tabular.csv"

# Define crop types and damage mapping (expanded for PlantVillage dataset)
CROP_TYPES = ["Tomato", "Corn", "Potato", "Apple"]
DAMAGE_MAP = {
    "healthy": 0.0,
    "bacterial_spot": 0.5,
    "late_blight": 0.8,
    "gray_leaf_spot": 0.6,
    "common_rust": 0.7,
    "early_blight": 0.6,
    "mosaic_virus": 0.9,
    "leaf_spot": 0.5,
    "apple_scab": 0.7,
    "cedar_rust": 0.6,
}

def preprocess_images():
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
    data = []
    unprocessed_folders = []
    
    print(f"Scanning directories in {RAW_IMAGE_DIR}")
    folders = os.listdir(RAW_IMAGE_DIR)
    print(f"Found folders: {folders}")
    
    for folder in folders:
        # Extract crop type (more flexible matching)
        crop_type = next((c for c in CROP_TYPES if c.lower() in folder.lower().split('___')[0].lower()), None)
        if not crop_type:
            unprocessed_folders.append(folder)
            print(f"Skipping folder {folder}: No matching crop type in {CROP_TYPES}")
            continue
        
        # Extract damage (match any part of folder name)
        damage = next((v for k, v in DAMAGE_MAP.items() if k.lower() in folder.lower()), 0.0)
        if damage == 0.0 and "healthy" not in folder.lower():
            print(f"No damage match for {folder}, defaulting to 0.0")
        
        print(f"Processing folder: {folder}, Crop: {crop_type}, Damage: {damage}")
        
        folder_path = os.path.join(RAW_IMAGE_DIR, folder)
        if not os.path.isdir(folder_path):
            unprocessed_folders.append(folder)
            print(f"Skipping {folder}: Not a directory")
            continue
            
        img_files = os.listdir(folder_path)[:100]  # Limit to 100 images per folder
        print(f"Found {len(img_files)} images in {folder}")
        
        for img_file in img_files:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            img = cv2.resize(img, (224, 224))
            new_path = os.path.join(PROCESSED_IMAGE_DIR, f"{crop_type}_{damage}_{img_file}")
            cv2.imwrite(new_path, img)
            data.append({"path": new_path, "crop_type": crop_type, "damage": damage})
            print(f"Processed image: {img_path} -> {new_path}")

    if unprocessed_folders:
        print(f"Warning: Unprocessed folders: {unprocessed_folders}")
    if not data:
        print("Error: No image data processed. Check PlantVillage directory, CROP_TYPES, and DAMAGE_MAP.")
    else:
        df = pd.DataFrame(data)
        print(f"Processed {len(data)} images")
        df.to_csv("data/processed/image_labels.csv", index=False)
        print(f"Saved image metadata to data/processed/image_labels.csv")

def preprocess_tabular():
    if not os.path.exists(RAW_TABULAR):
        print(f"Error: {RAW_TABULAR} not found")
        return
    df = pd.read_csv(RAW_TABULAR)
    if df.empty:
        print(f"Error: {RAW_TABULAR} is empty")
        return
    df = df.dropna()
    if df.empty:
        print(f"Warning: All rows dropped due to NaN values in {RAW_TABULAR}")
        return
    expected_columns = ["Crop", "Year", "Yield", "Rainfall", "Temperature"]
    available_columns = [col for col in expected_columns if col in df.columns]
    if not available_columns:
        print(f"Error: No expected columns found in {RAW_TABULAR}")
        return
    df = df[available_columns]
    df.to_csv(PROCESSED_TABULAR, index=False)
    print(f"Saved processed tabular data to {PROCESSED_TABULAR}")

if __name__ == "__main__":
    preprocess_images()
    preprocess_tabular()