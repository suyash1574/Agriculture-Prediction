import pandas as pd
import os

# Paths
RAW_DIR = "data/raw/"  # Updated to match your directory structure
OUTPUT_FILE = "data/raw/yield_prediction.csv"

# Load CSV files
yield_df = pd.read_csv(os.path.join(RAW_DIR, "yield_df.csv"))
rainfall_df = pd.read_csv(os.path.join(RAW_DIR, "rainfall.csv"))
temp_df = pd.read_csv(os.path.join(RAW_DIR, "temp.csv"))

# Print column names for debugging
print("yield_df columns:", yield_df.columns.tolist())
print("rainfall_df columns:", rainfall_df.columns.tolist())
print("temp_df columns:", temp_df.columns.tolist())

# Standardize column names
yield_df = yield_df.rename(columns={"hg/ha_yield": "Yield", "avg_temp": "Temperature", "pesticides_tonnes": "Pesticides"})
rainfall_df = rainfall_df.rename(columns={" Area": "Area", "average_rain_fall_mm_per_year": "Rainfall"})  # Remove leading space
temp_df = temp_df.rename(columns={"year": "Year", "country": "Area", "avg_temp": "Temperature"})

# Debug unique values to check for matching keys
print("Unique Areas in yield_df:", yield_df["Area"].unique()[:5])  # Show first 5 for brevity
print("Unique Years in yield_df:", yield_df["Year"].unique()[:5])
print("Unique Areas in rainfall_df:", rainfall_df["Area"].unique()[:5])
print("Unique Years in rainfall_df:", rainfall_df["Year"].unique()[:5])
print("Unique Areas in temp_df:", temp_df["Area"].unique()[:5])
print("Unique Years in temp_df:", temp_df["Year"].unique()[:5])

# First merge (yield_df with rainfall_df)
merged_df = yield_df.merge(rainfall_df, on=["Area", "Year"], how="left")
print("After yield_df + rainfall_df merge, columns:", merged_df.columns.tolist())
print("Shape after first merge:", merged_df.shape)

# Second merge (with temp_df)
merged_df = merged_df.merge(temp_df, on=["Area", "Year"], how="left")
print("After temp_df merge, columns:", merged_df.columns.tolist())
print("Shape after second merge:", merged_df.shape)

# Filter for crops matching PlantVillage
crops_of_interest = ["Tomato", "Corn", "Potato"]
merged_df = merged_df[merged_df["Item"].isin(crops_of_interest)]

# Select and rename columns
# Check if 'Temperature' exists; if not, exclude it
available_columns = [col for col in ["Item", "Year", "Yield", "Rainfall", "Temperature"] if col in merged_df.columns]
print("Available columns for selection:", available_columns)
merged_df = merged_df[available_columns]
merged_df.columns = ["Crop", "Year", "Yield"] + (["Rainfall", "Temperature"] if "Rainfall" in available_columns and "Temperature" in available_columns else ["Rainfall"] if "Rainfall" in available_columns else [])

# Handle missing values
merged_df = merged_df.dropna()

# Save
os.makedirs(RAW_DIR, exist_ok=True)
merged_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved merged data to {OUTPUT_FILE}")