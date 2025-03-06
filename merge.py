import pandas as pd
import os

# Paths
RAW_DIR = "data/raw/"
OUTPUT_FILE = "data/raw/yield_prediction.csv"

# Load CSV files
yield_df = pd.read_csv(os.path.join(RAW_DIR, "yield_df.csv"))
rainfall_df = pd.read_csv(os.path.join(RAW_DIR, "rainfall.csv"))
temp_df = pd.read_csv(os.path.join(RAW_DIR, "temp.csv"))

# Standardize column names immediately
yield_df = yield_df.rename(columns={"hg/ha_yield": "Yield", "avg_temp": "Temperature", "pesticides_tonnes": "Pesticides"})
rainfall_df = rainfall_df.rename(columns={" Area": "Area", "average_rain_fall_mm_per_year": "Rainfall"})
temp_df = temp_df.rename(columns={"year": "Year", "country": "Area", "avg_temp": "Temperature"})

# Normalize Area names
def normalize_area(area):
    if isinstance(area, str):
        return area.replace("Côte D'Ivoire", "Cote d'Ivoire").replace("United States", "USA").strip()
    return area

yield_df["Area"] = yield_df["Area"].apply(normalize_area)
rainfall_df["Area"] = rainfall_df["Area"].apply(normalize_area)
temp_df["Area"] = temp_df["Area"].apply(normalize_area)

# Print column names and shapes for debugging
print("yield_df columns:", yield_df.columns.tolist())
print("yield_df shape:", yield_df.shape)
print("rainfall_df columns:", rainfall_df.columns.tolist())
print("rainfall_df shape:", rainfall_df.shape)
print("temp_df columns:", temp_df.columns.tolist())
print("temp_df shape:", temp_df.shape)

# Print unique values for merge keys
print("Unique Areas in yield_df:", yield_df["Area"].unique()[:5])
print("Unique Years in yield_df:", yield_df["Year"].unique()[:5])
print("Unique Areas in rainfall_df:", rainfall_df["Area"].unique()[:5])
print("Unique Years in rainfall_df:", rainfall_df["Year"].unique()[:5])
print("Unique Areas in temp_df:", temp_df["Area"].unique()[:5])
print("Unique Years in temp_df:", temp_df["Year"].unique()[:5])

# Print unique crops in yield_df to check for Tomato, Corn, Potato
print("Unique crops in yield_df:", yield_df["Item"].unique()[:10])

# First merge with left join
merged_df = yield_df.merge(rainfall_df, on=["Area", "Year"], how="left")
print("After yield_df + rainfall_df merge, shape:", merged_df.shape)

# Second merge with left join
merged_df = merged_df.merge(temp_df, on=["Area", "Year"], how="left")
print("After temp_df merge, shape:", merged_df.shape)

# Filter for crops (adjusted to match actual names)
crops_of_interest = ["Maize", "Potatoes"]  # Adjusted based on unique crops (e.g., Maize for Corn, Potatoes for Potato)
if "Item" in merged_df.columns:
    print("Number of rows before crop filter:", len(merged_df))
    merged_df = merged_df[merged_df["Item"].isin(crops_of_interest)]
    print("Number of rows after crop filter:", len(merged_df))
else:
    print("Warning: 'Item' column not found after merge")

# Select and rename columns
available_columns = [col for col in ["Item", "Year", "Yield", "Rainfall", "Temperature"] if col in merged_df.columns]
if not available_columns:
    print("Error: No expected columns found after merge")
    exit()
merged_df = merged_df[available_columns]
merged_df.columns = ["Crop", "Year", "Yield"] + (["Rainfall", "Temperature"] if "Rainfall" in available_columns and "Temperature" in available_columns else ["Rainfall"] if "Rainfall" in available_columns else [])

# Handle missing values
merged_df = merged_df.fillna(0)

# Save
os.makedirs(RAW_DIR, exist_ok=True)
merged_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved merged data to {OUTPUT_FILE}")
print(f"Final shape of merged data: {merged_df.shape}")
