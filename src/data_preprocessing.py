import pandas as pd
import numpy as np
import ast
import re
import os

def flatten_cardekho_row(row, city_name):
    """Parses nested dictionary strings from the raw Excel format."""
    flat_data = {'city': city_name}

    def safe_eval(col_name):
        try:
            val = row.get(col_name, '{}')
            # Handles the string representation of Python dictionaries
            return ast.literal_eval(str(val)) if pd.notna(val) else {}
        except: return {}

    # Extract from 'new_car_detail'
    detail = safe_eval('new_car_detail')
    flat_data.update({
        'fuel_type': detail.get('ft'),
        'body_type': detail.get('bt'),
        'kilometers': detail.get('km'),
        'transmission': detail.get('transmission'),
        'owner_count': detail.get('ownerNo'),
        'brand': detail.get('oem'),
        'model': detail.get('model'),
        'model_year': detail.get('modelYear'),
        'variant': detail.get('variantName'),
        'price': detail.get('price')
    })

    # Extract from 'new_car_overview'
    overview = safe_eval('new_car_overview')
    for item in overview.get('top', []):
        flat_data[f"overview_{item['key'].lower().replace(' ', '_')}"] = item['value']

    # Extract from 'new_car_specs'
    specs = safe_eval('new_car_specs')
    for item in specs.get('top', []):
        flat_data[f"spec_{item['key'].lower().replace(' ', '_')}"] = item['value']

    return flat_data

def clean_price(val):
    """Converts price strings (Lakh/Cr) to uniform numeric values."""
    if pd.isna(val) or val == '': return None
    s = str(val).lower()
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", s.replace(',', ''))
    if not nums: return None
    num = float(nums[0])
    if 'lakh' in s: return num * 100000
    if 'cr' in s or 'crore' in s: return num * 10000000
    return num

def extract_numeric(text):
    """Extracts first numeric value from a string (e.g., '1197 cc' -> 1197)."""
    if pd.isna(text) or str(text).lower() == 'nan': return np.nan
    match = re.search(r'(\d+\.?\d*)', str(text))
    return float(match.group(1)) if match else np.nan

def impute_by_group(df, target_col, group_cols):
    """Tiered imputation: fills missing values based on Model, then Brand."""
    df[target_col] = df[target_col].fillna(df.groupby(group_cols[0])[target_col].transform('median'))
    df[target_col] = df[target_col].fillna(df.groupby(group_cols[1])[target_col].transform('median'))
    df[target_col] = df[target_col].fillna(df[target_col].median())
    return df

def remove_outliers(df, col):
    """Uses IQR method to filter extreme price/km values."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

# --- MAIN PIPELINE ---

def run_preprocessing_pipeline(dataset_dir='Dataset', output_dir='Cleaned_Combined_Dataset'):
    """Full execution flow from raw files to final ML dataset."""
    
    # 1. Load and Flatten Data
    file_list = [f for f in os.listdir(dataset_dir) if f.endswith('.xlsx')]
    all_records = []

    print(f"Phase 1: Flattening {len(file_list)} files from {dataset_dir}...")
    for file in file_list:
        city = file.split('_')[0].capitalize()
        path = os.path.join(dataset_dir, file)
        df_raw = pd.read_excel(path)
        for _, row in df_raw.iterrows():
            all_records.append(flatten_cardekho_row(row, city))

    master_df = pd.DataFrame(all_records)
    
    # 2. Basic Cleaning & Extraction
    print("Phase 2: Numeric Extraction & Initial Cleaning...")
    master_df['price_numeric'] = master_df['price'].apply(clean_price)
    master_df['km_numeric'] = master_df['kilometers'].apply(extract_numeric)
    master_df['engine_cc'] = master_df['spec_engine'].apply(extract_numeric)
    master_df['mileage_numeric'] = master_df['spec_mileage'].apply(extract_numeric)
    master_df['max_power_numeric'] = master_df['spec_max_power'].apply(extract_numeric)
    master_df['torque_numeric'] = master_df['spec_torque'].apply(extract_numeric)
    master_df['car_age'] = 2024 - pd.to_numeric(master_df['model_year'], errors='coerce')

    # 3. Handling Missing Values (Imputation)
    print("Phase 3: Handling Missing Values (Imputation)...")
    numeric_cols = ['mileage_numeric', 'max_power_numeric', 'torque_numeric', 'engine_cc', 'spec_seats']
    for col in numeric_cols:
        master_df = impute_by_group(master_df, col, ['model', 'brand'])

    # 4. Outlier Removal
    print("Phase 4: Filtering Outliers...")
    df_cleaned = remove_outliers(master_df, 'price_numeric')
    df_cleaned = remove_outliers(df_cleaned, 'km_numeric')

    # 5. Final Formatting
    final_features = [
        'city', 'fuel_type', 'body_type', 'transmission', 'owner_count', 
        'brand', 'model', 'variant', 'car_age', 'price_numeric', 
        'km_numeric', 'engine_cc', 'mileage_numeric', 'max_power_numeric', 
        'torque_numeric', 'spec_seats'
    ]
    
    final_df = df_cleaned[final_features]
    
    # 6. Saving Results
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    final_csv_path = os.path.join(output_dir, 'Final_ML_Ready_Cars.csv')
    final_df.to_csv(final_csv_path, index=False)
    
    print(f"\nSUCCESS! 100% Preprocessing Complete.")
    print(f"Final dataset saved to: {final_csv_path}")
    print(f"Final shape: {final_df.shape}")

if __name__ == "__main__":
    # Ensure folders exist relative to where you run this script
    run_preprocessing_pipeline()