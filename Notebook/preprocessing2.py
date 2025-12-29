import pandas as pd
import numpy as np
import re

def extract_numeric(text):
    """
    Extracts the first numeric value from a string.
    Example: '23.1 kmpl' -> 23.1
    """
    if pd.isna(text) or str(text).lower() == 'nan':
        return np.nan
    match = re.search(r'(\d+\.?\d*)', str(text))
    return float(match.group(1)) if match else np.nan

def impute_by_group(df, target_col, group_cols):
    """
    Fills missing values using a hierarchical approach:
    1. Median of the specific Model
    2. Median of the Brand (if Model median is unavailable)
    3. Global Median
    """
    # Step 1: Impute by Model
    df[target_col] = df[target_col].fillna(df.groupby(group_cols[0])[target_col].transform('median'))
    # Step 2: Impute by Brand
    df[target_col] = df[target_col].fillna(df.groupby(group_cols[1])[target_col].transform('median'))
    # Step 3: Global median fallback
    df[target_col] = df[target_col].fillna(df[target_col].median())
    return df

def remove_outliers(df, col):
    """
    Removes statistical outliers using the Interquartile Range (IQR) method.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

def preprocess_car_data(file_path):
    # 1. Load the dataset
    print(f"Loading {file_path}...")
    df = pd.read_excel(file_path)

    # 2. Feature Extraction: Convert strings to numeric
    print("Extracting numeric values from specification strings...")
    df['mileage_numeric'] = df['spec_mileage'].apply(extract_numeric)
    df['max_power_numeric'] = df['spec_max_power'].apply(extract_numeric)
    df['torque_numeric'] = df['spec_torque'].apply(extract_numeric)

    # 3. Handling Missing Values: Tiered Imputation
    print("Imputing missing values...")
    numeric_cols_to_fix = ['mileage_numeric', 'max_power_numeric', 'torque_numeric', 'engine_cc', 'spec_seats']
    for col in numeric_cols_to_fix:
        df = impute_by_group(df, col, ['model', 'brand'])

    # Categorical Imputation for Body Type
    df['body_type'] = df['body_type'].fillna(df.groupby('model')['body_type'].transform(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    ))
    df['body_type'] = df['body_type'].fillna('Hatchback')

    # 4. Outlier Removal: Filtering Price and Kilometers
    print("Removing outliers to improve model performance...")
    df_cleaned = remove_outliers(df, 'price_numeric')
    df_cleaned = remove_outliers(df_cleaned, 'km_numeric')

    # 5. Feature Selection: Keep only the relevant ML features
    final_features = [
        'city', 'fuel_type', 'body_type', 'transmission', 'owner_count', 
        'brand', 'model', 'variant', 'car_age', 'price_numeric', 
        'km_numeric', 'engine_cc', 'mileage_numeric', 'max_power_numeric', 
        'torque_numeric', 'spec_seats'
    ]
    
    final_df = df_cleaned[final_features]

    # 6. Final Export
    output_name = 'Final_ML_Ready_Cars.csv'
    final_df.to_csv(output_name, index=False)
    
    print("\n--- Preprocessing Complete ---")
    print(f"Initial Records: {len(df)}")
    print(f"Final Records (after outlier removal): {len(final_df)}")
    print(f"Missing Values: \n{final_df.isnull().sum()}")
    print(f"\nSaved cleaned data to: {output_name}")
    
    return final_df

# Execute the preprocessing pipeline
if __name__ == "__main__":
    file_name = 'Final_Cleaned_Combined_Cars.xlsx'
    cleaned_data = preprocess_car_data(file_name)