import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Automatically create the plots folder
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load your combined and cleaned data
# Note: Ensure you are running this from the main project folder
df = pd.read_excel('Cleaned_Combined_Dataset/Final_Cleaned_Combined_Cars.xlsx')

# Set the style for professional plots
sns.set_theme(style="whitegrid")

# --- PLOT 1: Price Distribution ---
plt.figure(figsize=(10, 6))
sns.histplot(df['price_numeric'], bins=30, kde=True, color='teal')
plt.title('Distribution of Car Prices', fontsize=15)
plt.xlabel('Price (in ₹)')
plt.savefig('plots/1_price_distribution.png')
plt.close() # Close figure to free up memory

# --- PLOT 2: Price vs Car Age (Depreciation) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='car_age', y='price_numeric', hue='fuel_type', alpha=0.6)
plt.title('Car Price vs Age (Depreciation)', fontsize=15)
plt.xlabel('Age of Car (Years)')
plt.savefig('plots/2_depreciation_curve.png')
plt.close()

# --- PLOT 3: Brand Value Analysis ---
plt.figure(figsize=(10, 6))
top_brands = df.groupby('brand')['price_numeric'].mean().sort_values(ascending=False).head(10)
top_brands.plot(kind='bar', color='salmon')
plt.title('Top 10 Brands by Average Resale Price', fontsize=15)
plt.xticks(rotation=45)
plt.ylabel('Average Price (₹)')
plt.savefig('plots/3_brand_value.png')
plt.close()

# --- PLOT 4: Transmission Impact ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='transmission', y='price_numeric', palette='pastel')
plt.title('Price Variation: Manual vs Automatic', fontsize=15)
plt.ylim(0, df['price_numeric'].quantile(0.95)) 
plt.savefig('plots/4_transmission_impact.png')
plt.close()

# --- PLOT 5: Correlation Heatmap (Required for Streamlit Tab) ---
plt.figure(figsize=(10, 8))
numeric_cols = df[['price_numeric', 'km_numeric', 'car_age', 'engine_cc', 'owner_count']]
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap', fontsize=15)
plt.savefig('plots/5_correlation_heatmap.png')
plt.close()

# Print statistical summary to the terminal
print("--- DATA SUMMARY ---")
print(df[['price_numeric', 'km_numeric', 'car_age', 'engine_cc']].describe())
print("\n✅ Success! 5 charts have been saved to the 'plots/' folder.")