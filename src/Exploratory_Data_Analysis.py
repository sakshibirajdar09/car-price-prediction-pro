import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Settings for professional plots
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Load the preprocessed data
df = pd.read_csv('./Cleaned_Combined_Dataset/Final_ML_Ready_Cars.csv')

# --- SECTION 1: TARGET VARIABLE ANALYSIS ---
plt.figure(figsize=(10, 6))
sns.histplot(df['price_numeric'], kde=True, color='blue')
plt.title('Distribution of Car Prices')
plt.xlabel('Price (in INR)')
plt.ylabel('Frequency')
plt.show()
# Note: If the plot is heavily right-skewed, we may need log transformation.

# --- SECTION 2: CATEGORICAL INSIGHTS ---
# Top 10 Brands by Volume
plt.figure(figsize=(12, 6))
df['brand'].value_counts().head(10).plot(kind='bar', color='teal')
plt.title('Top 10 Car Brands in Dataset')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Price vs Fuel Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='fuel_type', y='price_numeric', data=df)
plt.title('Price Distribution by Fuel Type')
plt.show()

# --- SECTION 3: NUMERICAL RELATIONSHIPS ---
# Age vs Price (Major Driver)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='car_age', y='price_numeric', data=df, alpha=0.5)
plt.title('Impact of Car Age on Price')
plt.show()

# Kilometers vs Price
plt.figure(figsize=(10, 6))
sns.regplot(x='km_numeric', y='price_numeric', data=df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Kilometers Driven vs Price')
plt.show()

# --- SECTION 4: MULTIVARIATE ANALYSIS ---
# Correlation Heatmap
plt.figure(figsize=(12, 10))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# --- SECTION 5: KEY INSIGHTS FOR PRESENTATION ---
print("--- SUMMARY STATISTICS ---")
print(f"Average Car Price: ₹{df['price_numeric'].mean():,.2f}")
print(f"Most Common Fuel Type: {df['fuel_type'].mode()[0]}")
print(f"Strongest Correlation with Price: {numeric_df.corr()['price_numeric'].sort_values(ascending=False).index[1]}")