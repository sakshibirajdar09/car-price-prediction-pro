import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

# 1. Load Data
df = pd.read_excel('Cleaned_Combined_Dataset/Final_Cleaned_Combined_Cars.xlsx')

# 2. Basic Preprocessing
# We select the most important features for prediction
features = ['city', 'fuel_type', 'body_type', 'transmission', 'owner_count', 'brand', 'km_numeric', 'car_age', 'engine_cc']
target = 'price_numeric'

# Drop rows where essential data is missing
df_ml = df[features + [target]].dropna()

# 3. Encoding (Machines only understand numbers, so we convert 'Petrol' -> 1, etc.)
encoders = {}
for col in ['city', 'fuel_type', 'body_type', 'transmission', 'brand']:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    encoders[col] = le # Save encoders to reuse later for new inputs

# 4. Split Data (80% for training, 20% for testing)
X = df_ml[features]
y = df_ml[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Model
print("Training the model... please wait...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# 6. Evaluate
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n--- Model Performance ---")
print(f"Accuracy (R2 Score): {score:.2f}")
print(f"Average Error: ₹{mae:,.2f}")

# 7. Feature Importance (Which factor matters most?)
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n--- What affects price most? ---")
print(importances)

# 8. Save the model so you can use it later without re-training
with open('car_price_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'encoders': encoders}, f)
print("\nModel saved as car_price_model.pkl")