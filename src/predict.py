import pickle
import pandas as pd

# 1. Load the saved model and encoders
with open('model/car_price_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
encoders = data['encoders']

# 2. Define a new car you want to value (Example input)
# Note: These must match the categories in your dataset exactly
new_car = {
    'city': 'Bangalore',
    'fuel_type': 'Petrol',
    'body_type': 'Hatchback',
    'transmission': 'Manual',
    'owner_count': 1,
    'brand': 'Maruti',
    'km_numeric': 50000,
    'car_age': 5,
    'engine_cc': 1200
}

# 3. Convert text to numbers using the saved encoders
input_df = pd.DataFrame([new_car])
for col in ['city', 'fuel_type', 'body_type', 'transmission', 'brand']:
    input_df[col] = encoders[col].transform(input_df[col])

# 4. Predict!
prediction = model.predict(input_df)
print(f"\nPredicted Resale Price: ₹{prediction[0]:,.2f}")