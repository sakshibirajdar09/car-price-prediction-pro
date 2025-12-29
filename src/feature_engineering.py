import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib

def perform_feature_engineering(input_path='Cleaned_Combined_Dataset/Final_ML_Ready_Cars.csv', output_dir='Processed_Data'):
    df = pd.read_csv(input_path)
    df_ml = df.drop(['variant'], axis=1)
    
    # Encoders
    le_brand = LabelEncoder()
    le_model = LabelEncoder()
    le_city = LabelEncoder()
    
    df_ml['brand'] = le_brand.fit_transform(df_ml['brand'])
    df_ml['model'] = le_model.fit_transform(df_ml['model'])
    df_ml['city'] = le_city.fit_transform(df_ml['city'])
    
    # One-Hot Encoding (Matches your previous output)
    df_ml = pd.get_dummies(df_ml, columns=['fuel_type', 'transmission', 'body_type'], drop_first=True)
    
    X = df_ml.drop('price_numeric', axis=1)
    y = df_ml['price_numeric']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_cols = ['owner_count', 'car_age', 'km_numeric', 'engine_cc', 
                    'mileage_numeric', 'max_power_numeric', 'torque_numeric', 'spec_seats']
    
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
    
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le_brand, 'models/le_brand.pkl')
    joblib.dump(le_model, 'models/le_model.pkl')
    joblib.dump(le_city, 'models/le_city.pkl') # Added this!
    
    print("Done! Encoders and Scaler saved. Now re-run your model training.")

if __name__ == "__main__":
    perform_feature_engineering()