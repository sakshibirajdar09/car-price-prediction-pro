import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_model(name, y_true, y_pred):
    """Prints evaluation metrics for a given model."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n--- {name} Performance ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Absolute Error: ₹{mae:,.2f}")
    print(f"Root Mean Squared Error: ₹{rmse:,.2f}")
    return r2

def train_and_evaluate(data_dir='Processed_Data', model_dir='models'):
    # 1. Load the processed data
    print("Loading processed data...")
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()

    # 2. Initialize Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    best_model = None
    best_r2 = -float('inf')

    # 3. Train and Compare
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        current_r2 = evaluate_model(name, y_test, preds)

        # Save the best model based on R2 Score
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_model = model
            best_model_name = name

    # 4. Save the Best Model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = f'{model_dir}/best_car_price_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"\n🏆 Best Model: {best_model_name} with R2: {best_r2:.4f}")
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_evaluate()