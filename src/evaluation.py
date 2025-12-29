import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import os
import numpy as np

def plot_accuracy():
    # 1. Create the Reports directory if it doesn't exist
    if not os.path.exists('Reports'):
        os.makedirs('Reports')
        print("Created 'Reports' directory.")

    # 2. Load test data and model
    print("Loading model and test data...")
    X_test = pd.read_csv('Processed_Data/X_test.csv')
    y_test = pd.read_csv('Processed_Data/y_test.csv')
    model = joblib.load('models/best_car_price_model.pkl')
    
    # 3. Get predictions
    y_pred = model.predict(X_test)
    
    # 4. Create Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test.values.flatten(), y=y_pred, alpha=0.5)
    
    # Add a diagonal line for reference
    line_min = min(y_test.min().iloc[0], y_pred.min())
    line_max = max(y_test.max().iloc[0], y_pred.max())
    plt.plot([line_min, line_max], [line_min, line_max], '--r', linewidth=2, label='Perfect Prediction')
    
    plt.title('Actual vs. Predicted Car Prices (XGBoost)')
    plt.xlabel('Actual Price (₹)')
    plt.ylabel('Predicted Price (₹)')
    plt.legend()
    
    # 5. Save the plot
    save_path = 'Reports/accuracy_check.png'
    plt.savefig(save_path)
    plt.close() # Close plot to free memory
    print(f"Success! Accuracy plot saved to: {save_path}")

if __name__ == "__main__":
    plot_accuracy()