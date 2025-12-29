# 🚗 CarDekho Used Car Price Prediction

## 📌 Project Overview
The objective of this project is to develop a data science solution that accurately predicts the market valuation of used cars.  
By analyzing a diverse dataset from **CarDekho** including car model, manufacturing year, fuel type, kilometers driven, and location we built a machine learning pipeline that provides instant price estimates via an interactive web application.

#### Live Application Link : https://car-price-predictor-pro.streamlit.app/
---

## 🛠️ Tech Stack
- **Language:** Python 3.11  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn  
- **Deployment:** Streamlit  
- **Tools:** VS Code, Git, GitHub  

---

## 📂 Project Structure

```plaintext
CAR_PRICE_PREDICTION/
├── .venv/                         # Virtual environment
├── app/
│   └── main.py                    # Streamlit application entry point
├── Cleaned_Combined_Dataset/       # Final cleaned & merged dataset
├── Dataset/                       # Raw datasets (city-wise)
├── eda_plots/                     # EDA visualizations
├── models/                        # Saved models & preprocessing objects
│   ├── best_car_price_model.pkl   # Trained XGBoost model
│   ├── le_brand.pkl               # Label encoder for brand
│   ├── le_city.pkl                # Label encoder for city
│   ├── le_model.pkl               # Label encoder for car model
│   └── scaler.pkl                 # Feature scaler
├── Notebook/
│   ├── Car_Preprocessing.ipynb    # Data preprocessing & EDA notebook
│   └── preprocessing2.py          # Additional preprocessing script
├── Processed_Data/                # Encoded & scaled data
├── Reports/
│   └── accuracy_check.png         # Model evaluation plot
├── src/                           # Core pipeline scripts
│   ├── data_preprocessing.py      # Data cleaning & transformation
│   ├── Exploratory_Data_Analysis.py # EDA logic
│   ├── feature_engineering.py     # Encoding & scaling
│   ├── model_training.py          # Model training & selection
│   └── evaluation.py              # Model evaluation & metrics
├── .gitignore                     # Git ignored files
├── readme.md                      # Project documentation
└── requirements.txt               # Python dependencies

```
## 🚀 How to Run the Project

### 1️⃣ Setup Environment
```bash
# Clone the repository
git clone https://github.com/sakshibirajdar09/Car-Price-Prediction.git
cd Car-Price-Prediction

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2️⃣ Run the Pipeline

| Step | Description | Command |
|-----|------------|--------|
| Data Cleaning | Flattens raw data and handles missing values | `python src/data_preprocessing.py` |
| Feature Engineering | Encodes categories and scales numbers | `python src/feature_engineering.py` |
| Model Training | Trains models and selects the best one | `python src/model_training.py` |
| Evaluation | Generates performance plots | `python src/evaluation.py` |

---

## 3️⃣ Launch the Application
```bash
python -m streamlit run app/main.py
```
## 📊 Model Performance

After evaluating multiple regression models (**Linear Regression**, **Random Forest**, **XGBoost**), the **XGBoost Regressor** was selected as the final model.

- **Best Model:** XGBoost  
- **R² Score:** 0.9192 (91.9%)  
- **Mean Absolute Error (MAE):** ₹ 67,702.25  
- **Root Mean Squared Error (RMSE):** ₹ 103,020.15  

---

## 🔍 Key EDA Insights
- **Depreciation:** Car age is the strongest predictor of price, showing a clear downward trend as age increases.
- **Brand Impact:** Premium brands like **Audi** and **BMW** retain value differently compared to budget brands like **Maruti**.
- **Usage:** Kilometers driven has a significant negative correlation with price, with a non-linear effect.

---

## 🤝 Contact
**Developed by Sakshi**

- GitHub: sakshibirajdar09   

⭐ If you like this project, don’t forget to star the repository!