# 🚗 Car Price Prediction – AI-Based Used Car Valuation System

An end-to-end **Machine Learning & Data Analytics project** that predicts the fair market value of used cars in India using historical data and provides clear market insights through visual analysis and an interactive web app.

This project uses **Random Forest Regression**, **Explainable AI concepts**, and a **Streamlit dashboard** to deliver transparent and data-driven car price predictions.

---

## 📌 1. Problem Statement

The used car market often lacks transparency. Buyers and sellers face **price uncertainty** due to multiple influencing factors such as:

* Brand value
* Engine capacity
* Vehicle age
* Distance driven
* City-wise demand

### 🎯 Goal

To build a **reliable AI system** that:

* Predicts used car prices accurately
* Removes guesswork from pricing decisions
* Provides insights into market trends using data visualization

---

## 💡 2. Solution Overview

We designed a complete **data science pipeline**:

* **Data Cleaning & Integration:** Combined and cleaned datasets from multiple cities
* **Feature Engineering:** Converted textual units (Lakh, CC, KM) into numerical form
* **Machine Learning Model:** Trained a Random Forest Regressor to capture non-linear pricing patterns
* **Visualization & Insights:** Automated EDA for understanding depreciation and brand trends
* **Web Application:** Streamlit-based UI for real-time price prediction

---

## 🤖 3. Model Details

* **Algorithm:** Random Forest Regressor
* **Evaluation Metric:**

  * R² Score ≈ **0.76**
  * Mean Absolute Error ≈ **₹1.79 Lakhs**

### 🔑 Key Price Influencing Factors

* **Engine CC** – Indicates vehicle segment and performance
* **Car Age** – Major contributor to depreciation
* **Kilometers Driven** – Reflects vehicle usage and condition

---

## 🌟 4. Key Features

* 📊 **Exploratory Data Analysis (EDA):**

  * Brand-wise price comparison
  * Depreciation curves
  * Correlation heatmaps

* 🧠 **AI Price Prediction:**

  * Trained ML model saved and reused using Pickle

* 🌐 **Interactive Web App:**

  * User-friendly Streamlit interface
  * Real-time car price prediction

* 🗂️ **Well-Structured Project:**

  * Modular Python scripts for training, prediction, and analysis

---

## 💻 5. Installation & Setup

### 🔧 Prerequisites

* Python 3.8+
* pip

### 📥 Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/car-price-prediction.git
cd car-price-prediction
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ 6. How to Run the Project

### Step 1: Train the Model

```bash
python src/model.py
```

➡️ Saves trained model as:

```
model/car_price_model.pkl
```

### Step 2: Perform EDA & Generate Plots

```bash
python src/eda.py
```

➡️ Saves charts inside the `plots/` folder

### Step 3: Run Streamlit Web App

```bash
streamlit run streamlit_app.py
```

➡️ Opens the AI dashboard in your browser

---

## 📁 7. Project Folder Structure

```plaintext
CAR PRICE PREDICTION
│
├── Cleaned_Combined_Dataset/
│   └── Final_Cleaned_Combined_Cars.xlsx
│
├── Dataset/                      # Raw datasets (if any)
│
├── model/
│   └── car_price_model.pkl       # Trained ML model
│
├── Notebook/
│   └── Car_Preprocessing.ipynb   # Data preprocessing notebook
│
├── plots/                        # Generated EDA visualizations
│
├── src/
│   ├── eda.py                    # Exploratory Data Analysis
│   ├── model.py                  # Model training script
│   └── predict.py                # Prediction logic
│
├── streamlit_app.py              # Streamlit web application
├── requirements.txt              # Python dependencies
└── readme.md                     # Project documentation
```

---

## 🛠️ 8. Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit
* Pickle

---

## 🤝 9. Author & Contact

**Developed by:** Sakshi Birajdar
Passionate about applying AI to solve real-world business problems.

* 🔗 **LinkedIn:** https://www.linkedin.com/in/sakshibirajdar/

* 💻 **GitHub:** https://github.com/sakshibirajdar09
* 📧 **Email:** sakshibirajdar34@gmail.com

---

⭐ If you like this project, consider giving it a star on GitHub!
