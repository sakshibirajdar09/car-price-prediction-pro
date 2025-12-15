import streamlit as st
import pandas as pd
import pickle
import os
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="CarDekho Pro Predictor", page_icon="🚗", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #ff4b4b; color: white; font-weight: bold; font-size: 18px; }
    .prediction-box { font-size: 24px; font-weight: bold; color: #1e3d59; text-align: center; background: #e3f2fd; padding: 30px; border-radius: 15px; border: 2px solid #90caf9; margin-top: 20px; }
    .metric-title { color: #555; font-size: 14px; margin-bottom: 5px; }
    .explanation-title { margin-top: 30px; font-weight: bold; color: #1e3d59; }
    </style>
    """, unsafe_allow_html=True)

# 1. Load the saved model, encoders, and initialize SHAP
@st.cache_resource
def load_assets():
    # Make sure path matches your folder structure
    with open('model/car_price_model.pkl', 'rb') as f:
        data = pickle.load(f)
    
    model = data['model']
    # Initialize SHAP explainer once
    explainer = shap.TreeExplainer(model)
    return data, explainer

assets, explainer = load_assets()
model = assets['model']
encoders = assets['encoders']

# --- Sidebar ---
st.sidebar.title("🚗 CarDekho Pro AI")
st.sidebar.info("Model Accuracy: **76%**\n\nAlgorithm: **Random Forest**")
st.sidebar.divider()
st.sidebar.write("Developed by Sakshi ❤️")

# --- Main App Tabs ---
tab1, tab2 = st.tabs(["🔮 AI Price Predictor", "📊 Market Analysis"])

# --- TAB 1: PREDICTION ---
with tab1:
    st.title("Price Prediction & AI Explanation")
    st.write("Adjust the car details to see the predicted value and the AI's reasoning.")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        brand = st.selectbox("Select Brand", sorted(list(encoders['brand'].classes_)))
        city = st.selectbox("Select City", sorted(list(encoders['city'].classes_)))
        body_type = st.selectbox("Body Type", sorted(list(encoders['body_type'].classes_)))
        fuel_type = st.selectbox("Fuel Type", sorted(list(encoders['fuel_type'].classes_)))
        transmission = st.selectbox("Transmission", sorted(list(encoders['transmission'].classes_)))

    with col2:
        car_age = st.number_input("Car Age (Years)", min_value=0, max_value=30, value=5)
        km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=50000, step=1000)
        engine_cc = st.number_input("Engine Capacity (CC)", min_value=600, max_value=6000, value=1200, step=100)
        owners = st.slider("Number of Previous Owners", 1, 5, 1)

    if st.button("Calculate Value & Explain"):
        # Prepare input
        input_df = pd.DataFrame({
            'city': [city], 'fuel_type': [fuel_type], 'body_type': [body_type],
            'transmission': [transmission], 'owner_count': [owners], 'brand': [brand],
            'km_numeric': [km_driven], 'car_age': [car_age], 'engine_cc': [engine_cc]
        })

        try:
            # Encoding
            encoded_df = input_df.copy()
            for col in ['city', 'fuel_type', 'body_type', 'transmission', 'brand']:
                encoded_df[col] = encoders[col].transform(input_df[col])
            
            # Prediction
            prediction = model.predict(encoded_df)[0]
            
            # Result Display
            st.markdown(f"""
                <div class="prediction-box">
                    <span class="metric-title">ESTIMATED MARKET VALUE</span><br>
                    <span style="font-size: 42px;">₹ {prediction:,.2f}</span>
                </div>
            """, unsafe_allow_html=True)

            # --- ADVANCED: SHAP EXPLANATION ---
            st.markdown("<h3 class='explanation-title'>AI Decision Breakdown</h3>", unsafe_allow_html=True)
            st.write("The chart below shows how each feature influenced the final price (Red pushes price up, Blue pulls it down).")
            
            # Calculate SHAP values for this specific prediction
            shap_values = explainer.shap_values(encoded_df)
            
            # Display SHAP Force Plot
            st_shap(shap.force_plot(explainer.expected_value, shap_values[0], encoded_df.iloc[0,:]))
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- TAB 2: VISUALIZATION ---
with tab2:
    st.title("📊 Market Trends")
    plot_path = "plots/" 
    
    if os.path.exists(plot_path):
        c1, c2 = st.columns(2)
        with c1:
            st.image(f"{plot_path}1_price_distribution.png", caption="Overall Price Range", use_container_width=True)
            st.image(f"{plot_path}3_brand_value.png", caption="Brand Value Ranking", use_container_width=True)
        with c2:
            st.image(f"{plot_path}2_depreciation_curve.png", caption="Depreciation over Age", use_container_width=True)
            st.image(f"{plot_path}5_correlation_heatmap.png", caption="Factor Dependencies", use_container_width=True)
    else:
        st.warning("Please run 'src/eda.py' to generate charts.")

st.divider()
st.caption("Advanced AI Model for Car Price Estimation | Data Source: CarDekho")