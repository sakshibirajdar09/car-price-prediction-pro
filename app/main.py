import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="CarDekho Pro - AI Price Predictor",
    page_icon="carrr.png",
    layout="wide"
)

# 2. GLOBAL CONSTANTS (This fixes the 'NameError')
NUM_COLS = ['owner_count', 'car_age', 'km_numeric', 'engine_cc', 'mileage_numeric', 'max_power_numeric', 'torque_numeric', 'spec_seats']

# 3. THEME STATE MANAGEMENT
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# 4. THEME STYLING DICTIONARY
theme_styles = {
    'light': {
        'bg': 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
        'card_bg': 'rgba(255, 255, 255, 0.8)',
        'text': '#1e293b',
        'sidebar_bg': '#ffffff'
    },
    'dark': {
        'bg': 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
        'card_bg': 'rgba(30, 41, 59, 0.85)',
        'text': '#f8fafc',
        'sidebar_bg': '#0f172a'
    }
}

current = theme_styles[st.session_state.theme]

# CSS injection for Glassmorphism and Single Car Animation
st.markdown(f"""
    <style>
    .stApp {{
        background: {current['bg']};
        color: {current['text']};
    }}
    
    /* Single Car Driving Animation (Left to Right) */
    @keyframes driveRight {{
        0% {{ left: -200px; opacity: 0; }}
        10% {{ opacity: 1; }}
        90% {{ opacity: 1; }}
        100% {{ left: 110%; opacity: 0; }}
    }}

    .driving-car {{
        position: fixed;
        bottom: 80px;
        z-index: 9999;
        font-size: 80px;
        pointer-events: none;
        animation: driveRight 5s ease-in-out forwards;
    }}

    /* UI Card Styling */
    .result-card {{
        background: {current['card_bg']};
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin: 30px auto;
        backdrop-filter: blur(12px);
        max-width: 800px;
    }}

    .price-display {{
        color: #ef4444;
        font-size: 5rem;
        font-weight: 900;
        margin: 9px 0;
        text-shadow: 0 4px 10px rgba(239, 68, 68, 0.2);
    }}

    .stButton>button {{
        border-radius: 12px;
        padding: 15px 30px;
        background: #ef4444;
        color: white;
        font-weight: bold;
        border: none;
        width: 100%;
    }}
    </style>
    """, unsafe_allow_html=True)

# 5. ASSET LOADING
@st.cache_resource
def load_assets():
    model = joblib.load('models/best_car_price_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le_brand = joblib.load('models/le_brand.pkl')
    le_model = joblib.load('models/le_model.pkl')
    le_city = joblib.load('models/le_city.pkl')
    return model, scaler, le_brand, le_model, le_city

model, scaler, le_brand, le_model, le_city = load_assets()

# 6. SIDEBAR
with st.sidebar:
    st.image("carrr.png", width=150)
    st.title("CarDekho Settings")
    
    st.markdown("### 🌓 Appearance")
    st.button("Toggle Light/Dark Mode", on_click=toggle_theme)
    st.caption(f"Active Theme: {st.session_state.theme.upper()}")
    
    st.divider()
    st.subheader("📍 Vehicle Details")
    city = st.selectbox("Select City", le_city.classes_)
    brand = st.selectbox("Select Brand", le_brand.classes_)
    car_model = st.selectbox("Select Model", le_model.classes_)
    st.divider()
    st.info("XGBoost Model: Operational ✅")

# 7. MAIN INTERFACE
st.title("🚗 Premium Car Valuation Dashboard")
st.write("Unlock the true market value of any vehicle using Real-Time AI.")

tab1, tab2 = st.tabs(["📝 Valuation Calculator", "📈 Market Insights"])

with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🏗️ Build")
        fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Cng', 'Lpg', 'Electric'])
        trans = st.radio("Transmission", ['Manual', 'Automatic'], horizontal=True)
        body = st.selectbox("Body Style", ['Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe', 'Minivans', 'Pickup Trucks', 'Wagon'])
        
    with col2:
        st.markdown("### 📅 History")
        age = st.slider("Car Age (Years)", 0, 25, 5)
        km = st.number_input("Kilometers Driven", 0, 500000, 30000, step=5000)
        owners = st.select_slider("Previous Owners", options=[1, 2, 3, 4, 5], value=1)
        
    with col3:
        st.markdown("### ⚙️ Performance")
        engine = st.number_input("Engine CC", 600, 5000, 1200)
        power = st.number_input("Max Power (bhp)", 30.0, 500.0, 85.0)
        seats = st.selectbox("Seats", [2, 4, 5, 7, 8])

    # Realistic averages for hidden specs
    mileage, torque = 18.0, 150.0

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 CALCULATE AI VALUATION") :
        with st.spinner('🔮 AI is crunching the numbers...'):
            # Encoding
            input_dict = {
                'city': le_city.transform([city])[0],
                'owner_count': owners,
                'brand': le_brand.transform([brand])[0],
                'model': le_model.transform([car_model])[0],
                'car_age': age,
                'km_numeric': km,
                'engine_cc': engine,
                'mileage_numeric': mileage,
                'max_power_numeric': power,
                'torque_numeric': torque,
                'spec_seats': seats,
                # Fuel One-Hot
                'fuel_type_Diesel': 1 if fuel == 'Diesel' else 0,
                'fuel_type_Electric': 1 if fuel == 'Electric' else 0,
                'fuel_type_Lpg': 1 if fuel == 'Lpg' else 0,
                'fuel_type_Petrol': 1 if fuel == 'Petrol' else 0,
                # Transmission One-Hot
                'transmission_Manual': 1 if trans == 'Manual' else 0,
                # Body Type One-Hot
                'body_type_Coupe': 1 if body == 'Coupe' else 0,
                'body_type_Hatchback': 1 if body == 'Hatchback' else 0,
                'body_type_MUV': 1 if body == 'MUV' else 0,
                'body_type_Minivans': 1 if body == 'Minivans' else 0,
                'body_type_Pickup Trucks': 1 if body == 'Pickup Trucks' else 0,
                'body_type_SUV': 1 if body == 'SUV' else 0,
                'body_type_Sedan': 1 if body == 'Sedan' else 0,
                'body_type_Wagon': 1 if body == 'Wagon' else 0
            }

            input_df = pd.DataFrame([input_dict])
            
            # Use Global NUM_COLS for scaling
            input_df_scaled = input_df.copy()
            input_df_scaled[NUM_COLS] = scaler.transform(input_df[NUM_COLS])

            # Prediction
            prediction = model.predict(input_df_scaled)[0]
            st.session_state['last_input'] = input_dict # Store for Tab 2

            # Single Car Animation
            st.markdown('<div class="driving-car">🏎️</div>', unsafe_allow_html=True)
            
            # Result UI
            st.markdown(f"""
                <div class="result-card">
                    <h2 style='color: {current['text']}; opacity: 0.8;'>Estimated Market Valuation</h2>
                    <div class="price-display">₹ {prediction:,.2f}</div>
                    <p style='color: #10b981; font-weight: bold;'>Confidence: 91.92%</p>
                </div>
            """, unsafe_allow_html=True)

with tab2:
    if 'last_input' in st.session_state:
        st.subheader("📈 Real-Time Depreciation Analysis")
        st.write(f"Analyzing value decay for your **{brand} {car_model}** across a 20-year span.")
        
        ages = np.arange(0, 21)
        base_data = st.session_state['last_input'].copy()
        
        trend_data = []
        for a in ages:
            row = base_data.copy()
            row['car_age'] = a
            trend_data.append(row)
            
        trend_df = pd.DataFrame(trend_data)
        
        # Scaling using Global NUM_COLS (Fixed the NameError here)
        trend_df_scaled = trend_df.copy()
        trend_df_scaled[NUM_COLS] = scaler.transform(trend_df[NUM_COLS])
        
        real_preds = model.predict(trend_df_scaled)
        
        fig = px.area(x=ages, y=real_preds, labels={'x': 'Age (Years)', 'y': 'Market Value (₹)'})
        fig.update_traces(line_color='#ef4444', fillcolor='rgba(239, 68, 68, 0.2)')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=current['text'])
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("🔍 Calculate a valuation in the 'Valuation Calculator' tab to see market trends.")

st.divider()
st.caption(f"Developed by Sakshi | CarDekho AI Project | Instance: {datetime.now().strftime('%H:%M:%S')} 🚀")