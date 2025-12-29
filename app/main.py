import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# 1. Page Configuration & Theme
st.set_page_config(
    page_title="CarDekho Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Asset Loading
@st.cache_resource
def load_assets():
    model = joblib.load('models/best_car_price_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le_brand = joblib.load('models/le_brand.pkl')
    le_model = joblib.load('models/le_model.pkl')
    le_city = joblib.load('models/le_city.pkl')
    return model, scaler, le_brand, le_model, le_city

model, scaler, le_brand, le_model, le_city = load_assets()

# 3. Sidebar - Primary Filters
st.sidebar.image("https://img.icons8.com/clouds/200/car.png") # Visual branding
st.sidebar.title("Select Car Configuration")
st.sidebar.markdown("Adjust the primary details here.")

city = st.sidebar.selectbox("📍 Select City", le_city.classes_)
brand = st.sidebar.selectbox("🏢 Select Brand", le_brand.classes_)
car_model = st.sidebar.selectbox("🚘 Select Model", le_model.classes_)
trans = st.sidebar.radio("⚙️ Transmission", ['Manual', 'Automatic'])
fuel = st.sidebar.selectbox("⛽ Fuel Type", ['Petrol', 'Diesel', 'Cng', 'Lpg', 'Electric'])

# 4. Main Panel - Detailed Specifications
st.title("🚗 Car Price Prediction")
st.markdown("Enter technical details for a precise market estimate.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("General Details")
    age = st.slider("📅 Car Age (Years)", 0, 25, 5)
    km = st.number_input("🛣️ Kilometers Driven", 0, 500000, 30000, step=5000)
    owners = st.selectbox("👤 Previous Owners", [1, 2, 3, 4, 5])
    body_type = st.selectbox("🚙 Body Type", ['Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe', 'Minivans', 'Pickup Trucks', 'Wagon'])

with col2:
    st.subheader("Technical Specs")
    engine = st.number_input("🔧 Engine Capacity (CC)", 600, 5000, 1200)
    mileage = st.slider("⛽ Mileage (kmpl)", 5.0, 35.0, 18.0)
    power = st.number_input("⚡ Max Power (bhp)", 30.0, 500.0, 85.0)
    torque = st.number_input("🌀 Torque (Nm)", 50.0, 700.0, 150.0)
    seats = st.selectbox("💺 Seating Capacity", [2, 4, 5, 7, 8])

st.markdown("---")

# 5. Prediction Logic
if st.button("Generate Valuation Estimate"):
    with st.spinner('Analyzing market trends and predicting price...'):
        # Prepare Encoding
        city_enc = le_city.transform([city])[0]
        brand_enc = le_brand.transform([brand])[0]
        model_enc = le_model.transform([car_model])[0]

        # Construct 24-feature DataFrame
        input_dict = {
            'city': city_enc, 'owner_count': owners, 'brand': brand_enc, 'model': model_enc,
            'car_age': age, 'km_numeric': km, 'engine_cc': engine, 'mileage_numeric': mileage,
            'max_power_numeric': power, 'torque_numeric': torque, 'spec_seats': seats,
            'fuel_type_Diesel': 1 if fuel == 'Diesel' else 0,
            'fuel_type_Electric': 1 if fuel == 'Electric' else 0,
            'fuel_type_Lpg': 1 if fuel == 'Lpg' else 0,
            'fuel_type_Petrol': 1 if fuel == 'Petrol' else 0,
            'transmission_Manual': 1 if trans == 'Manual' else 0,
            'body_type_Coupe': 1 if body_type == 'Coupe' else 0,
            'body_type_Hatchback': 1 if body_type == 'Hatchback' else 0,
            'body_type_MUV': 1 if body_type == 'MUV' else 0,
            'body_type_Minivans': 1 if body_type == 'Minivans' else 0,
            'body_type_Pickup Trucks': 1 if body_type == 'Pickup Trucks' else 0,
            'body_type_SUV': 1 if body_type == 'SUV' else 0,
            'body_type_Sedan': 1 if body_type == 'Sedan' else 0,
            'body_type_Wagon': 1 if body_type == 'Wagon' else 0
        }

        input_df = pd.DataFrame([input_dict])

        # Scaling
        num_cols = ['owner_count', 'car_age', 'km_numeric', 'engine_cc', 'mileage_numeric', 'max_power_numeric', 'torque_numeric', 'spec_seats']
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        # Model Prediction
        prediction = model.predict(input_df)[0]
        
        # Display Result with Metrics
        st.markdown(f"""
            <div class="prediction-card">
                <h2 style='color: #ff4b4b;'>Estimated Market Value</h2>
                <h1 style='font-size: 50px;'>₹ {prediction:,.2f}</h1>
            </div>
        """, unsafe_allow_html=True)

        # 6. Aesthetic Insight: Plotly Chart
        st.subheader("Price Insight: Impact of Car Age")
        # Visualizing a trend: how age typically affects price for this model
        ages = np.arange(0, 21)
        temp_df = pd.concat([input_df]*len(ages), ignore_index=True)
        # Update ages and re-scale for trend visualization
        temp_df['car_age'] = (ages - np.mean(ages)) / np.std(ages) # Approximate scaling for visual
        trend_preds = model.predict(temp_df)
        
        fig = px.line(x=ages, y=trend_preds, labels={'x': 'Age of Car', 'y': 'Predicted Price'},
                     title=f"How price depreciates for {brand} {car_model} over time")
        fig.update_traces(line_color='#ff4b4b')
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Developed for CarDekho Used Car Price Prediction Project | Powered by XGBoost 🚀")