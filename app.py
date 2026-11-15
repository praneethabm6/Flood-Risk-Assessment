import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="FloodSafe | AI Flood Risk & Disaster Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .hero-box {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border-left: 6px solid #00d2ff;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        background: -webkit-linear-gradient(#ffffff, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cfd8dc;
        margin-top: 0.4rem;
    }
    .factor-card {
        background-color: #1e2530;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #2e3846;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .badge-low {
        background-color: #1b4332; color: #74c69d;
        padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;
    }
    .badge-mod {
        background-color: #5c4018; color: #ffd166;
        padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;
    }
    .badge-high {
        background-color: #59181d; color: #ff758f;
        padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Trained Model ---
model_path = "models/best_flood_model.pkl"
model = joblib.load(model_path) if os.path.exists(model_path) else None

# --- Helper Functions for Weather API & Classification ---
def fetch_live_weather(city_name, api_key):
    """Fetches real-time temperature and humidity from OpenWeatherMap API"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            humidity = data['main']['humidity']
            # Estimate rainfall or default if not raining heavily
            rain = data.get('rain', {}).get('1h', 15.0) * 10  # Scale up for simulation if present
            return humidity, max(rain, 20.0), True
    except Exception as e:
        pass
    return 60.0, 150.0, False

def classify_rainfall(val):
    if val < 150: return "Low", "badge-low"
    elif val < 280: return "Moderate", "badge-mod"
    else: return "High", "badge-high"

def classify_discharge(val):
    if val < 700: return "Low", "badge-low"
    elif val < 1400: return "Moderate", "badge-mod"
    else: return "High", "badge-high"

def classify_elevation(val):
    if val > 300: return "Low", "badge-low"
    elif val > 120: return "Moderate", "badge-mod"
    else: return "High", "badge-high"

def classify_moisture(val):
    if val < 45: return "Low", "badge-low"
    elif val < 75: return "Moderate", "badge-mod"
    else: return "High", "badge-high"

# --- Sidebar: Location & Weather API Telemetry ---
st.sidebar.markdown("## 📡 Telemetry & Location")
location_mode = st.sidebar.radio("Input Source:", ["Live Weather API & City", "Regional Presets", "Manual Sensor Override"])

# OpenWeatherMap API Key Input (You can paste yours here or leave blank to fall back to smart presets)
api_key = st.sidebar.text_input("OpenWeatherMap API Key (Optional)", type="password")

if location_mode == "Live Weather API & City":
    target_city = st.sidebar.text_input("Enter City Name:", "Bengaluru")
    
    # Try fetching live data if API key is provided
    live_humidity, live_rain, success = (60.0, 150.0, False)
    if api_key:
        live_humidity, live_rain, success = fetch_live_weather(target_city, api_key)
    
    if success:
        st.sidebar.success(f"🌐 Live Weather Connected for {target_city}!")
    else:
        st.sidebar.info(f"ℹ️ Using estimated telemetry for {target_city}. (Add API key for live feed)")
        
    rainfall = st.sidebar.slider("🌧️ Rainfall (mm)", 50.0, 400.0, float(live_rain))
    soil_moisture = st.sidebar.slider("💧 Soil Moisture / Humidity (%)", 20.0, 95.0, float(live_humidity))
    river_discharge = st.sidebar.slider("🌊 River Discharge (m³/s)", 100.0, 2000.0, 500.0)
    elevation = st.sidebar.slider("⛰️ Elevation (meters)", 5.0, 500.0, 150.0)

elif location_mode == "Regional Presets":
    city = st.sidebar.selectbox("Select Monitored Zone:", ["Bengaluru, Karnataka", "Mangaluru Coastal Zone", "Kodagu Hill Basin", "Mysuru Cauvery Basin"])
    st.sidebar.success(f"📍 Active Node: **{city}**")
    
    presets = {
        "Bengaluru, Karnataka": (180.0, 500.0, 150.0, 60.0),
        "Mangaluru Coastal Zone": (320.0, 1550.0, 25.0, 85.0),
        "Kodagu Hill Basin": (290.0, 1200.0, 380.0, 78.0),
        "Mysuru Cauvery Basin": (120.0, 420.0, 260.0, 42.0)
    }
    def_rain, def_dis, def_elev, def_moist = presets[city]
    
    rainfall = st.sidebar.slider("🌧️ Rainfall (mm)", 50.0, 400.0, def_rain)
    river_discharge = st.sidebar.slider("🌊 River Discharge (m³/s)", 100.0, 2000.0, def_dis)
    elevation = st.sidebar.slider("⛰️ Elevation (meters)", 5.0, 500.0, def_elev)
    soil_moisture = st.sidebar.slider("💧 Soil Moisture (%)", 20.0, 95.0, def_moist)
else:
    st.sidebar.markdown("Enter custom sensor readings:")
    rainfall = st.sidebar.number_input("🌧️ Rainfall (mm)", 0.0, 500.0, 150.0)
    river_discharge = st.sidebar.number_input("🌊 River Discharge (m³/s)", 0.0, 3000.0, 450.0)
    elevation = st.sidebar.number_input("⛰️ Elevation (meters)", 0.0, 1000.0, 200.0)
    soil_moisture = st.sidebar.number_input("💧 Soil Moisture (%)", 0.0, 100.0, 55.0)

st.sidebar.markdown("---")

# --- MAIN PAGE LAYOUT ---
st.markdown("""
    <div class="hero-box">
        <div class="hero-title">🛡️ FloodSafe</div>
        <div class="hero-subtitle">AI-Powered Flood Risk Intelligence, Factor Attribution & Disaster Assistance Platform</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("### 📊 Contributing Environmental Factors")
st.caption("Individual parameter risk contribution based on real-time hydrological thresholds:")

r_lvl, r_cls = classify_rainfall(rainfall)
d_lvl, d_cls = classify_discharge(river_discharge)
e_lvl, e_cls = classify_elevation(elevation)
m_lvl, m_cls = classify_moisture(soil_moisture)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="factor-card">
            <div style="font-size:1.5rem;">🌧️</div>
            <div style="color:#90a4ae; font-size:0.9rem;">Rainfall</div>
            <div style="font-size:1.4rem; font-weight:700; margin:6px 0;">{rainfall:.1f} mm</div>
            <span class="{r_cls}">{r_lvl} Contribution</span>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="factor-card">
            <div style="font-size:1.5rem;">🌊</div>
            <div style="color:#90a4ae; font-size:0.9rem;">River Discharge</div>
            <div style="font-size:1.4rem; font-weight:700; margin:6px 0;">{river_discharge:.0f} m³/s</div>
            <span class="{d_cls}">{d_lvl} Contribution</span>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="factor-card">
            <div style="font-size:1.5rem;">⛰️</div>
            <div style="color:#90a4ae; font-size:0.9rem;">Terrain Elevation</div>
            <div style="font-size:1.4rem; font-weight:700; margin:6px 0;">{elevation:.0f} m</div>
            <span class="{e_cls}">{e_lvl} Risk Factor</span>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="factor-card">
            <div style="font-size:1.5rem;">💧</div>
            <div style="color:#90a4ae; font-size:0.9rem;">Soil Moisture</div>
            <div style="font-size:1.4rem; font-weight:700; margin:6px 0;">{soil_moisture:.1f} %</div>
            <span class="{m_cls}">{m_lvl} Contribution</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Prediction & Explainable Output
if model is None:
    st.error("⚠️ Trained model not found! Please run `python train_model.py` first.")
else:
    input_data = pd.DataFrame(
        [[rainfall, river_discharge, elevation, soil_moisture]],
        columns=['rainfall_mm', 'river_discharge', 'elevation_m', 'soil_moisture_%']
    )
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    
    res_col, chart_col = st.columns([1.2, 1])
    
    with res_col:
        st.markdown("### 🎯 Overall FloodSafe Verdict")
        if prediction == "Very High":
            st.error(f"### 🚨 CRITICAL ALERT: {prediction.upper()} RISK")
        elif prediction == "High":
            st.warning(f"### ⚠️ WARNING: {prediction.upper()} RISK")
        elif prediction == "Moderate":
            st.info(f"### 🔍 ADVISORY: {prediction.upper()} RISK")
        else:
            st.success(f"### ✅ SAFE: {prediction.upper()} RISK")
            
        st.markdown("#### 🛡️ Actionable Disaster Assistance")
        if prediction in ["High", "Very High"]:
            st.markdown("""
            - **Immediate Evacuation:** Move immediately to designated high-ground shelters.
            - **Utility Shutdown:** Disconnect electrical mains and close gas valves before leaving.
            - **Emergency Kit:** Carry waterproofed IDs, medicines, flashlight, and drinking water.
            - **Emergency Helpline:** Contact State Disaster Response Force (SDRF) / Dial **112**.
            """)
        elif prediction == "Moderate":
            st.markdown("""
            - **Active Monitoring:** Keep alerts enabled for sudden river discharge or rainfall spikes.
            - **Drainage Check:** Ensure local storm drains and home gutters are clear of debris.
            - **Readiness:** Keep power banks charged and move ground-floor valuables higher.
            """)
        else:
            st.markdown("""
            - **Normal Operations:** Environmental parameters are within safe thresholds.
            - **Routine Preparedness:** Keep emergency contacts accessible and review local evacuation routes.
            """)

    with chart_col:
        st.markdown("### 📈 AI Confidence & Risk Breakdown")
        prob_df = pd.DataFrame({
            "Risk Category": classes,
            "Confidence (%)": np.round(probabilities * 100, 1)
        }).set_index("Risk Category")
        st.bar_chart(prob_df, color="#00d2ff")
        st.caption("Model probability distribution across all four flood severity classes.")