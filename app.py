import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os # Import os for better file handling checks

# Define the model file name
MODEL_FILE = 'EV_full_pipeline_SP.pkl'

# ------------------- 1. PAGE CONFIGURATION (MUST BE FIRST STREAMLIT COMMAND) -------------------
st.set_page_config(
    page_title="EV Price Predictor",
    page_icon="⚡",
    layout="wide"
)

# ------------------- 2. LOAD MODEL SAFELY -------------------
@st.cache_resource
def load_model(file_name):
    """Loads the pickled model with caching."""
    if not os.path.exists(file_name):
        st.error(f"❌ Model file '{file_name}' not found. Please ensure it’s in the app directory.")
        st.stop()
        
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"⚠️ Error loading model. The pipeline structure might be incompatible: {e}")
        st.stop()

# Load the model outside the prediction button to prevent reloading on every interaction
model = load_model(MODEL_FILE)

# ------------------- 3. APP STYLING AND HEADER -------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #e6f7ff; /* Light blue background */
        padding: 2rem;
        border-radius: 15px;
    }
    .title {
        text-align: center;
        color: #004c99; /* Dark blue title */
        font-family: 'Arial', sans-serif;
    }
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #004c99;
    }
    /* Custom styling for the Predict button */
    div.stButton > button:first-child {
        background-color: #004c99;
        color: white;
        font-size: 1.2em;
        font-weight: bold;
        border-radius: 12px;
        border: none;
        padding: 10px 20px;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px rgba(0, 76, 153, 0.3);
    }
    div.stButton > button:first-child:hover {
        background-color: #0066cc;
        transform: translateY(-2px);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>⚡ Electric Vehicle (EV) Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("Predict the market price (€) of an Electric Vehicle based on its technical specifications and features.", unsafe_allow_html=True)
st.divider()

# ------------------- 4. INFERRED FEATURE LISTS -------------------
# (These lists are taken directly from your code)
BRAND_OPTIONS = [
    'Aiways', 'Audi', 'BMW', 'Byton', 'CUPRA', 'Citroen', 'Ford', 'Honda', 'Hyundai', 'Kia', 
    'Lexus', 'Lightyear', 'MG', 'Mazda', 'Mercedes', 'Mini', 'Nissan', 'Opel', 'Peugeot', 
    'Polestar', 'Porsche', 'Renault', 'SEAT', 'Skoda', 'Smart', 'Sono', 'Tesla', 
    'Volkswagen', 'Volvo'
]
RAPID_CHARGE_OPTIONS = ['No', 'Yes']
POWERTRAIN_OPTIONS = ['AWD', 'FWD', 'RWD']
PLUG_TYPE_OPTIONS = ['Type 1 CHAdeMO', 'Type 2', 'Type 2 CCS', 'Type 2 CHAdeMO']
BODY_STYLE_OPTIONS = ['Cabrio', 'Hatchback', 'Liftback', 'MPV', 'Pickup', 'SPV', 'SUV', 'Sedan', 'Station']
SEGMENT_OPTIONS = ['A', 'B', 'C', 'D', 'E', 'F', 'N', 'S']

# ------------------- 5. USER INPUT SECTION -------------------
st.header("⚙️ Enter Vehicle Specifications")

# Group 1: Core Performance and Efficiency
st.subheader("Performance & Efficiency")
col1, col2, col3 = st.columns(3)

with col1:
    accel_sec = st.number_input(
        "Acceleration (0-100 Km/h in Sec)", 
        min_value=2.0, max_value=20.0, value=7.0, step=0.1, 
        format="%.1f", help="Lower is faster."
    )
    range_km = st.number_input(
        "Range (Km)", 
        min_value=100, max_value=1000, value=400, step=10,
        help="Official driving range on a single charge."
    )
with col2:
    efficiency_whkm = st.number_input(
        "Efficiency (Wh/Km)", 
        min_value=100, max_value=300, value=170, step=1,
        help="Energy consumed per kilometer (Lower is better)."
    )
    fast_charge_kmh = st.number_input(
        "Fast Charge Speed (Km/h)", 
        min_value=0, max_value=1000, value=400, step=10,
        help="Km of range added per hour using a fast charger."
    )
    
with col3:
    rapid_charge = st.selectbox("Rapid Charge Support", RAPID_CHARGE_OPTIONS)
    plug_type = st.selectbox("Plug Type", PLUG_TYPE_OPTIONS)


st.subheader("Categorical Details")
col4, col5, col6 = st.columns(3)

with col4:
    brand = st.selectbox("Brand", BRAND_OPTIONS)
    
with col5:
    powertrain = st.selectbox("Powertrain", POWERTRAIN_OPTIONS)
    segment = st.selectbox("Segment (Market Classification)", SEGMENT_OPTIONS)
    
with col6:
    body_style = st.selectbox("Body Style", BODY_STYLE_OPTIONS)


st.divider()

# ------------------- 6. PREDICTION LOGIC (FIXED) -------------------
if st.button("💰 Predict EV Price", use_container_width=True):
    # Data is collected into a dictionary
    input_data = {
        'AccelSec': [accel_sec],
        'Range_Km': [range_km],
        'Efficiency_WhKm': [efficiency_whkm],
        'FastCharge_KmH': [fast_charge_kmh],
        'Brand': [brand],
        'RapidCharge': [rapid_charge],
        'PowerTrain': [powertrain],
        'PlugType': [plug_type],
        'BodyStyle': [body_style],
        'Segment': [segment]
    }
    
    # Create DataFrame and ensure correct column order (CRUCIAL)
    input_df = pd.DataFrame(input_data)
    COLUMNS_ORDER = [
        'AccelSec', 'Range_Km', 'Efficiency_WhKm', 'FastCharge_KmH', 
        'Brand', 'RapidCharge', 'PowerTrain', 'PlugType', 
        'BodyStyle', 'Segment'
    ]
    input_df = input_df[COLUMNS_ORDER]

    with st.spinner("Calculating the EV market price... ⏳"):
        try:
            # 1. Predict the price (Output is log(1+Price))
            log1p_price = model.predict(input_df)[0]
            
            # 2. CRITICAL STEP: INVERSE TRANSFORMATION (np.expm1)
            # This reverts the np.log1p() applied to your target variable (y).
            predicted_price_actual = np.expm1(log1p_price)
            
            # Handle potential negative prices from Linear Regression
            if predicted_price_actual < 0:
                predicted_price_actual = 0
                
            # 3. Round to the nearest 100 Euros for a clean display
            predicted_price_rounded = np.round(predicted_price_actual / 100) * 100
            
            # Formatting the price
            formatted_price = f"{predicted_price_rounded:,.0f} €"

            # 4. Determine price tier for visual feedback
            if predicted_price_rounded >= 100000:
                color = "#008000"  # Green - Premium
                message = "Premium Segment EV"
                icon = "💎"
            elif predicted_price_rounded >= 45000:
                color = "#ff8c00"  # Dark Orange - Mid-Range
                message = "Mid-Range Market EV"
                icon = "📈"
            else:
                color = "#8b0000"  # Dark Red - Budget
                message = "Budget-Friendly EV"
                icon = "💸"

            # 5. Display the result in a striking block
            st.markdown(
                f"<div style='border: 3px solid {color}; padding: 25px; border-radius: 15px; background-color: #ffffff; text-align: center; margin-top: 20px;'>", 
                unsafe_allow_html=True
            )
            st.markdown(f"### Predicted Market Price", unsafe_allow_html=True)
            st.markdown(
                f"<h1 style='color: {color}; font-size: 3.5em; margin: 10px 0;'>{icon} {formatted_price}</h1>",
                unsafe_allow_html=True
            )
            st.markdown(f"<p style='color: #333333; font-weight: bold;'>{message}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


            with st.expander("📝 View Input Data"):
                st.dataframe(input_df, hide_index=True)

        except Exception as e:
            st.error(f"⚠️ An error occurred during prediction. Please check your inputs or the model pipeline: {e}")

# ------------------- 7. FOOTER -------------------
st.divider()
st.caption("🔹 EV Price Predictor | Model v1.0 | Note: The model was trained on a Linear Regression pipeline with Robust Scaling for outlier handling.")
