import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        background-color: #1a73e8;
        color: white;
        font-size: 18px;
        padding: 12px;
        border-radius: 8px;
        border: none;
        margin-top: 10px;
    }
    .stButton>button:hover { background-color: #1558b0; }
    .result-box {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        color: white;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
    }
    .result-box h2 { font-size: 36px; margin: 0; }
    .result-box p  { font-size: 18px; margin: 8px 0 0; opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ── Load model and columns ────────────────────────────────────
@st.cache_resource
def load_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path   = os.path.join('model', 'house_price_model.pkl')
    columns_path = os.path.join('model', 'columns.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(columns_path, 'rb') as f:
        columns = pickle.load(f)
    return model, columns

model, columns = load_model()

locations = sorted([
    col.replace('location_', '')
    for col in columns if col.startswith('location_')
])

# ── UI ────────────────────────────────────────────────────────
st.title("🏠 Bangalore House Price Predictor")
st.markdown("Fill in the details below to get an instant price estimate.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    location   = st.selectbox("📍 Location", locations)
    total_sqft = st.number_input("📐 Total Sqft", min_value=300, max_value=10000, value=1000, step=50)
    balcony    = st.selectbox("🌿 Balconies", [0, 1, 2, 3])

with col2:
    bhk      = st.selectbox("🛏️ BHK", [1, 2, 3, 4, 5])
    bath     = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4, 5])
    is_ready = st.radio("🏗️ Availability", ["Ready To Move", "Under Construction"])

is_ready_val = 1 if is_ready == "Ready To Move" else 0

st.markdown("---")

if st.button("🔍 Predict Price"):
    input_df = pd.DataFrame([np.zeros(len(columns))], columns=columns)
    input_df['total_sqft'] = total_sqft
    input_df['bath']       = bath
    input_df['balcony']    = balcony
    input_df['bhk']        = bhk
    input_df['is_ready']   = is_ready_val

    loc_col = 'location_' + location
    if loc_col in columns:
        input_df[loc_col] = 1

    log_price = model.predict(input_df)[0]
    price     = np.expm1(log_price)

    st.markdown(f"""
    <div class="result-box">
        <h2>₹ {price:.2f} Lakhs</h2>
        <p>≈ ₹ {price/100:.2f} Crore &nbsp;|&nbsp; {location} &nbsp;|&nbsp; {bhk} BHK &nbsp;|&nbsp; {total_sqft} sqft</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Price Breakdown")
    c1, c2, c3 = st.columns(3)
    c1.metric("Price per sqft", f"₹ {(price * 100000 / total_sqft):,.0f}")
    c2.metric("Total (lakhs)",  f"₹ {price:.2f}L")
    c3.metric("Total (crore)",  f"₹ {price/100:.2f}Cr")

st.markdown("---")
st.caption("Model trained on Bengaluru House Price Dataset · R² = 0.75 · Built with scikit-learn & Streamlit")
