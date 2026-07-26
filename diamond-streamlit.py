import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Configuration ---
st.set_page_config(page_title="Diamond Dynamics", page_icon="💎", layout="wide")

# --- Load Models & Preprocessors ---
@st.cache_resource
def load_models():
    with open('diamond_models.pkl', 'rb') as f:
        return pickle.load(f)

try:
    models = load_models()
except FileNotFoundError:
    st.error("Model file 'diamond_models.pkl' not found. Please ensure it is in the same directory as app.py.")
    st.stop()

# Extract model components
reg_model = models['reg_model']
cluster_model = models['cluster_model']
scaler = models['scaler']
cut_map = models['cut_map']
color_map = models['color_map']
clarity_map = models['clarity_map']
cluster_names = models['cluster_names']
# feature_order now strictly contains only the features that survived RFE selection
feature_order = models['feature_order'] 

# --- UI Design: Sidebar for Inputs ---
st.sidebar.header("Diamond Specifications")

# We collect all 9 possible inputs to maintain a comprehensive UI.
# If RFE dropped some (like x, y, z), they will be safely ignored during the prediction phase.
carat = st.sidebar.number_input("Carat", min_value=0.1, max_value=10.0, value=0.50, step=0.01)
cut = st.sidebar.selectbox("Cut", options=list(cut_map.keys()))
color = st.sidebar.selectbox("Color", options=list(color_map.keys()))
clarity = st.sidebar.selectbox("Clarity", options=list(clarity_map.keys()))
depth = st.sidebar.number_input("Depth %", min_value=40.0, max_value=85.0, value=61.50, step=0.1)
table = st.sidebar.number_input("Table", min_value=40.0, max_value=95.0, value=57.00, step=0.1)
x = st.sidebar.number_input("Length (x) in mm", min_value=0.1, max_value=15.0, value=5.00, step=0.01)
y = st.sidebar.number_input("Width (y) in mm", min_value=0.1, max_value=60.0, value=5.00, step=0.01)
z = st.sidebar.number_input("Depth (z) in mm", min_value=0.1, max_value=35.0, value=3.00, step=0.01)

# --- Main Interface ---
st.title("💎 Diamond Dynamics: Pricing & Segmentation")
st.markdown("Predict the market value and segment of a diamond based on its physical attributes.")

if st.sidebar.button("Predict"):
    # 1. Map all inputs into a dictionary, encoding the categorical variables immediately
    input_data = {
        'carat': carat,
        'cut_encoded': cut_map[cut],
        'color_encoded': color_map[color],
        'clarity_encoded': clarity_map[clarity],
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z
    }
    
    # 2. Build the DataFrame and dynamically filter columns
    # Passing [feature_order] forces pandas to keep only the columns the model expects,
    # safely discarding the rest to prevent dimension errors.
    input_df = pd.DataFrame([input_data])[feature_order]
    
    # 3. Apply standard scaling
    input_scaled = scaler.transform(input_df)
    
    # 4. Generate Predictions
    # The regression target was log-transformed during training to handle skewness.
    # We must reverse this using np.expm1() to output actual INR.
    predicted_price_log = reg_model.predict(input_scaled)[0]
    predicted_price_inr = np.expm1(predicted_price_log)
    
    predicted_cluster = cluster_model.predict(input_scaled)[0]
    segment_name = cluster_names.get(predicted_cluster, "Unknown Segment")
    
    # 5. Display Results
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"### Predicted Price\n# ₹{predicted_price_inr:,.2f}")
        
    with col2:
        st.info(f"### Market Segment\n# {segment_name}")
