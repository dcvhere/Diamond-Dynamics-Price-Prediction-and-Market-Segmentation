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

# This list contains ONLY the features selected by RFE during training
feature_order = models['feature_order'] 

# --- UI Design: Dynamic Sidebar ---
st.sidebar.header("Diamond Specifications")
st.sidebar.markdown("*Inputs dynamically adapted based on optimal training features.*")

input_data = {}

# Dynamically render ONLY the inputs that the model requires
if 'carat' in feature_order:
    input_data['carat'] = st.sidebar.number_input("Carat", min_value=0.1, max_value=10.0, value=1.00, step=0.01)

if 'cut_encoded' in feature_order:
    cut = st.sidebar.selectbox("Cut", options=list(cut_map.keys()))
    input_data['cut_encoded'] = cut_map[cut]

if 'color_encoded' in feature_order:
    color = st.sidebar.selectbox("Color", options=list(color_map.keys()))
    input_data['color_encoded'] = color_map[color]

if 'clarity_encoded' in feature_order:
    clarity = st.sidebar.selectbox("Clarity", options=list(clarity_map.keys()))
    input_data['clarity_encoded'] = clarity_map[clarity]

if 'depth' in feature_order:
    # Capped at realistic limits to prevent K-Means distance hijacking
    input_data['depth'] = st.sidebar.number_input("Depth %", min_value=40.0, max_value=75.0, value=61.50, step=0.1)

if 'table' in feature_order:
    input_data['table'] = st.sidebar.number_input("Table", min_value=40.0, max_value=75.0, value=57.00, step=0.1)

if 'x' in feature_order:
    input_data['x'] = st.sidebar.number_input("Length (x) in mm", min_value=0.1, max_value=15.0, value=6.00, step=0.01)

if 'y' in feature_order:
    input_data['y'] = st.sidebar.number_input("Width (y) in mm", min_value=0.1, max_value=60.0, value=6.00, step=0.01)

if 'z' in feature_order:
    input_data['z'] = st.sidebar.number_input("Depth (z) in mm", min_value=0.1, max_value=35.0, value=4.00, step=0.01)

# --- Main Interface ---
st.title("💎 Diamond Dynamics: Pricing & Segmentation")
st.markdown("Predict the market value and segment of a diamond based on its physical attributes.")

# Debugging expander to verify active features
with st.expander("View Active Model Features (Debugging)"):
    st.write(f"The model is currently utilizing these {len(feature_order)} optimized features for prediction:")
    st.code(feature_order)

if st.sidebar.button("Predict"):
    # 1. Build DataFrame enforcing the exact structure from training
    input_df = pd.DataFrame([input_data])[feature_order]
    
    # 2. Apply the scaler
    input_scaled = scaler.transform(input_df)
    
    # 3. Generate Predictions
    predicted_price_log = reg_model.predict(input_scaled)[0]
    predicted_price_inr = np.expm1(predicted_price_log) # Reverse the log transformation
    
    predicted_cluster = cluster_model.predict(input_scaled)[0]
    segment_name = cluster_names.get(predicted_cluster, "Unknown Segment")
    
    # 4. Display Results
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"### Predicted Price\n# ₹{predicted_price_inr:,.2f}")
        
    with col2:
        st.info(f"### Market Segment\n# {segment_name}")
