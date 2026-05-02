import streamlit as st
import numpy as np
import joblib


model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")


st.title("💎 Diamond Price Predictor")

# Numeric inputs
carat = st.number_input("Carat", min_value=0.0, step=0.1)
depth = st.number_input("Depth", min_value=0.0)
table = st.number_input("Table", min_value=0.0)
x = st.number_input("Length (x)", min_value=0.0)
y = st.number_input("Width (y)", min_value=0.0)
z = st.number_input("Depth (z)", min_value=0.0)

# Dropdown inputs
cut = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox("Color", ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

# Button
predict_btn = st.button("Predict Price")

# PREDICTION
# -------------------------

if predict_btn:

    # Encode categorical values
    input_data = [[cut, color, clarity]]
    encoded = encoder.transform(input_data)

    cut_enc, color_enc, clarity_enc = encoded[0]

    # Feature engineering (same as training)
    carat_log = np.log1p(carat)
    z_log = np.log1p(z)

    volume = x * y * z_log

    # Final feature order (MUST match training)
    features = np.array([[
        carat_log,
        cut_enc,
        color_enc,
        clarity_enc,
        depth,
        table,
        x,
        y,
        z_log,
        volume
    ]])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)

    st.success(f"💰 Estimated Price: ₹ {round(prediction[0], 2)}")



kmeans_model = joblib.load("kmeans_model.pkl")
kmeans_scaler = joblib.load("kmeans_scaler.pkl")
kmeans_encoder = joblib.load("kmeans_encoder.pkl")

# -------------------------
# CLUSTER PREDICTION BUTTON
# -------------------------

cluster_btn = st.button("Predict Market Category")

if cluster_btn:

    # Encode categorical features (use kmeans encoder)
    cluster_encoded = kmeans_encoder.transform([[cut, color, clarity]])
    cut_c, color_c, clarity_c = cluster_encoded[0]

    # Prepare features (NO log, match training)
    cluster_features = np.array([[
        carat,
        cut_c,
        color_c,
        clarity_c,
        depth,
        table,
        x,
        y,
        z
    ]])

    # Scale
    cluster_scaled = kmeans_scaler.transform(cluster_features)

    # Predict cluster
    cluster = kmeans_model.predict(cluster_scaled)[0]

    # Label mapping (⚠️ update if you saved dynamic mapping)
    cluster_map = {
        0: 'Affordable Small Diamonds',
        1: 'Mid-range Balanced Diamonds',
        2: 'Premium Heavy Diamonds'
    }

    cluster_label = cluster_map.get(cluster, "Unknown")

    # Output
    st.info(f"💎 Market Category: {cluster_label}")
