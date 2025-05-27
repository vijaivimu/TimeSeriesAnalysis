import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model
model = joblib.load('xgb_model.pkl')

st.title("Energy Consumption Forecasting App")
st.markdown("🔮 Predict the next hour's energy consumption using the last 10 hours of Energy, Temperature, and Occupancy data.")

st.sidebar.header("Manual Input: Last 10 Hours")

# Collect all EC lags first
ec_lags = []
temp_lags = []
occ_lags = []

for i in range(1, 11):
    ec = st.sidebar.number_input(f"Energy Consumption lag {i}", value=75.0)
    temp = st.sidebar.number_input(f"Temperature lag {i}", value=22.0)
    occ = st.sidebar.number_input(f"Occupancy lag {i}", value=50.0)

    ec_lags.append(ec)
    temp_lags.append(temp)
    occ_lags.append(occ)

# Concatenate in correct order
input_data = ec_lags + temp_lags + occ_lags

# Feature names must match training order
feature_names = (
    [f'EC_lag_{i}' for i in range(1, 11)] +
    [f'Temp_lag_{i}' for i in range(1, 11)] +
    [f'Occ_lag_{i}' for i in range(1, 11)]
)

input_df = pd.DataFrame([input_data], columns=feature_names)

# Run prediction
if st.button("Predict Energy Consumption"):
    try:
        input_df = input_df[model.feature_names_in_]  # 🔐 Match training feature order
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Energy Consumption (next hour): {prediction:.2f} kWh")
        st.line_chart(pd.Series(ec_lags[::-1], index=[f'-{i}h' for i in range(10, 0, -1)]))
    except ValueError as e:
        st.error(f"Prediction failed due to feature mismatch: {e}")