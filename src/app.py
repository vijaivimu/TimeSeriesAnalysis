import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# Load your trained model
model = joblib.load('xgb_model.pkl')  # Ensure this file is in your app directory

st.title("Energy Consumption Forecasting App")
st.markdown("Predict the next hour's energy consumption using the last 10 hours of data.")

st.sidebar.header("Manual Input: Last 10 Hours")

# Input sliders for EC_lag_1 to EC_lag_10
ec_lags = [st.sidebar.number_input(f"Energy Consumption lag {i}", value=75.0) for i in range(1, 11)]

# Input sliders for Temp_lag_1 to Temp_lag_10
temp_lags = [st.sidebar.number_input(f"Temperature lag {i}", value=22.0) for i in range(1, 11)]

# Input sliders for Occ_lag_1 to Occ_lag_10
occ_lags = [st.sidebar.number_input(f"Occupancy lag {i}", value=50.0) for i in range(1, 11)]

# Add dummy current values for Temperature and Occupancy (required due to training feature shape)
current_temp = st.sidebar.number_input("Current Temperature", value=22.0)
current_occ = st.sidebar.number_input("Current Occupancy", value=50.0)

# Assemble input feature vector
input_data = [current_temp, current_occ] + ec_lags + temp_lags + occ_lags

feature_names = [
    'Temperature', 'Occupancy'
] + [
    f'EC_lag_{i}' for i in range(1, 11)
] + [
    f'Temp_lag_{i}' for i in range(1, 11)
] + [
    f'Occ_lag_{i}' for i in range(1, 11)
]

input_df = pd.DataFrame([input_data], columns=feature_names)

# Run prediction
if st.button("Predict Energy Consumption"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Energy Consumption (next hour): {prediction:.2f} kWh")

    # Optional chart (show last 10 EC values)
    st.line_chart(pd.Series(ec_lags[::-1], index=[f'-{i}h' for i in range(10, 0, -1)]))