import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# Load the trained model
model = joblib.load("xgb_model.pkl")

st.set_page_config(page_title="Energy Forecasting", layout="wide")

st.title("⚡ Energy Consumption Forecasting App")
st.markdown("🔍 Predict the next hour's energy consumption using the last 10 hours of Energy, Temperature, and Occupancy data.")

st.sidebar.header("Input Features")

# User inputs for 10 lags of each feature
ec_lags = [st.sidebar.slider(f"Energy Consumption lag {i}", 0.0, 150.0, 75.0, key=f"ec{i}") for i in range(1, 11)]
temp_lags = [st.sidebar.slider(f"Temperature lag {i}", 0.0, 50.0, 22.0, key=f"temp{i}") for i in range(1, 11)]
occ_lags = [st.sidebar.slider(f"Occupancy lag {i}", 0.0, 100.0, 50.0, key=f"occ{i}") for i in range(1, 11)]

# Combine inputs into DataFrame for prediction
input_data = ec_lags + temp_lags + occ_lags
feature_names = (
    [f'EC_lag_{i}' for i in range(1, 11)] +
    [f'Temp_lag_{i}' for i in range(1, 11)] +
    [f'Occ_lag_{i}' for i in range(1, 11)]
)
input_df = pd.DataFrame([input_data], columns=feature_names)

# Prediction and visualization
if st.button("🎯 Predict Energy Consumption"):
    try:
        # Ensure only expected features are passed
        input_df = input_df[model.feature_names_in_]
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted Energy Consumption (next hour): **{prediction:.2f} kWh**")

        # Prepare data for chart
        hours = [f"-{i}h" for i in range(10, 0, -1)] + ["+1h"]
        values = ec_lags + [prediction]
        types = ["Actual"] * 10 + ["Prediction"]

        chart_df = pd.DataFrame({
            "Hour": hours,
            "EnergyConsumption": values,
            "Type": types
        })

        # Define x-axis order
        hour_order = hours

        # Base line chart
        line = alt.Chart(chart_df).mark_line(point=True).encode(
            x=alt.X('Hour', sort=hour_order, title='Time'),
            y=alt.Y('EnergyConsumption', title='Energy Consumption (kWh)'),
            color=alt.Color('Type', scale=alt.Scale(domain=['Actual', 'Prediction'], range=['#1f77b4', '#ff7f0e'])),
            tooltip=['Hour', 'EnergyConsumption', 'Type']
        )

        # Connector line between last actual and prediction
        connector = pd.DataFrame({
            "Hour": ["-1h", "+1h"],
            "EnergyConsumption": [ec_lags[-1], prediction],
            "Type": ["Prediction", "Prediction"]
        })

        connect_line = alt.Chart(connector).mark_line(point=True).encode(
            x=alt.X("Hour", sort=hour_order),
            y="EnergyConsumption",
            color=alt.value("#ff7f0e")
        )

        # Combine charts
        final_chart = (line + connect_line).properties(
            title="📊 Last 10 Hours + Next Hour Prediction"
        )

        st.altair_chart(final_chart, use_container_width=True)

    except ValueError as e:
        st.error(f"Prediction failed due to feature mismatch: {e}")