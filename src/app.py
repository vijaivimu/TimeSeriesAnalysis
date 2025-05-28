import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

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

if st.button("Predict Energy Consumption"):
    try:
        input_df = input_df[model.feature_names_in_]
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Energy Consumption (next hour): {prediction:.2f} kWh")

        # Prepare data
        actual_hours = [f"-{i}h" for i in range(10, 1, -1)]   # -10h to -2h
        actual_values = ec_lags[:-1]                         # exclude -1h
        last_actual_hour = "-1h"
        last_actual_value = ec_lags[-1]
        prediction_hour = "+1h"

        # DataFrames
        df_actual = pd.DataFrame({
            "Hour": pd.Categorical(actual_hours, categories=actual_hours + [last_actual_hour, prediction_hour], ordered=True),
            "EnergyConsumption": actual_values,
            "Type": "Actual"
        })

        df_pred_line = pd.DataFrame({
            "Hour": [last_actual_hour, prediction_hour],
            "EnergyConsumption": [last_actual_value, prediction],
            "Type": "Prediction"
        })

        full_df = pd.concat([df_actual, df_pred_line], ignore_index=True)

        # Altair chart
        line = alt.Chart(full_df).mark_line(point=True).encode(
            x=alt.X('Hour', title='Time'),
            y=alt.Y('EnergyConsumption', title='Energy Consumption (kWh)'),
            color=alt.Color('Type', scale=alt.Scale(domain=['Actual', 'Prediction'], range=['#1f77b4', '#ff7f0e'])),
            tooltip=['Hour', 'EnergyConsumption', 'Type']
        ).properties(
            title="Last 10 Hours + Next Hour Prediction"
        ).configure_axisX(
            labelAngle=0
        )

        st.altair_chart(line, use_container_width=True)

    except ValueError as e:
        st.error(f"Prediction failed due to feature mismatch: {e}")