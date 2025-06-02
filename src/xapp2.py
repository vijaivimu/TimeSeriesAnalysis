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

st.markdown("""
### 🧠 Model Overview: XGBoost Regressor
This app uses an **XGBoost (Extreme Gradient Boosting)** regression model, a powerful machine learning algorithm known for its accuracy and efficiency.  
The model was trained on historical energy consumption data with lagged features to forecast the **next hour's energy usage**.   
""")

st.sidebar.header("Input: Last 10 Hours")

# 3-column input layout: Energy, Temperature, Occupancy using sliders
ec_lags, temp_lags, occ_lags = [], [], []
for i in range(1, 11):
    with st.sidebar.container():
        st.markdown(f"**Hour -{i}**")
        cols = st.columns(3)
        ec = cols[0].slider("EC", min_value=0.0, max_value=100.0, value=75.0, key=f"ec_{i}")
        temp = cols[1].slider("Temp", min_value=20.0, max_value=32.0, value=22.0, key=f"temp_{i}")
        occ = cols[2].slider("Occ", min_value=0.0, max_value=15.0, value=2.0, key=f"occ_{i}")
        ec_lags.append(ec)
        temp_lags.append(temp)
        occ_lags.append(occ)

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
        input_df = input_df[model.feature_names_in_]
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted Energy Consumption (next hour): **{prediction:.2f} kWh**")

        # Prepare data for chart
        hours = [f"-{i}h" for i in range(10, 0, -1)] + ["+1h"]
        values = ec_lags + [prediction]
        types = ["Actual"] * 10 + ["Prediction"]

        chart_df = pd.DataFrame({
            "Hour": pd.Categorical(hours, categories=hours, ordered=True),
            "EnergyConsumption": values,
            "Type": types
        })

        # Main chart
        line = alt.Chart(chart_df).mark_line(point=True).encode(
            x=alt.X('Hour', title='Time'),
            y=alt.Y('EnergyConsumption', title='Energy Consumption (kWh)'),
            color=alt.Color('Type', scale=alt.Scale(domain=['Actual', 'Prediction'], range=['#1f77b4', '#ff7f0e'])),
            tooltip=['Hour', 'EnergyConsumption', 'Type']
        )

        # Connector between -1h and +1h
        connector = pd.DataFrame({
            "Hour": pd.Categorical(["-1h", "+1h"], categories=hours, ordered=True),
            "EnergyConsumption": [ec_lags[-1], prediction],
            "Type": ["Prediction", "Prediction"]
        })

        connect_line = alt.Chart(connector).mark_line(point=True).encode(
            x='Hour',
            y='EnergyConsumption',
            color=alt.value("#ff7f0e")
        )

        final_chart = (line + connect_line).properties(
            title="📊 Last 10 Hours + Next Hour Prediction"
        )

        st.altair_chart(final_chart, use_container_width=True)

    except ValueError as e:
        st.error(f"Prediction failed due to feature mismatch: {e}")