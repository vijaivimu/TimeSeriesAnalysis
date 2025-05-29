import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the SARIMAX fitted model (results object)
results = joblib.load("sarimax_model.pkl")

st.title("Energy Forecast Heatmap")
st.markdown("Predict energy consumption based on **Temperature** and **Occupancy** using SARIMAX.")

# Sidebar input ranges
temp_range = st.slider("Select Temperature Range", min_value=10, max_value=40, value=(15, 30))
occ_range = st.slider("Select Occupancy Range", min_value=0, max_value=100, value=(0, 50), step=5)

temps = list(range(temp_range[0], temp_range[1] + 1))
occs = list(range(occ_range[0], occ_range[1] + 1, 5))

# Build prediction grid
data = []
for temp in temps:
    row = []
    for occ in occs:
        exog_input = pd.DataFrame({'Temperature': [temp], 'Occupancy': [occ]})
        
        # ✅ Use get_prediction instead of predict
        forecast_result = results.get_prediction(start=results.nobs, end=results.nobs, exog=exog_input)
        pred = forecast_result.predicted_mean.iloc[0]
        row.append(pred)
    data.append(row)

# Heatmap DataFrame
heatmap_df = pd.DataFrame(data, index=temps, columns=occs)

# Plot the heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="YlOrRd")
ax.set_xlabel("Occupancy")
ax.set_ylabel("Temperature")
ax.set_title("Predicted Energy Consumption (kWh)")

st.pyplot(fig)