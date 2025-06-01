import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib



# Load SARIMAX model
results = joblib.load("sarimax_model.pkl")

# Page config
st.set_page_config(page_title="Watt-Wise Forecast", layout="wide")

# Create two columns (left = summary, right = inputs + output)
left_col, right_col = st.columns([1, 2])

# === LEFT COLUMN: App Description ===
with left_col:
    st.markdown("## 📘 App Overview")
    st.markdown("""
    **This app uses a pre-trained SARIMAX model** to predict energy consumption based on:

    - Temperature  
    - Occupancy  

    The model forecasts energy consumption **1-hour ahead** for various scenarios.

    ---
    ### 🧠 Model Summary
    - SARIMAX order: (1, 0, 1)
    - Trained on 80% of hourly data  
    - Uses exogenous features: Temperature, Occupancy  
    - Performance evaluated with RMSE on holdout set  

    ---
    ### 📊 Dataset
    **Source**: [Kaggle - Energy Consumption Prediction](https://www.kaggle.com/datasets/mrsimple07/energy-consumption-prediction)
    """)

# === RIGHT COLUMN: Image, Input, Heatmap ===
with right_col:
    st.image("bg.png", use_container_width=True)

    st.markdown("## ⚙️ Forecast Inputs")
    temp_range = st.slider("Temperature Range (°C)", 18, 32, (18, 32))
    occ_range = st.slider("Occupancy Range", 0, 15, (0, 15), step=1)

    temps = list(range(temp_range[0], temp_range[1] + 1))
    occs = list(range(occ_range[0], occ_range[1] + 1))

    # Generate forecast data
    data = []
    for temp in temps:
        row = []
        for occ in occs:
            exog_input = pd.DataFrame({'Temperature': [temp], 'Occupancy': [occ]})
            forecast_result = results.get_prediction(start=results.nobs, end=results.nobs, exog=exog_input)
            pred = forecast_result.predicted_mean.iloc[0]
            row.append(pred)
        data.append(row)

    # Heatmap
    heatmap_df = pd.DataFrame(data, index=temps, columns=occs)

    st.markdown("## 🔥 Predicted Energy Consumption (kWh)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="YlOrRd")
    ax.set_xlabel("Occupancy")
    ax.set_ylabel("Temperature")
    #ax.set_title("Predicted Energy Consumption (kWh)")
    st.pyplot(fig)