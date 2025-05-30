import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the SARIMAX model
results = joblib.load("sarimax_model.pkl")

# Page config
st.set_page_config(page_title="Energy Heatmap", layout="wide")

# Display top background image
st.image("/Users/vimu/Documents/Data Science/SDS/Github/TimeSeriesAnalysis/src/energy consumption bg.png", use_column_width=True)

# Title
st.title("⚡ Energy Consumption Heatmap Forecast")

# Create two columns
left_col, right_col = st.columns([1, 2])

# ==== Left Side: Model Info ====
with left_col:
    st.subheader("📘 Model & Dataset Info")
    st.markdown("""
    **Model**: SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)  
    **Target**: Energy Consumption  
    **Exogenous Variables**: Temperature, Occupancy  
    **Training Size**: 80% of the dataset  
    **Evaluation Metric**: RMSE  

    📊 **Data Source**:  
    [Kaggle - Energy Consumption Prediction](https://www.kaggle.com/datasets/mrsimple07/energy-consumption-prediction)
    """)
    st.markdown("---")
    st.markdown("This model uses the latest **Temperature** and **Occupancy** values to estimate the energy consumption for the next time period. Adjust the sliders on the right to explore predictions.")

# ==== Right Side: Input & Output ====
with right_col:
    st.subheader("🎛️ Predict with Temperature & Occupancy")

    temp_range = st.slider("Select Temperature Range", min_value=10, max_value=40, value=(15, 30))
    occ_range = st.slider("Select Occupancy Range", min_value=0, max_value=100, value=(0, 50), step=5)

    temps = list(range(temp_range[0], temp_range[1] + 1))
    occs = list(range(occ_range[0], occ_range[1] + 1, 5))

    data = []
    for temp in temps:
        row = []
        for occ in occs:
            exog_input = pd.DataFrame({'Temperature': [temp], 'Occupancy': [occ]})
            forecast_result = results.get_prediction(start=results.nobs, end=results.nobs, exog=exog_input)
            pred = forecast_result.predicted_mean.iloc[0]
            row.append(pred)
        data.append(row)

    heatmap_df = pd.DataFrame(data, index=temps, columns=occs)

    st.subheader("🔥 Predicted Energy Consumption (kWh)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="YlOrRd")
    ax.set_xlabel("Occupancy")
    ax.set_ylabel("Temperature")
    ax.set_title("Predicted Energy Consumption (kWh)")
    st.pyplot(fig)