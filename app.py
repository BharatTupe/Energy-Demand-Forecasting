# Imports
import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib

from collections import deque
import plotly.graph_objects as go

from model import LSTMModel

#Load Dataset
df = pd.read_csv(
    "AEP_hourly.csv",
    parse_dates=['Datetime'],
    index_col='Datetime'
)

df = df[~df.index.duplicated(keep='first')]
df = df.sort_index()

# Load Scaler
scaler = joblib.load("scaler.pkl")

# Load LSTM model
model = LSTMModel()

model.load_state_dict(
    torch.load(
        "lstm_model.pth",
        map_location=torch.device('cpu')
    )
)

model.eval()


# Forecast function
def forecast_future(last_values,n_hours):

    lags = deque(last_values,maxlen=24)

    predictions = []

    model.eval()

    for i in range(n_hours):

        x_original = np.array(lags).reshape(-1,1)

        x_scaled = scaler.transform(x_original)

        x_scaled = x_scaled.reshape(1,24,1)

        x_scaled = torch.FloatTensor(x_scaled)

        with torch.no_grad():

            pred_scaled = model(x_scaled).numpy()

        pred_original = scaler.inverse_transform(pred_scaled)

        pred_value = pred_original[0,0]

        predictions.append(pred_value)

        lags.append(pred_value)

    return predictions


# Project Title
st.title("⚡ Energy Demand Forecasting")

# Subtitle
st.subheader("Hourly Energy Consumption Analysis and Forecasting")

# Description
st.markdown("""
This project focuses on forecasting future electricity demand using **over 10 years of hourly energy consumption data from PJM**, measured in **Megawatts (MW)**.

The application uses a **Long Short-Term Memory (LSTM)** neural network built with PyTorch to capture temporal patterns and generate forecasts for customizable horizons.

### Features

- 📈 Forecast electricity demand for the next 24 hours, one week, one month, or any custom number of hours.
- 📊 Interactive visualization of historical and predicted demand.
- 📋 Prediction tables with downloadable CSV files.
- ⚡ Recursive multi-step forecasting using LSTM.
- 🔍 Analysis of long-term consumption trends.
""")


st.success("""
⚡ **Forecast Information**

The historical dataset contains hourly energy consumption records up to **2018-08-02 23:00:00**. Using the trained LSTM model, future electricity demand is forecasted beginning from **2018-08-03 00:00:00** onward.
""")
st.write(
    "Forecast future electricity demand using LSTM."
)

# Forecast Horizon
option = st.selectbox(
    "Forecast Horizon",
    (
        "Next 24 Hours",
        "Next Week",
        "Next Month",
        "Custom"
    )
)

# Number of hours we want to predict
if option=="Next 24 Hours":

    hours = 24

elif option=="Next Week":

    hours = 168

elif option=="Next Month":

    hours = 720

else:

    hours = st.number_input(
        "Enter hours",
        min_value=1,
        max_value=2000,
        value=100
    )


# Prediction Button
if st.button("Forecast"):
    preds = forecast_future(
        df['AEP_MW'].values[-24:],
        hours
    )

    future_dates = pd.date_range(
        start=df.index[-1]+pd.Timedelta(hours=1),
        periods=hours,
        freq='h'
    )

    forecast_df = pd.DataFrame(
        {
            "Datetime":future_dates,
            "Predicted_AEP_MW":preds
        }
    )

    st.subheader("Predictions")
    st.dataframe(forecast_df)

    csv = forecast_df.to_csv(index=False)

    st.download_button(
        "Download Forecast CSV",
        csv,
        file_name="forecast.csv",
        mime="text/csv"
    )

    fig = go.Figure()

    history = df.tail(24*7)

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history['AEP_MW'],
            mode='lines',
            name='Historical'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=preds,
            mode='lines',
            name='Forecast'
        )
    )

    fig.update_layout(

        title="Energy Demand Forecast",

        xaxis_title="Datetime",

        yaxis_title="MW",

        hovermode='x unified',

        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


st.header("📊 Model Performance Comparison")
from PIL import Image

lstm_img = Image.open("LSTM_img.png")
xgb_img = Image.open("Xgboost_img.png")

col1, col2 = st.columns(2)

with col1:
    st.subheader("LSTM")
    st.image(
        lstm_img,
        caption="Actual vs Predicted (LSTM)",
        use_container_width=True
    )

with col2:
    st.subheader("XGBoost")
    st.image(
        xgb_img,
        caption="Actual vs Predicted (XGBoost)",
        use_container_width=True
    )


st.header("📈 Evaluation Metrics")
lstm_mse = 42952.84
lstm_rmse = 207.25

xgb_mse = 79798.74
xgb_rmse = 282.486             


col1,col2 = st.columns(2)

with col1:

    st.subheader("LSTM")

    st.metric(
        "MSE",
        f"{lstm_mse:.2f}"
    )

    st.metric(
        "RMSE",
        f"{lstm_rmse:.2f}"
    )

with col2:

    st.subheader("XGBoost")

    st.metric(
        "MSE",
        f"{xgb_mse:.2f}"
    )

    st.metric(
        "RMSE",
        f"{xgb_rmse:.2f}"
    )

import pandas as pd

metrics_df = pd.DataFrame(
    {
        "Model": ["LSTM","XGBoost"],
        "MSE":[lstm_mse,xgb_mse],
        "RMSE":[lstm_rmse,xgb_rmse]
    }
)

st.subheader("Performance Summary")

st.dataframe(
    metrics_df,
    use_container_width=True
)

st.success(
"""
🏆 **LSTM achieved the best forecasting performance** with lower MSE and RMSE values compared to XGBoost.

Therefore, the LSTM model has been selected as the final model for generating future energy demand forecasts.
"""
)