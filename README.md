# ⚡ Energy Demand Forecasting using LSTM and XGBoost

## Overview

This project focuses on forecasting future electricity demand using over **10 years of hourly energy consumption data from PJM Interconnection**, measured in Megawatts (MW). The objective is to capture temporal patterns in historical energy consumption and generate accurate forecasts for future demand.

Two machine learning approaches were implemented and evaluated:

* **XGBoost**
* **Long Short-Term Memory (LSTM)**

Based on evaluation metrics, the **LSTM model achieved superior performance** and was selected as the final model for multi-step forecasting.

---

## Features

* 📈 Hourly energy demand forecasting
* ⚡ Deep learning model using PyTorch LSTM
* 🌲 XGBoost model for comparison
* 📊 Actual vs Predicted visualization
* 📉 Interactive forecasting dashboard using Streamlit
* 📅 Forecast next 24 hours, one week, one month, or custom horizons
* 📋 Download prediction results as CSV
* 📈 Historical and forecast demand visualization
* 📊 Model performance comparison using MSE and RMSE

---

## Dataset

**Hourly Energy Consumption Dataset (PJM)**

* Over 10 years of hourly energy consumption data
* Unit: Megawatts (MW)
* Source: PJM Interconnection

---

## Technologies Used

### Machine Learning and Deep Learning

* PyTorch
* XGBoost
* Scikit-Learn

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Plotly

### Deployment

* Streamlit

---

## Model Architecture

### LSTM Model

* Input Size: 1
* Hidden Size: 64
* Number of Layers: 2
* Dropout: 0.2
* Sequence Length: 24 hours

The model performs recursive multi-step forecasting by using previous predictions as inputs for future predictions.

---

## Project Workflow

1. Data preprocessing
2. Feature scaling using MinMaxScaler
3. Sequence generation
4. Model training
5. Model evaluation
6. Recursive forecasting
7. Streamlit deployment

---

## Forecast Horizons

Users can generate forecasts for:

* Next 24 Hours
* Next Week
* Next Month
* Any Custom Number of Hours

---

## Model Comparison

Both models were evaluated on the testing dataset using:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

### Models Compared

* XGBoost
* LSTM

The LSTM model demonstrated better forecasting performance and was selected for future demand prediction.

---

## Streamlit Dashboard

The dashboard provides:

* Future Forecasting
* Interactive Graphs
* Prediction Tables
* Downloadable CSV Results
* Model Performance Comparison
* Actual vs Predicted Plots for LSTM and XGBoost

---

## Results

The LSTM model successfully captures temporal dependencies in electricity demand and produces accurate forecasts for future consumption.

---

## GitHub Repository

**GitHub:**
https://github.com/BharatTupe/Energy-Demand-Forecasting

---

## Streamlit Application

**Live Demo:**
https://energy-demand-forecasting-vcgufvtxgd8unsksf3bmwd.streamlit.app/

---

## Installation

Clone the repository:

```bash
git clone https://github.com/BharatTupe/Energy-Demand-Forecasting
```

Move into the project directory:

```bash
cd Energy-Demand-Forecasting
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Future Improvements

* SARIMA Model
* Prophet Model
* Confidence Intervals
* Hyperparameter Optimization
* Model Explainability
* Multi-step Direct Forecasting
* GPU Support
* Advanced Dashboard Analytics

---

## Author

**Bharat Tupe**

Machine Learning | Deep Learning | Time Series Forecasting | Data Science
