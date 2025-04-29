import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("solar_data.csv")  # Change this to your actual file name
    df["date"] = pd.date_range(start="2025-01-01", periods=len(df), freq='H')  # Start from 2025-01-01
    df.set_index("date", inplace=True)
    return df

df = load_data()

# Streamlit Dashboard
st.title("🔆 Solar Panel Performance Monitoring & Prediction")

# Sidebar for User Input
st.sidebar.title("Model Parameters")
model_option = st.sidebar.selectbox("Select Model", ["ARIMA", "SARIMA"])

# ARIMA/SARIMA Parameters Based on Model Selection
if model_option == "ARIMA":
    st.sidebar.subheader("ARIMA Model Parameters")
    p = st.sidebar.slider("p (AR order)", 0, 5, 2)
    d = st.sidebar.slider("d (Differencing order)", 0, 2, 1)
    q = st.sidebar.slider("q (MA order)", 0, 5, 1)
    
    # Dynamic Best ARIMA Order based on user input
    st.sidebar.write(f"Best ARIMA Order : ({p}, {d}, {q})")  # Showing user-defined parameters

elif model_option == "SARIMA":
    st.sidebar.subheader("SARIMA Model Parameters")
    p = st.sidebar.slider("p (AR order)", 0, 5, 2)
    d = st.sidebar.slider("d (Differencing order)", 0, 2, 1)
    q = st.sidebar.slider("q (MA order)", 0, 5, 1)
    s = st.sidebar.slider("S (Seasonal Periodicity)", 12, 24, 24)  # Adjust based on seasonality
    
    # Dynamic Best SARIMA Order based on user input
    st.sidebar.write(f"Best SARIMA Order : ({p}, {d}, {q}, {s})")  # Showing user-defined parameters

# Data Preview
if st.checkbox("Show Raw Data"):
    st.write(df.head())

# Check Stationarity (Dickey-Fuller Test)
from statsmodels.tsa.stattools import adfuller
def adf_test(series):
    result = adfuller(series)
    return result[1]  # p-value

st.subheader("📉 Checking Stationarity of Data")
p_value = adf_test(df["power-generated"])
if p_value < 0.05:
    st.write("✅ Data is Stationary (p-value:", p_value, ")")
else:
    st.write("❌ Data is NOT Stationary (p-value:", p_value, ") - Differencing Needed")
    df['power-generated'] = df['power-generated'].diff().dropna()

# Decomposing Power Generation into Trend, Seasonality, and Residuals
try:
    # Use additive decomposition if there are negative/zero values in the data
    df['power-generated'] = df['power-generated'].apply(lambda x: x if x > 0 else 0.001)  # Fix negative values for multiplicative
    decomposition = seasonal_decompose(df['power-generated'], model='multiplicative', period=24)  # Using 'multiplicative' for positive data
except ValueError:
    # Use additive decomposition if multiplicative fails due to negative/zero values
    decomposition = seasonal_decompose(df['power-generated'], model='additive', period=24)

# Define User-selected Model (ARIMA or SARIMA)
def calculate_accuracy(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)
    r2 = r2_score(actual, predicted)
    return mae, mse, rmse, r2

if model_option == "ARIMA":
    st.subheader("🔧 Using User-defined ARIMA Parameters")
    model = ARIMA(df['power-generated'].dropna(), order=(p, d, q))
    model_fit = model.fit()

    # Forecasting
    st.subheader("📈 Future Power Prediction")
    n_periods = st.slider("Select Forecast Period (hours)", 1, 48, 24)
    forecast = model_fit.forecast(steps=n_periods)

    # Generate Future Dates
    dates_future = [df.index[-1] + timedelta(hours=i) for i in range(1, n_periods+1)]
    forecast_df = pd.DataFrame({"date": dates_future, "predicted_power": forecast})
    st.write(forecast_df)

    # Model Accuracy Calculation (on training data for now)
    train_actual = df['power-generated'][-n_periods:].dropna()
    train_predicted = model_fit.fittedvalues[-n_periods:]

    mae, mse, rmse, r2 = calculate_accuracy(train_actual, train_predicted)
    st.subheader("📊 Model Accuracy (ARIMA)")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R-Squared (R2): {r2:.2f}")

    # Plot Forecast (Line Plot with Predictions)
    st.subheader("📈 Future Power Prediction")

    # Plot actual vs forecasted data
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(forecast_df["date"], forecast_df["predicted_power"], label='Forecasted Power', color='red', marker='x')

    # Labeling and adding a title
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Power Generated (W)")
    ax2.set_title("Forecasted Power Generation")
    ax2.legend()

    # Add grid
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    st.pyplot(fig2)

elif model_option == "SARIMA":
    st.subheader("🔧 Using User-defined SARIMA Parameters")
    seasonal_order = (p, d, q, s)
    sarima_model = SARIMAX(df['power-generated'].dropna(), order=(p, d, q), seasonal_order=seasonal_order)
    sarima_model_fit = sarima_model.fit()

    # Forecasting
    st.subheader("📈 Future Power Prediction (SARIMA)")
    n_periods = st.slider("Select Forecast Period (hours)", 1, 48, 24)
    forecast = sarima_model_fit.forecast(steps=n_periods)

    # Generate Future Dates
    dates_future = [df.index[-1] + timedelta(hours=i) for i in range(1, n_periods+1)]
    forecast_df = pd.DataFrame({"date": dates_future, "predicted_power": forecast})
    st.write(forecast_df)

    # Model Accuracy Calculation (on training data for now)
    train_actual = df['power-generated'][-n_periods:].dropna()
    train_predicted = sarima_model_fit.fittedvalues[-n_periods:]

    mae, mse, rmse, r2 = calculate_accuracy(train_actual, train_predicted)
    st.subheader("📊 Model Accuracy (SARIMA)")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R-Squared (R2): {r2:.2f}")

    # Plot Forecast (Line Plot with Predictions)
    st.subheader("📈 Future Power Prediction (SARIMA)")

    # Plot actual vs forecasted data
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(forecast_df["date"], forecast_df["predicted_power"], label='Forecasted Power', color='red', marker='x')

    # Labeling and adding a title
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Power Generated (W)")
    ax2.set_title("Forecasted Power Generation (SARIMA)")
    ax2.legend()

    # Add grid
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    st.pyplot(fig2)
