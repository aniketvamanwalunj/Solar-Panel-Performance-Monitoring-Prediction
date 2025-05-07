import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("solar_data.csv")
    df["date"] = pd.date_range(start="2024-01-01", periods=len(df), freq='D')
    
    # Filter only required date range (Jan 2025 to Apr 2025)
    df = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2025-04-30")]
    
    # Remove 0 or negative values
    df = df[df["power-generated"] > 0]
    
    df.set_index("date", inplace=True)
    return df

df = load_data()

# Streamlit Title
st.title("🔆 Solar Panel Performance Monitoring & Prediction")

# Sidebar
st.sidebar.title("Model Parameters")
model_option = st.sidebar.selectbox("Select Model", ["ARIMA", "SARIMA"])

if model_option == "ARIMA":
    st.sidebar.subheader("ARIMA Parameters")
    p = st.sidebar.slider("p (AR)", 0, 5, 2)
    d = st.sidebar.slider("d (Diff)", 0, 2, 1)
    q = st.sidebar.slider("q (MA)", 0, 5, 1)
    st.sidebar.write(f"Selected ARIMA Order: ({p},{d},{q})")
elif model_option == "SARIMA":
    st.sidebar.subheader("SARIMA Parameters")
    p = st.sidebar.slider("p (AR)", 0, 5, 2)
    d = st.sidebar.slider("d (Diff)", 0, 2, 1)
    q = st.sidebar.slider("q (MA)", 0, 5, 1)
    s = st.sidebar.slider("Seasonal Period (S)", 7, 30, 7)
    st.sidebar.write(f"Selected SARIMA Order: ({p},{d},{q},{s})")

# User Input for Forecast Duration (Days)
forecast_days = st.sidebar.slider("Select Forecast Duration (Days)", 1, 30, 30)

# Data Preview
if st.checkbox("Show Filtered Data (Jan-Apr 2025)"):
    st.write(df.head())

# Seasonal Decomposition
try:
    df['power-generated'] = df['power-generated'].apply(lambda x: x if x > 0 else 0.001)
    decomposition = seasonal_decompose(df['power-generated'], model='multiplicative', period=7)
except:
    decomposition = seasonal_decompose(df['power-generated'], model='additive', period=7)

# Plot Historical Data
st.subheader("📊 Historical Data (Jan-Apr 2025)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["power-generated"], label="Actual Data (Jan-Apr 2025)", color='blue', linestyle='-', linewidth=2)
ax.set_xlabel("Date")
ax.set_ylabel("Power Generated (W)")
ax.set_title("Historical Power Generation (Jan-Apr 2025)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Forecasting: User-Defined Forecast Period
forecast_start = datetime(2025, 5, 1)
forecast_end = forecast_start + timedelta(days=forecast_days - 1)
n_periods = (forecast_end - forecast_start).days + 1

if model_option == "ARIMA":
    st.subheader("🔧 Training ARIMA Model")
    model = ARIMA(df["power-generated"], order=(p, d, q))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=n_periods)
    forecast = forecast[forecast > 0]  # Remove zeros and negatives

elif model_option == "SARIMA":
    st.subheader("🔧 Training SARIMA Model")
    model = SARIMAX(df["power-generated"], order=(p, d, q), seasonal_order=(p, d, q, s))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=n_periods)
    forecast = forecast[forecast > 0]  # Remove zeros and negatives

# Prepare forecast dataframe
forecast_dates = pd.date_range(start=forecast_start, periods=len(forecast))
forecast_df = pd.DataFrame({
    "date": forecast_dates,
    "predicted_power": forecast.values
})

# Show Forecast
st.subheader(f"📈 Forecast Using {model_option} for {forecast_days} Days")
st.write(forecast_df)

# Plot Forecast with Past Data
st.subheader(f"📊 Forecast Plot: Using {model_option}")

# Plot historical data and forecasted data
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the actual data (past data)
ax.plot(df.index, df["power-generated"], label="Actual Data (Jan-Apr 2025)", color='blue', linestyle='-', linewidth=2)

# Plot the forecasted data (May 2025)
ax.plot(forecast_df["date"], forecast_df["predicted_power"], color='red', marker='o', label=f'Forecasted Data ({forecast_days} Days)')

ax.set_xlabel("Date")
ax.set_ylabel("Power Generated (W)")
ax.set_title(f"Forecasted vs Actual Power Generation ({model_option})")
ax.legend()
ax.grid(True)

# Display the plot
st.pyplot(fig)
