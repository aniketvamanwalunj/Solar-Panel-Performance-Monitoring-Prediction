import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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
model_option = st.sidebar.selectbox("Select Model", ["ARIMA", "SARIMA", "XGBoost", "LightGBM", "Random Forest Regressor"])

# Show n_lags slider only for ML models
if model_option in ["XGBoost", "LightGBM", "Random Forest Regressor"]:
    n_lags = st.sidebar.slider("Select Number of Lags", 1, 30, 7)  # Default to 7 lags
else:
    n_lags = None  # Not used for ARIMA/SARIMA

if model_option in ["ARIMA", "SARIMA"]:
    st.sidebar.subheader(f"{model_option} Parameters")
    p = st.sidebar.slider("p (AR)", 0, 5, 2)
    d = st.sidebar.slider("d (Diff)", 0, 2, 1)
    q = st.sidebar.slider("q (MA)", 0, 5, 1)
    if model_option == "SARIMA":
        s = st.sidebar.slider("Seasonal Period (S)", 7, 30, 7)

forecast_days = st.sidebar.slider("Select Forecast Duration (Days)", 1, 30, 30)

# Show Data & Stats
if st.checkbox("Show Filtered Data (Jan-Apr 2025)"):
    st.write(df.head())
    st.write("\n**Summary Statistics:**")
    st.write(df.describe())

# Weekly Trends
if st.checkbox("Show Weekly Trend"):
    st.line_chart(df['power-generated'].resample('W').mean(), use_container_width=True)

# Decomposition
try:
    df['power-generated'] = df['power-generated'].apply(lambda x: x if x > 0 else 0.001)
    decomposition = seasonal_decompose(df['power-generated'], model='multiplicative', period=7)
except:
    decomposition = seasonal_decompose(df['power-generated'], model='additive', period=7)

# Interactive Historical Plot
st.subheader("📊 Historical Power Generation")
fig = px.line(df, y="power-generated", title="Daily Power Generation (Jan-Apr 2025)", labels={"power-generated": "Power Generated (W)", "date": "Date"})
fig.update_traces(line_color='blue')
fig.update_layout(xaxis_title="Date", yaxis_title="Power Generated (W)", title_x=0.5)
st.plotly_chart(fig)

# ======================
# Forecasting Future Data
# ======================
forecast_start = datetime(2025, 5, 1)
forecast_end = forecast_start + timedelta(days=forecast_days - 1)
n_periods = (forecast_end - forecast_start).days + 1

def create_supervised(data, n_lags=7):
    df_supervised = pd.DataFrame()
    for i in range(n_lags, 0, -1):
        df_supervised[f'lag_{i}'] = data.shift(i)
    df_supervised['target'] = data.values
    df_supervised.dropna(inplace=True)
    return df_supervised

if model_option == "ARIMA":
    st.subheader("🔧 Training ARIMA Model")
    model = ARIMA(df["power-generated"], order=(p, d, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=n_periods)
    forecast = forecast[forecast > 0]

elif model_option == "SARIMA":
    st.subheader("🔧 Training SARIMA Model")
    model = SARIMAX(df["power-generated"], order=(p, d, q), seasonal_order=(p, d, q, s))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=n_periods)
    forecast = forecast[forecast > 0]

elif model_option in ["XGBoost", "LightGBM", "Random Forest Regressor"]:
    st.subheader(f"🔧 Training {model_option} Model")
    full_supervised = create_supervised(df['power-generated'], n_lags)
    X_full = full_supervised.drop('target', axis=1)
    y_full = full_supervised['target']

    if model_option == "XGBoost":
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    elif model_option == "LightGBM":
        model = LGBMRegressor(n_estimators=100)
    elif model_option == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=100)

    model.fit(X_full, y_full)

    last_known = df['power-generated'].values[-n_lags:]
    forecast = []
    for _ in range(n_periods):
        input_data = pd.DataFrame([last_known], columns=[f'lag_{i}' for i in range(n_lags, 0, -1)])
        pred = model.predict(input_data)[0]
        forecast.append(pred)
        last_known = np.append(last_known[1:], pred)

forecast_dates = pd.date_range(start=forecast_start, periods=len(forecast))
forecast_df = pd.DataFrame({"date": forecast_dates, "predicted_power": forecast})

# Show Forecast Table
st.subheader(f"📈 Forecast Using {model_option} for {forecast_days} Days")
st.write(forecast_df)

# Download Button
st.download_button("📥 Download Forecast Data", forecast_df.to_csv(index=False), "forecast_data.csv")

# Plot Forecast with Plotly
st.subheader(f"📊 Forecast vs Historical Power Generation")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["power-generated"], mode='lines', name='Actual (Jan-Apr)', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['predicted_power'], mode='lines+markers', name='Forecast', line=dict(color='red')))
fig.update_layout(title=f"{model_option} Forecast", xaxis_title="Date", yaxis_title="Power Generated (W)", title_x=0.5)
st.plotly_chart(fig)
