
# Solar Panel Performance Monitoring & Prediction

This project aims to visualize and predict the power generation of solar panels using time series models. It includes interactive visualizations and model forecasts using ARIMA and SARIMA.

## Features

- **Data Visualization**: Visualize historical power generation data and the trend over time.
- **Modeling**: Predict future power generation using ARIMA or SARIMA models based on user inputs.
- **Decomposition**: Decompose the time series data into trend, seasonality, and residuals using seasonal decomposition.
- **Stationarity Check**: Perform the Dickey-Fuller test to check the stationarity of the time series data.

## Installation

To run the project, follow the steps below:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/solar-panel-performance.git
    cd solar-panel-performance
    ```

2. Install dependencies using `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

3. Make sure you have the `solar_data.csv` file in the project directory.

## Running the Application

After setting up the environment, run the Streamlit app:

```bash
streamlit run solar_app.py
```

This will launch a local server and open the application in your default web browser.

## User Input

- **Select Model**: Choose between ARIMA and SARIMA.
- **Model Parameters**: Adjust parameters such as p, d, q for ARIMA and p, d, q, s for SARIMA.
- **Forecast Period**: Select the number of hours for future power predictions.

## Files

- `solar_app.py`: Main Streamlit app script for interactive data visualization and forecasting.
- `solar_data.csv`: Historical solar panel performance data (must be in the same directory as the script).

## Dependencies

- `pandas`, `numpy`, `matplotlib`, `seaborn`, `streamlit`: Required for data manipulation, visualization, and app interface.
- `statsmodels`, `pmdarima`: For ARIMA and SARIMA models and statistical tests.
- `scikit-learn`, `scipy`: Additional libraries for model performance and scientific computing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
