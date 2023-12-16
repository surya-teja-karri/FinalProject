import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Load the dataset
df = pd.read_csv('/Users/tanayparikh/Downloads/Train.csv')

# Extract time-related features from the datetime column
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

# Function to calculate profit
def calculate_profit(x):
    casual_customers = x['casual']
    registered_customers = x['registered']
    casual_price_per_day = 20
    registered_price_per_day = 5
    taxes_percent = 0.14
    maintenance_per_hour = 1500 / (365 * 24)

    profit_cash = casual_customers * casual_price_per_day + registered_price_per_day * registered_customers
    profit_with_taxes = profit_cash - (profit_cash * taxes_percent)
    total_profit = profit_with_taxes - maintenance_per_hour

    return total_profit

# Function to calculate count (total rentals)
def calculate_count(x):
    return x['casual'] + x['registered']

# Add profit and count columns to the DataFrame
df['Profit'] = df[['casual', 'registered']].apply(calculate_profit, axis=1)
df['Count'] = df[['casual', 'registered']].apply(calculate_count, axis=1)

# Streamlit app
st.title("Bike Rental Forecasting with Different Profit Model")

# Sidebar with user inputs
st.sidebar.header("Choose Parameters")
season_options = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
selected_seasons = st.sidebar.multiselect("Seasons", list(season_options.keys()), default=[1, 2, 3, 4], format_func=lambda x: season_options[x])
#weather_options = {1: "Clear", 2: "Mist", 3: "Snow"}
weather_options = {1: "snowy", 2: "Mist", 3: "Clear"}
weather = st.sidebar.selectbox("Weather", list(weather_options.keys()), format_func=lambda x: weather_options[x])
forecast_type = st.sidebar.selectbox("Choose Forecast Type", ["Count", "Profit"])

# Dropdowns for additional parameters
day_type = st.sidebar.multiselect("Day Type", ["Holiday", "Working Day"], default=["Holiday", "Working Day"])

# Extract time-related features from the datetime column for filtering
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

# Filter data based on user inputs
filtered_data = df[
    (df['season'].isin(selected_seasons)) & 
    (df['weather'] == weather) & 
    (((df['holiday'] == 1) & ("Holiday" in day_type)) | ((df['workingday'] == 1) & ("Working Day" in day_type)))
]

# Hour selection in sidebar using a slider
hour_range = st.sidebar.slider("Select Hour Range", min_value=0, max_value=23, value=(0, 23))
filtered_data = filtered_data[(filtered_data['hour'] >= hour_range[0]) & (filtered_data['hour'] <= hour_range[1])]
# Check if 'datetime' is in the columns before using it
if 'datetime' in filtered_data.columns:
    filtered_data['datetime'] = pd.to_datetime(filtered_data['datetime'])
    filtered_data.set_index('datetime', inplace=True)

# Display the filtered data (showing only 10 rows)
st.subheader("Filtered Data (First 10 Rows)")
st.write(filtered_data.head(10))

# Profit forecasting using Random Forest
X_rf = filtered_data[['weather','holiday', 'workingday', 'casual', 'registered', 'temp', 'season', 'atemp', 'windspeed', 'humidity', 'hour', 'day_of_week', 'month', 'year']]
y_rf = filtered_data['Profit']

# Split the data into training and testing sets
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

# Train the Random Forest model for profit forecasting
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

# Predictions
profit_predictions_rf = rf_model.predict(X_test_rf)

# Calculate mean squared error as a measure of performance
mse_rf = mean_squared_error(y_test_rf, profit_predictions_rf)

# Show the performance metrics for Random Forest
st.subheader("Performance Metrics (Random Forest)")
st.write(f"Mean Squared Error: {mse_rf:.2f}")


# Forecasting using ARIMA
# Example: Including more columns for ARIMA forecasting
arima_data = filtered_data[['weather', 'holiday','workingday', 'windspeed', 'casual', 'registered', 'temp', 'season', 'atemp', 'humidity', 'hour', 'day_of_week', 'month', 'year', forecast_type]]
if 'datetime' in arima_data.columns:
    arima_data['datetime'] = pd.to_datetime(arima_data['datetime'])
    arima_data.set_index('datetime', inplace=True)

X_arima = arima_data.drop(columns=[forecast_type])
y_arima = arima_data[forecast_type]

# Split the data into training and testing sets
X_train_arima, X_test_arima, y_train_arima, y_test_arima = train_test_split(X_arima, y_arima, test_size=0.2, random_state=42)

# Train the ARIMA model for forecasting
arima_model = ARIMA(y_train_arima, order=(2, 2, 2))
arima_results = arima_model.fit()

# Predictions
future_steps = 300
forecast_index = pd.date_range(arima_data.index[-1], periods=future_steps + 1, freq='D')[1:]
forecast_df = pd.DataFrame({'Forecast': arima_results.get_forecast(steps=future_steps).predicted_mean.values}, index=forecast_index)

# Calculate the mean value of the forecast
mean_value = forecast_df['Forecast'].mean()

