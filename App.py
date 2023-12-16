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
