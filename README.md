# FinalProject
Overview 

This project presents a bike rental forecasting app, focusing on profit and rental count predictions using machine learning and time series forecasting methods. It's built using Streamlit, Pandas, Matplotlib, Scikit-Learn, and Statsmodels. 

Features 
Data Processing: Extracts time-related features from a datetime column, calculates profit and total rentals. 
Interactive UI: Streamlit-based interface with options to filter data based on seasons, weather conditions, day types, and hours. 
Forecasting Models: Includes Random Forest for profit prediction and ARIMA, as well as Triple Exponential Smoothing (Holt-Winters' Method) for forecasting rental counts or profits. 
Data Visualization: Plots historical data and forecasts for easy comparison and analysis.

Usage 
Run the Streamlit app: streamlit run app.py. 
Use the sidebar to filter data based on seasons, weather, day type, and hour range. 
Select the forecast type: Count or Profit. 
View the filtered data and forecasts.

Data Preparation 
The dataset is loaded and processed to extract useful features like hour, day of the week, month, and year. 
Custom functions are used to calculate profit and total rental count. 
The app filters data based on user-selected parameters. 

Forecasting Models 
Random Forest Regressor: Used for profit forecasting. It considers features like weather, holiday, working day, and rental counts. 
ARIMA: Used for time series forecasting of count or profit. The model is configurable based on the chosen forecast type. 
Triple Exponential Smoothing: Another method for forecasting, focusing on capturing seasonality.

Data Insights 
The app provides insights into bike rental patterns during different seasons, weather conditions, and types of days (holidays, working days). It helps in understanding the impact of various factors on bike rentals and profits.

Codelab:
https://codelabs-preview.appspot.com/?file_id=1Maj53V9jiP4I5KIsQB29ZDQB7DI_7uGiTw4sBeNf5iM#0
