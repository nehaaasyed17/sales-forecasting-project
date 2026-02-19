import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# App title
st.title("AI-Based Sales Forecasting App")

# Load dataset
data = pd.read_csv("Walmart.xlsx")

# Convert Date column to datetime (DD-MM-YYYY format)
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Sort data by date
data = data.sort_values('Date')

# Dataset preview
st.subheader("Dataset Preview")
st.write(data.head())

# Sales Trend Graph
st.subheader("Weekly Sales Trend")
plt.figure(figsize=(10, 4))
plt.plot(data['Date'], data['Weekly_Sales'])
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.title("Weekly Sales Over Time")
st.pyplot(plt)

# Prepare data for ML model
data['Day'] = np.arange(len(data))
X = data[['Day']]
y = data['Weekly_Sales']

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Forecast section
st.subheader("AI Sales Forecast")
weeks = st.slider("Select number of weeks to forecast", 1, 52, 12)

future_days = np.arange(len(data), len(data) + weeks).reshape(-1, 1)
forecast = model.predict(future_days)

# Generate future dates (after last actual date)
future_dates = pd.date_range(
    start=data['Date'].iloc[-1] + pd.Timedelta(weeks=1),
    periods=weeks,
    freq='W'
)

# Plot forecast
plt.figure(figsize=(10, 4))
plt.plot(data['Date'], y, label="Actual Sales")
plt.plot(future_dates, forecast, label="Forecasted Sales")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.title("Sales Forecast")
plt.legend()
st.pyplot(plt)

# Model performance
st.subheader("Model Performance")
r2 = model.score(X, y)

st.write(f"R² Score: {r2:.2f}")
