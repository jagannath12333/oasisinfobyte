import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

# Load datasets
unemployment_data = pd.read_csv('Unemployment_in_India.csv')
unemployment_rate_data = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')

# Clean and preprocess data
unemployment_data['Date'] = pd.to_datetime(unemployment_data['Date'], errors='coerce')
unemployment_rate_data['Date'] = pd.to_datetime(unemployment_rate_data['Date'], errors='coerce')
unemployment_data = unemployment_data.dropna(subset=['Date'])
unemployment_rate_data = unemployment_rate_data.dropna(subset=['Date'])

# Data Analysis
average_unemployment_rate = unemployment_rate_data.groupby('Date').mean()

# Data Visualization
plt.figure(figsize=(14, 7))
sns.lineplot(data=average_unemployment_rate)
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.show()

# Forecasting with ARIMA
model = ARIMA(average_unemployment_rate, order=(5, 1, 0))
model_fit = model.fit(disp=0)
forecast = model_fit.forecast(steps=12)

# Plot the forecast
plt.figure(figsize=(14, 7))
plt.plot(average_unemployment_rate, label='Actual')
plt.plot(forecast, label='Forecast', color='red')
plt.title('Unemployment Rate Forecast')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.legend()
plt.show()
