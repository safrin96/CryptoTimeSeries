import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import zscore
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import seaborn as sns

# Create directories for organizing output
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

def fetch_crypto_data(crypto_id='bitcoin', vs_currency='usd', days=365):
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart'
    params = {'vs_currency': vs_currency, 'days': days}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Fetch Bitcoin data for the last year
crypto_df = fetch_crypto_data()
crypto_df.to_csv('data/bitcoin_prices.csv', index=False)

# Load the data
crypto_df = pd.read_csv('data/bitcoin_prices.csv')
crypto_df.set_index('timestamp', inplace=True)
crypto_df.index = pd.to_datetime(crypto_df.index)

# Resample data to daily frequency
daily_data = crypto_df['price'].resample('D').mean().fillna(method='ffill')

# Save resampled data
daily_data.to_csv('data/daily_bitcoin_prices.csv')

# Plot and save original data
plt.figure(figsize=(12, 6))
plt.plot(daily_data, label='Daily Prices')
plt.title('Cryptocurrency Daily Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.savefig('plots/daily_prices.png')
plt.close()

# Seasonal Decomposition
decomposition = seasonal_decompose(daily_data, model='multiplicative', period=30)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(daily_data, label='Original', color='blue')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend', color='red')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality', color='green')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residuals', color='gray')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('plots/seasonal_decomposition.png')
plt.close()

# Moving Averages
daily_data = pd.DataFrame(daily_data)
daily_data['7-day MA'] = daily_data['price'].rolling(window=7).mean()
daily_data['30-day MA'] = daily_data['price'].rolling(window=30).mean()

plt.figure(figsize=(12, 6))
plt.plot(daily_data['price'], label='Daily Prices', color='blue')
plt.plot(daily_data['7-day MA'], label='7-Day Moving Average', color='orange')
plt.plot(daily_data['30-day MA'], label='30-Day Moving Average', color='red')
plt.title('Daily Prices with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.savefig('plots/moving_averages.png')
plt.close()

# ARIMA Model
train_size = int(len(daily_data) * 0.8)
train, test = daily_data['price'][:train_size], daily_data['price'][train_size:]
arima_model = ARIMA(train, order=(5, 1, 0))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=len(test))

plt.figure(figsize=(12, 6))
plt.plot(daily_data['price'], label='Actual Prices')
plt.plot(test.index, arima_forecast, color='red', label='ARIMA Forecast')
plt.legend()
plt.title('ARIMA Model Forecast vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.savefig('plots/arima_forecast.png')
plt.close()

# Save ARIMA results
arima_mae = mean_absolute_error(test, arima_forecast)
with open('results/arima_mae.txt', 'w') as f:
    f.write(f'ARIMA MAE: {arima_mae}\n')

# Prophet Model
prophet_df = daily_data.reset_index().rename(columns={'timestamp': 'ds', 'price': 'y'})
prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(prophet_df.iloc[:train_size])
future = prophet_model.make_future_dataframe(periods=len(test))
prophet_forecast = prophet_model.predict(future)
prophet_model.plot(prophet_forecast)
plt.title('Prophet Forecast')
plt.savefig('plots/prophet_forecast.png')
plt.close()

# Evaluate Prophet
prophet_test = prophet_forecast.iloc[train_size:]['yhat'].values
prophet_mae = mean_absolute_error(test, prophet_test)
with open('results/prophet_mae.txt', 'w') as f:
    f.write(f'Prophet MAE: {prophet_mae}\n')

# Save ensemble forecast
ensemble_forecast = (arima_forecast.values + prophet_test) / 2
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual Prices', color='blue')
plt.plot(test.index, ensemble_forecast, label='Ensemble Forecast', color='purple', linestyle='--')
plt.legend()
plt.title('Ensemble Forecast')
plt.savefig('plots/ensemble_forecast.png')
plt.close()