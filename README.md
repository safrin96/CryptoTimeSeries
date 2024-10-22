# 🚀 Cryptocurrency Price Prediction using Time-Series Analysis 📈

Welcome to the **Cryptocurrency Price Prediction Project**! This project dives deep into the world of Bitcoin by analyzing its daily price movements and forecasting future trends using various time-series models, including ARIMA and Prophet.

## 🔍 Project Overview

This project involves:
- Fetching real-time Bitcoin data from CoinGecko.
- Conducting thorough exploratory data analysis (EDA) with visualizations.
- Using time-series models (ARIMA, SARIMA, and Prophet) for forecasting.
- Evaluating model accuracy and creating ensemble forecasts.
- Exploring seasonality, trends, and volatility in Bitcoin's price movements.

## 📊 Key Insights & Visualizations

- **Trend & Seasonality Analysis**: Decomposed the time series to observe trends and seasonal patterns in Bitcoin's prices.
- **Volatility Analysis**: Measured daily percentage changes to gauge market fluctuations.
- **Rolling Statistics**: Used moving averages to smooth out short-term price fluctuations.
- **Forecasting**: Predicted future Bitcoin prices using ARIMA, SARIMA, and Prophet models, with confidence intervals.

## 🔧 How to Get Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/crypto-price-prediction.git
   cd crypto-price-prediction
2. **Install the Required Packages**:
   ```bash
   pip install -r requirements.txt
3. **Run the Analysis: Execute the main script to see the results**:
   ```bash
    python crypto_forecast.py
   ```
## 📂 Project Structure

- data/: Contains the datasets used for analysis (bitcoin_prices.csv, daily_bitcoin_prices.csv).
- plots/: Includes all generated plots, such as seasonal decompositions, moving averages, and forecast results.
- results/: Stores evaluation metrics like mean absolute error (MAE) for the models.
## ✨ Key Findings

- Trends and Seasonality: Bitcoin prices exhibit clear trends with some seasonal effects, especially around significant events like halving.
- Volatility: The price shows high volatility, with sudden spikes and drops over short periods.
- Model Comparison: The ARIMA model performed well for short-term predictions, while the Prophet model captured seasonality effectively.
- Ensemble Approach: Combining ARIMA and Prophet predictions improved overall forecasting accuracy.

## 🔮 Further Work

- Incorporate Exogenous Variables: Add external factors like trading volume or market sentiment.
- Hyperparameter Tuning: Improve the accuracy of the models by tuning parameters.
- More Cryptocurrencies: Extend the analysis to other coins for a diversified portfolio forecast.

## 📜 License

This project is open-source and available under the MIT License.

## 💡 Contributing

Feel free to open issues or submit pull requests to improve the project!
📬 Contact

For questions or collaborations, please contact [me](mailto:sshrabony@gmail.com).

Happy Forecasting! 🚀

