# S\&P 500 Price Forecasting with Hybrid LSTM-Transformer

## Project Overview

This project implements a multi-stage forecasting approach, starting with established **Base Models** (ARIMA, SARIMA, Prophet) for benchmarking, before moving to a highly advanced **Hybrid LSTM-Transformer model** to forecast daily S\&P 500 index closing prices for a 30-day horizon. The hybrid model combines Long Short-Term Memory (LSTM) networks for capturing short-term sequential dependencies with the Transformer's attention mechanism for modeling long-term market trends.

An additional feature is the **SHAP interpretability** to provide transparent insights into the predictions. The output is delivered through an interactive Streamlit dashboard.

-----

## Problem statement

▸
Predicting the next day's closing price of a major financial index like the S&P 500 (^GSPC) is one of the most challenging tasks in time series analysis due to market noise, non-linearity, volatility, and the highly non-stationary nature of price data.
▸
Traditional methods (e.g., statistical models) often fail to capture complex patterns in financial markets.
▸
Goal: Build a robust deep learning model to predict the next day's closing price of the S&P 500, outperforming benchmarks.
▸
Why it matters: Accurate predictions can inform trading strategies, risk management, and investment decisions in volatile markets.

-----

##  Key Objectives

  * **Benchmarking (Base Models):** Establish performance baselines using classical time series models (ARIMA, SARIMA) and a modern additive model (Prophet).
  * **Advanced Forecasting:** Predict daily S\&P 500 closing prices using a Hybrid LSTM-Transformer, incorporating features like returns, volatility, and lagged prices.
  * **Risk Analysis:** Detect anomalous price movements (e.g., market crashes).
  * **Interpretability:** Analyze feature importance using **SHAP values** for transparent predictions.
  * **Deployment:** Deliver an **interactive Streamlit dashboard** for visualizing forecasts, uncertainty, and anomaly flags.

-----

## Data Source & Preprocessing

### Data Source

| Parameter | Detail |
| :--- | :--- |
| **Provider** | Yahoo Finance via the **`yfinance`** Python library. |
| **Ticker** | S\&P 500 index . |
| **Time Period** | January 1, 2010, to September 30, 2025. |
| **Attributes** | Date, Open, High, Low, Close, **Adjusted Close**, Volume. |
| **Access** | Data is dynamically fetched and saved locally to `data/sp500_daily_2010_2025.csv`. |

### Preprocessing Steps

1.  **Cleaning:** Missing values are to be removed or filled.
2.  **Feature Engineering:** Includes **Daily returns**, **20-day rolling volatility**, and **Lagged close**.
3.  **Transformation:** Data is checked for **stationarity** using the Augmented Dickey-Fuller (ADF) test, a key requirement for ARIMA/SARIMA models.
4.  **Normalization:** A **Scaler** is applied to key features for neural network model input (LSTM/Transformer).

-----

## Machine Learning Methodology

The project follows a rigorous, multi-level modeling approach:

### 1\. Base Models (Benchmarking)

| Model | Technique | Role |
| :--- | :--- | :--- |
| **ARIMA** | AutoRegressive Integrated Moving Average | Models simple time series dependencies and provides the fundamental performance baseline. |
| **SARIMA** | Seasonal ARIMA | Extends ARIMA to capture seasonality and cyclical patterns inherent in financial data. |
| **Prophet** | Additive Regression Model (Meta) | Models time series based on trend, seasonality, and holidays; serves as a strong modern benchmark for comparison. |

### 2\. Advanced Models

| Component | Technique | Role |
| :--- | :--- | :--- |
| **Core Model** | **Hybrid LSTM-Transformer** | **LSTM** captures short-term momentum; **Transformer's** multi-head attention models long-term market cycles and dependencies. |
| **Robustness** | Ensemble Stacking | Combines the LSTM-Transformer with **XGBoost** to enhance prediction stability and non-linear feature handling. |
| **Interpretability** | SHAP (SHapley Additive exPlanations) | Calculates feature contribution to each prediction, identifying the key drivers (e.g., volatility vs. lagged price). |
| **Optimization** | Optuna | Used for Bayesian hyperparameter tuning across all complex models. |

-----

## Project Structure

The structure now accounts for the implementation and notebooks of the Base Models.

```
sp500_forecasting/
├── data/
│   └── sp500_daily_2010_2025.csv  # Raw and preprocessed S&P 500 data
├── models/
│   ├── base_models.py             # ARIMA, SARIMA, and Prophet implementations
│   ├── lstm_transformer.py         # Hybrid model implementation
│   └── bayesian_nn.py             # Bayesian uncertainty modeling
├── notebooks/
│   ├── 01_data_preprocessing.ipynb # Data fetching and cleaning
│   ├── 02_base_model_benchmarking.ipynb # ARIMA, SARIMA, Prophet training and results
│   ├── 03_hybrid_model_training.ipynb    # LSTM-Transformer training and evaluation
│   ├── 04_anomaly_detection.ipynb # Anomaly detection analysis
│   └── 05_interpretability.ipynb  # SHAP analysis
├── scripts/
│   ├── fetch_data.py              # yfinance data retrieval
│   └── dashboard.py               # Streamlit dashboard
├── README.md                      # This file
└── requirements.txt               # Dependencies list
```

-----

## 🧪 Evaluation Metrics & Expected Results

### Evaluation Metrics

| Metric | Target | Description |
| :--- | :--- | :--- |
| **RMSE/MAE** | RMSE \< 50 points (on normalized prices) | Standard measure of forecasting error. |
| **Directional Accuracy** | \>60% | Percentage of correct up/down predictions. |
| **Coverage Probability** | \>90% | The percentage of true prices contained within the 95% confidence intervals. |

### Expected Results

  * **Forecasts:** Accurate 30-day price predictions with robust confidence intervals, demonstrating the **Hybrid LSTM-Transformer's superiority** over the Base Models.
  * **Insights:** Clear identification of **volatility** as a high-impact feature using SHAP, directly informing risk-averse investment strategies.
  * **Anomalies:** Successful flagging of major market events.

-----

## Future Extensions

  * **Multimodal Inputs:** Incorporate **news sentiment** data (via NewsAPI) for sentiment-driven forecasting.
  * **Real-Time Integration:** Implement scheduled daily `yfinance` pulls using **cron jobs** for true live prediction updates.



