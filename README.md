# S\&P 500 Price Forecasting with Hybrid LSTM-Transformer

## Project Overview

This project implements a **hybrid LSTM-Transformer model** to forecast daily S\&P 500 index closing prices for a 30-day horizon. It combines Long Short-Term Memory (LSTM) networks for capturing short-term sequential dependencies with the Transformer's attention mechanism for modeling long-term market trends.

An additional feature is the **SHAP interpretability** to provide transparent insights into the predictions. The output is delivered through an interactive Streamlit dashboard.

-----

## Key Objectives

  * **Forecasting:** Predict daily S\&P 500 closing prices for a 30-day horizon, incorporating features like returns, volatility and lagged prices.
  * **Risk Analysis:** Detect anomalous price movements.
  * **Interpretability:** Analyze feature importance using **SHAP values** for transparent predictions.
  * **Deployment:** Deliver an **interactive Streamlit dashboard** for visualizing forecasts, uncertainty, and anomaly flags.

-----

## Data Source & Preprocessing

### Data Source

  * **Provider:** Yahoo Finance via the **`yfinance`** Python library.
  * **Ticker:** S\&P 500 index.
  * **Time Period:** January 1, 2010, to September 30, 2025.
  * **Attributes:** Date, Open, High, Low, Close, Adjusted Close, Volume.
  * **Access:** Data is dynamically fetched and saved locally.

<!-- end list -->


### Preprocessing Steps

1.  **Cleaning:** Missing values are to be removed or filled .
2.  **Feature Engineering:**
      * Daily returns.
      * 20-day rolling volatility.
      * Lagged close.
3.  **Transformation:** Data is checked for **stationarity** using the Augmented Dickey-Fuller test.
4.  **Normalization:** **A Scaler** is applied to key features for model input.

-----

## Machine Learning Methodology

| Component | Technique | Role |
| :--- | :--- | :--- |
| **Core Model** | ** LSTM-Transformer** | LSTM captures short-term momentum; Transformer's multi-head attention models long-term market cycles. |
| **Robustness** | **Ensemble Stacking** | Combines the LSTM-Transformer with XGBoost to enhance prediction stability. |
| **Interpretability** | **SHAP (SHapley Additive exPlanations)** | Calculates feature contribution to each prediction, identifying the key drivers. |
| **Optimization** | **Optuna** | Used for Bayesian hyperparameter tuning (e.g., finding optimal LSTM units and attention heads). |

-----

## Project Structure

```
sp500_forecasting/
├── data/
│   └── sp500_daily_2010_2025.csv  # Raw and preprocessed S&P 500 data
├── models/
│   ├── lstm_transformer.py         # Hybrid model implementation
│   └── bayesian_nn.py             # Bayesian uncertainty modeling
├── notebooks/
│   ├── 01_data_preprocessing.ipynb # Data fetching and cleaning
│   ├── 02_model_training.ipynb    # Model training and evaluation
│   ├── 03_anomaly_detection.ipynb # Anomaly detection analysis
│   └── 04_interpretability.ipynb  # SHAP analysis
├── scripts/
│   ├── fetch_data.py              # yfinance data retrieval
│   └── dashboard.py               # Streamlit dashboard
├── README.md                      # This file
└── requirements.txt               # Dependencies list
```

-----


## Evaluation Metrics & Expected Results

### Evaluation Metrics

| Metric | Target | Description |
| :--- | :--- | :--- |
| **RMSE/MAE** | RMSE \< 50 points (on normalized prices) | Standard measure of forecasting error. |
| **Directional Accuracy** | \>60% | Percentage of correct up/down predictions. |
| **Precision-Recall AUC** | \>0.85 (for anomalies) | Measures the model's ability to detect rare, high-risk market events. |
| **Coverage Probability** | \>90% | The percentage of true prices contained within the 95% confidence intervals. |

### Expected Results

  * **Forecasts:** Accurate 30-day price predictions with robust confidence intervals.
  * **Insights:** Clear identification of **volatility** as a high-impact feature using SHAP, directly informing risk-averse investment strategies.
  * **Anomalies:** Successful flagging of major market events.

-----

##  Future Extensions

  * **Multimodal Inputs:** Incorporate **news sentiment** data (via NewsAPI) for sentiment-driven forecasting.
  * **Real-Time Integration:** Implement scheduled daily `yfinance` pulls using **cron jobs** for true live prediction updates.

-----



