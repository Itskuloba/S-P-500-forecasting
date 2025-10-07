import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import os
import matplotlib.pyplot as plt

# --- 1. Define Custom Objects for Keras Model Loading ---
CUSTOM_OBJECTS = {
    'MultiHeadAttention': MultiHeadAttention,
    'LayerNormalization': LayerNormalization,
    'Add': Add
}

# --- 2. Resource Loading ---
@st.cache_resource
def load_assets():
    """Loads the model and the two scalers."""
    MODEL_DIR = 'streamlit_model'
    
    if not os.path.exists(MODEL_DIR):
        st.error(f"Directory '{MODEL_DIR}' does not exist! Ensure it was created when saving in data.ipynb.")
        st.info("💡 Fix: Run the save cell in data.ipynb, or copy files to 'streamlit_model/'.")
        return None, None, None
    
    model_path = os.path.join(MODEL_DIR, 'hybrid_lstm_transformer.keras')
    x_scaler_path = os.path.join(MODEL_DIR, 'X_scaler.pkl')
    y_scaler_path = os.path.join(MODEL_DIR, 'y_scaler.pkl')
    
    if not os.path.exists(model_path):
        st.error(f"Model file missing: {model_path}")
        st.info("💡 Fix: Re-run the save cell in data.ipynb.")
        model = None
    else:
        try:
            model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            model = None
    
    if not os.path.exists(x_scaler_path):
        st.error(f"X_scaler file missing: {x_scaler_path}")
        X_scaler = None
    else:
        try:
            X_scaler = joblib.load(x_scaler_path)
        except Exception as e:
            st.error(f"Error loading X_scaler: {e}")
            X_scaler = None
    
    if not os.path.exists(y_scaler_path):
        st.error(f"y_scaler file missing: {y_scaler_path}")
        y_scaler = None
    else:
        try:
            y_scaler = joblib.load(y_scaler_path)
        except Exception as e:
            st.error(f"Error loading y_scaler: {e}")
            y_scaler = None
        
    return model, X_scaler, y_scaler

model, X_scaler, y_scaler = load_assets()

# --- 3. Fetch Historical Data ---
@st.cache_data
def fetch_historical_data():
    """Fetches the last 60 days of S&P 500 data, computes features, and returns the last 4 days."""
    try:
        end_date = pd.Timestamp.now().date()
        start_date = end_date - pd.Timedelta(days=60)
        sp500 = yf.download('^GSPC', start=start_date, end=end_date, interval='1d')
        
        if sp500.empty:
            st.error("Failed to fetch historical data: No data returned from yfinance.")
            return None, None
        
        # Flatten MultiIndex columns if present
        if sp500.columns.nlevels > 1:
            sp500.columns = sp500.columns.get_level_values(0)
        
        # Compute features
        sp500['Daily_Return'] = sp500['Close'].pct_change()
        sp500['MA_20'] = sp500['Close'].rolling(window=20).mean()
        sp500['Adj Close'] = sp500['Close']
        
        # Select only the required 8 features
        feature_names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Daily_Return', 'MA_20']
        sp500 = sp500[feature_names]
        
        sp500 = sp500.dropna()
        
        if len(sp500) < 4:
            st.error(f"Not enough historical data: Got {len(sp500)} days, need at least 4.")
            st.warning("Using fallback dummy data to proceed (predictions may be less accurate).")
            dummy_data = pd.DataFrame([
                {
                    'Open': 5000.0, 'High': 5050.0, 'Low': 4990.0, 'Close': 5025.0,
                    'Adj Close': 5025.0, 'Volume': 4600000000, 'Daily_Return': 0.0, 'MA_20': 5025.0
                }
            ] * 4, index=pd.date_range(end=end_date - pd.Timedelta(days=1), periods=4, freq='B'))
            return dummy_data, dummy_data
        
        historical = sp500.tail(4)
        return historical, sp500
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        st.warning("Using fallback dummy data to proceed (predictions may be less accurate).")
        dummy_data = pd.DataFrame([
            {
                'Open': 5000.0, 'High': 5050.0, 'Low': 4990.0, 'Close': 5025.0,
                'Adj Close': 5025.0, 'Volume': 4600000000, 'Daily_Return': 0.0, 'MA_20': 5025.0
            }
        ] * 4, index=pd.date_range(end=end_date - pd.Timedelta(days=1), periods=4, freq='B'))
        return dummy_data, dummy_data

# --- 4. Prediction Logic ---
def preprocess_and_predict(input_df):
    """Handles scaling, reshaping, prediction, and inverse scaling."""
    if None in [model, X_scaler, y_scaler]:
        st.error("Cannot run prediction: Model or Scalers failed to load.")
        return None, None

    SEQUENCE_LENGTH = 5
    NUM_FEATURES = 8
    
    if len(input_df) != SEQUENCE_LENGTH:
        st.error(f"Expected {SEQUENCE_LENGTH} days of data, got {len(input_df)}")
        return None, None
    
    # Ensure only the expected features are used
    feature_names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Daily_Return', 'MA_20']
    if list(input_df.columns) != feature_names:
        st.warning(f"Column mismatch: Expected {feature_names}, got {list(input_df.columns)}")
        try:
            input_df = input_df[feature_names]
        except KeyError as e:
            st.error(f"Failed to select expected columns: {e}")
            return None, None
    
    X_unscaled = input_df.values
    if X_unscaled.shape[1] != NUM_FEATURES:
        st.error(f"Scaling Error: Input has {X_unscaled.shape[1]} features, expected {NUM_FEATURES}.")
        return None, None
    
    try:
        X_scaled = X_scaler.transform(X_unscaled)  # Shape: (5, 8)
        # Debug: Show scaled input for last row
        with st.expander("Debug: Scaled Input Data"):
            st.write(f"Scaled input for last day (user input): {X_scaled[-1]}")
    except ValueError as e:
        st.error(f"Scaling Error: {e}")
        st.error("The number of input features (8) must match the number of features the scaler was fitted on.")
        return None, None
    
    X_pred_seq = X_scaled.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
    scaled_prediction = model.predict(X_pred_seq, verbose=0)[0, 0]
    real_prediction_2d = np.array([[scaled_prediction]])
    real_prediction = y_scaler.inverse_transform(real_prediction_2d)[0, 0]
    
    # Clip prediction to plausible S&P 500 range
    real_prediction = np.clip(real_prediction, 1000, 10000)
    
    return real_prediction, scaled_prediction

# --- Streamlit UI ---
st.set_page_config(
    page_title="S&P 500 Daily Close Price Predictor",
    layout="wide"
)

st.title("S&P 500 Next Day Close Price Prediction")
st.markdown("Enter today's key financial indicators to predict the S&P 500 Close Price for tomorrow. Historical data for the previous 4 days will be automatically fetched.")

if None in [model, X_scaler, y_scaler]:
    st.warning("Please ensure all model files are loaded correctly.")
    st.stop()

# --- User Input Section ---
st.header("Today's Key Market Features")

feature_map = {
    'Close': 'Close Price',
    'Volume': 'Volume',
    'MA_20': 'MA-20 Price'
}
feature_names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Daily_Return', 'MA_20']
INTERNAL_DEFAULTS = {
    'Open': 5000.0,
    'High': 5050.0,
    'Low': 4990.0,
    'Adj Close': 5020.0,
    'Daily_Return': 0.0000,
    'Close': 5025.0,
    'Volume': 4600000000,
    'MA_20': 5030.0
}

input_values = {}
cols = st.columns(3)

for i, key in enumerate(feature_map.keys()):
    label = feature_map[key]
    with cols[i]:
        if 'Price' in label or 'MA-20' in label:
            value_format = "%.2f"
            default_value = 5025.0 if key == 'Close' else (5030.0 if key == 'MA_20' else 5000.0)
            min_val = 0.0
        elif key == 'Volume':
            value_format = "%d"
            default_value = 4600000000
            min_val = 0
        input_values[key] = st.number_input(
            label,
            value=default_value,
            format=value_format,
            min_value=min_val,
            key=f"input_{key}"
        )

final_input_data = {}
for name in feature_names:
    final_input_data[name] = input_values.get(name, INTERNAL_DEFAULTS[name])

user_df = pd.DataFrame([final_input_data], columns=feature_names)

historical_data, sp500_full = fetch_historical_data()
if historical_data is None:
    st.stop()

# Ensure historical_data has correct columns
if list(historical_data.columns) != feature_names:
    st.warning(f"Historical data column mismatch: Expected {feature_names}, got {list(historical_data.columns)}")
    try:
        historical_data = historical_data[feature_names]
    except KeyError as e:
        st.error(f"Failed to select expected columns in historical_data: {e}")
        st.stop()

input_df = pd.concat([historical_data, user_df], ignore_index=True)

with st.expander("View Combined 5-Day Input Data"):
    st.dataframe(input_df)

# --- Prediction Button ---
st.markdown("---")
if st.button("Predict Next Day's Close Price", type="primary"):
    with st.spinner("Processing data and predicting..."):
        predicted_real_price, predicted_scaled_price = preprocess_and_predict(input_df)
        
        if predicted_real_price is not None:
            st.header("Prediction Result")
            
            todays_close = input_df['Close'].iloc[-1]
            delta_value = predicted_real_price - todays_close
            delta_text = f"{'UP' if delta_value >= 0 else 'DOWN'} by ${abs(delta_value):.2f}"
            color = "normal" if delta_value >= 0 else "inverse"
            
            st.metric(
                label="Predicted Next Day Close Price",
                value=f"${predicted_real_price:.2f}",
                delta=delta_text,
                delta_color=color
            )
            
            # Plot S&P 500 Open and Close prices
            if sp500_full is not None:
                st.header("S&P 500 Open and Close Prices")
                plt.figure(figsize=(14, 7))
                sp500_full['Open'].plot(label='Open Price')
                sp500_full['Close'].plot(label='Close Price')
                plt.title('S&P 500 Open and Close Prices Over Time')
                plt.xlabel('Date')
                plt.ylabel('Price (USD)')
                plt.legend()
                plt.grid(True)
                
                # Save the plot
                plot_path = 'sp500_open_close_prices.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                
                # Display the plot in Streamlit
                st.pyplot(plt)
                
                # Provide download button for the plot
                with open(plot_path, "rb") as file:
                    st.download_button(
                        label="Download Plot",
                        data=file,
                        file_name="sp500_open_close_prices.png",
                        mime="image/png"
                    )
                
                plt.close()
