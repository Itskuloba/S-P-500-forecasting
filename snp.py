import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
import numpy as np
import pandas as pd
import joblib

# --- 1. Define Custom Objects for Keras Model Loading ---
def custom_multi_head_attention(num_heads, key_dim):
    return MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

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
    
    # 2.1 Load Keras Model
    try:
        model = load_model(f'{MODEL_DIR}/hybrid_lstm_transformer.h5', custom_objects=CUSTOM_OBJECTS)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
        
    # 2.2 Load X_scaler (Feature Scaler)
    try:
        X_scaler = joblib.load(f'{MODEL_DIR}/X_scaler.pkl')
    except Exception as e:
        st.error(f"Error loading X_scaler: {e}")
        X_scaler = None
        
    # 2.3 Load Y_scaler (Target Scaler - for inverse transform)
    try:
        y_scaler = joblib.load(f'{MODEL_DIR}/y_scaler.pkl')
    except Exception as e:
        st.error(f"Error loading y_scaler: {e}")
        y_scaler = None
        
    return model, X_scaler, y_scaler

model, X_scaler, y_scaler = load_assets()

# --- 3. Prediction Logic ---

def preprocess_and_predict(input_df):
    """Handles scaling, reshaping, prediction, and inverse scaling."""
    
    if None in [model, X_scaler, y_scaler]:
        st.error("Cannot run prediction: Model or Scalers failed to load.")
        return None, None

    # THE CRITICAL CHANGE: Model now expects 8 features
    SEQUENCE_LENGTH = 5  # The time steps the model looks back
    NUM_FEATURES = 8     # MUST match the number of features the model was trained on

    
    # 1. Scaling the Input Data (8 features)
    X_unscaled = input_df.values
    
    try:
        X_scaled = X_scaler.transform(X_unscaled)
    except ValueError as e:
        st.error(f"Scaling Error: {e}")
        st.error("The number of input features (8) must match the number of features the scaler was fitted on.")
        return None, None
    
    # 2. Reshaping for Sequence Model (LSTM/Transformer)
    # Tile the single day's 8 features to create the (1, 5, 8) input shape
    X_pred_seq = np.tile(X_scaled[0], (SEQUENCE_LENGTH, 1))
    X_pred_seq = X_pred_seq.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)

    # 3. Prediction (outputs a scaled return)
    scaled_prediction = model.predict(X_pred_seq, verbose=0)[0, 0]
    
    # 4. Inverse Scaling (outputs the real return)
    real_prediction_2d = np.array([[scaled_prediction]])
    real_prediction = y_scaler.inverse_transform(real_prediction_2d)[0, 0]
    
    return real_prediction, scaled_prediction


#  Streamlit UI 

st.set_page_config(
    page_title="S&P 500 Daily Return Predictor",
    layout="wide"
)

st.title("🚀 Hybrid LSTM-Transformer: Daily Return Prediction")
st.markdown("Enter today's ** key financial indicators** to predict the S&P 500 Daily Return for tomorrow.")

if None in [model, X_scaler, y_scaler]:
    st.warning("Please ensure all model files are loaded correctly.")
    st.stop()


# --- User Input Section ---
st.header("1. Today's 8 Features")

# Feature columns based on the likely 8 columns in your X_train DataFrame
feature_names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Daily_Return', 'MA_20']
column_labels = [
    'Open Price', 'High Price', 'Low Price', 'Close Price', 
    'Adjusted Close Price', 'Volume', 'Daily Return (%)', 'MA-20 Price'
]
# FIX: Changed all price and MA values to floating-point numbers (e.g., 5000.0) 
# to match the format string '%.2f' and suppress the Streamlit warning.
default_values = [5000.0, 5050.0, 4990.0, 5020.0, 5020.0, 4500000000.0, 0.005, 5000.0] 

# Create input fields for the 8 features (using four columns for a clean layout)
cols = st.columns(4)
input_values = {}

for i in range(8):
    # Determine which column to place the input in
    col_index = i % 4
    
    with cols[col_index]:
        # Use an appropriate label and format based on the feature
        if 'Price' in column_labels[i]:
            value_format = "%.2f"
        elif 'Return' in column_labels[i]:
            value_format = "%.4f"
        else:
            # Volume is the only remaining feature; ensuring it's a float
            value_format = "%d" 

        input_values[feature_names[i]] = st.number_input(
            column_labels[i], 
            value=default_values[i], 
            format=value_format
        )

# Prepare the data frame for prediction, ensuring column order matches the original X_train
input_df = pd.DataFrame(
    [[input_values[name] for name in feature_names]], 
    columns=feature_names
)


# --- Prediction Button ---
st.markdown("---")
if st.button("Predict Next Day's Daily Return", type="primary"):
    
    with st.spinner("Processing data and predicting..."):
        
        # Run the prediction
        predicted_real_return, predicted_scaled_return = preprocess_and_predict(input_df)
        
        if predicted_real_return is not None:
            st.header("2. Prediction Result")
            
            # Convert prediction to percentage
            return_percentage = predicted_real_return * 100
            
            # Determine delta for visual feedback
            if return_percentage >= 0:
                delta_text = f"UP by {abs(return_percentage):.3f}%"
                color = "normal" 
            else:
                delta_text = f"DOWN by {abs(return_percentage):.3f}%"
                color = "inverse"
                
            st.metric(
                label="Predicted Daily Return (Tomorrow)", 
                value=f"{return_percentage:.3f}%",
                delta=delta_text,
                delta_color=color
            )
            
            st.markdown(f"""
            <div style="font-size: 14px; color: gray;">
                Raw Model Output (Scaled): {predicted_scaled_return:.6f}
            </div>
            """, unsafe_allow_html=True)

# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
# import numpy as np
# import pandas as pd
# import joblib

# # --- 1. Define Custom Objects for Keras Model Loading ---
# def custom_multi_head_attention(num_heads, key_dim):
#     return MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

# CUSTOM_OBJECTS = {
#     'MultiHeadAttention': MultiHeadAttention,
#     'LayerNormalization': LayerNormalization,
#     'Add': Add
# }

# # --- 2. Resource Loading ---

# @st.cache_resource
# def load_assets():
#     """Loads the model and the two scalers."""
    
#     MODEL_DIR = 'streamlit_model'
    
#     # 2.1 Load Keras Model
#     try:
#         model = load_model(f'{MODEL_DIR}/hybrid_lstm_transformer.h5', custom_objects=CUSTOM_OBJECTS)
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         model = None
        
#     # 2.2 Load X_scaler (Feature Scaler)
#     try:
#         X_scaler = joblib.load(f'{MODEL_DIR}/X_scaler.pkl')
#     except Exception as e:
#         st.error(f"Error loading X_scaler: {e}")
#         X_scaler = None
        
#     # 2.3 Load Y_scaler (Target Scaler - for inverse transform)
#     try:
#         y_scaler = joblib.load(f'{MODEL_DIR}/y_scaler.pkl')
#     except Exception as e:
#         st.error(f"Error loading y_scaler: {e}")
#         y_scaler = None
        
#     return model, X_scaler, y_scaler

# model, X_scaler, y_scaler = load_assets()

# # --- 3. Prediction Logic ---

# def preprocess_and_predict(input_df):
#     """Handles scaling, reshaping, prediction, and inverse scaling."""
    
#     if None in [model, X_scaler, y_scaler]:
#         st.error("Cannot run prediction: Model or Scalers failed to load.")
#         return None, None

#     # THE CRITICAL CHANGE: Model now expects 8 features
#     SEQUENCE_LENGTH = 5  # The time steps the model looks back
#     NUM_FEATURES = 8     # MUST match the number of features the model was trained on

    
#     # 1. Scaling the Input Data (8 features)
#     X_unscaled = input_df.values
    
#     try:
#         X_scaled = X_scaler.transform(X_unscaled)
#     except ValueError as e:
#         st.error(f"Scaling Error: {e}")
#         st.error("The number of input features (8) must match the number of features the scaler was fitted on.")
#         return None, None
    
#     # 2. Reshaping for Sequence Model (LSTM/Transformer)
#     # Tile the single day's 8 features to create the (1, 5, 8) input shape
#     X_pred_seq = np.tile(X_scaled[0], (SEQUENCE_LENGTH, 1))
#     X_pred_seq = X_pred_seq.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)

#     # 3. Prediction (outputs a scaled return)
#     scaled_prediction = model.predict(X_pred_seq, verbose=0)[0, 0]
    
#     # 4. Inverse Scaling (outputs the real return)
#     real_prediction_2d = np.array([[scaled_prediction]])
#     real_prediction = y_scaler.inverse_transform(real_prediction_2d)[0, 0]
    
#     return real_prediction, scaled_prediction


# # --- 4. Streamlit UI ---

# st.set_page_config(
#     page_title="S&P 500 Daily Return Predictor (Scaled)",
#     layout="wide"
# )

# st.title("🚀 Hybrid LSTM-Transformer: Daily Return Prediction")
# st.markdown("Enter today's **8 key financial indicators** to predict the S&P 500 Daily Return for tomorrow.")

# if None in [model, X_scaler, y_scaler]:
#     st.warning("Please ensure all model files are loaded correctly.")
#     st.stop()


# # --- User Input Section ---
# st.header("1. Today's 8 Features")

# # Feature columns based on the likely 8 columns in your X_train DataFrame
# feature_names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Daily_Return', 'MA_20']
# column_labels = [
#     'Open Price', 'High Price', 'Low Price', 'Close Price', 
#     'Adjusted Close Price', 'Volume', 'Daily Return (%)', 'MA-20 Price'
# ]
# default_values = [5000, 5050, 4990, 5020, 5020, 4500000000, 0.005, 5000]

# # Create input fields for the 8 features (using four columns for a clean layout)
# cols = st.columns(4)
# input_values = {}

# for i in range(8):
#     # Determine which column to place the input in
#     col_index = i % 4
    
#     with cols[col_index]:
#         # Use an appropriate label and format based on the feature
#         if 'Price' in column_labels[i]:
#             value_format = "%.2f"
#         elif 'Return' in column_labels[i]:
#             value_format = "%.4f"
#         else:
#             value_format = "%d"

#         input_values[feature_names[i]] = st.number_input(
#             column_labels[i], 
#             value=default_values[i], 
#             format=value_format
#         )

# # Prepare the data frame for prediction, ensuring column order matches the original X_train
# input_df = pd.DataFrame(
#     [[input_values[name] for name in feature_names]], 
#     columns=feature_names
# )


# # --- Prediction Button ---
# st.markdown("---")
# if st.button("Predict Next Day's Daily Return", type="primary"):
    
#     with st.spinner("Processing data, scaling, predicting, and inverse scaling..."):
        
#         # Run the prediction
#         predicted_real_return, predicted_scaled_return = preprocess_and_predict(input_df)
        
#         if predicted_real_return is not None:
#             st.header("2. Prediction Result")
            
#             # Convert prediction to percentage
#             return_percentage = predicted_real_return * 100
            
#             # Determine delta for visual feedback
#             # FIX: Change 'success' to 'normal' as required by StreamlitAPIException
#             if return_percentage >= 0:
#                 delta_text = f"UP by {abs(return_percentage):.3f}%"
#                 color = "normal" # Use 'normal' for green (positive)
#             else:
#                 delta_text = f"DOWN by {abs(return_percentage):.3f}%"
#                 color = "inverse" # Use 'inverse' for red (negative)
                
#             st.metric(
#                 label="Predicted Daily Return (Tomorrow)", 
#                 value=f"{return_percentage:.3f}%",
#                 delta=delta_text,
#                 delta_color=color
#             )
            
#             st.markdown(f"""
#             <div style="font-size: 14px; color: gray;">
#                 Raw Model Output (Scaled): {predicted_scaled_return:.6f}
#             </div>
#             """, unsafe_allow_html=True)





# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
# import numpy as np
# import pandas as pd

# # --- 1. Define Custom Objects for Keras Model Loading ---
# CUSTOM_OBJECTS = {
#     "MultiHeadAttention": MultiHeadAttention,
#     "LayerNormalization": LayerNormalization,
#     "Add": Add
# }

# # --- 2. Load Model ---
# @st.cache_resource
# def load_hybrid_model():
#     try:
#         model = load_model(
#             "streamlit_model/hybrid_lstm_transformer.h5",
#             custom_objects=CUSTOM_OBJECTS
#         )
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         st.info("Make sure you have saved the model using the previous code into the 'streamlit_model' folder.")
#         return None

# model = load_hybrid_model()

# # --- 3. Prediction Logic ---
# def preprocess_and_predict(input_data):
#     """
#     Prepares a single row of feature data for prediction and runs the model.
#     """
#     if model is None:
#         return "Model not loaded."

#     sequence_length = 5
#     num_features = 8  # <-- Updated to match training (Close, High, Low, Open, Volume, Daily_Return, MA_20, Lag_1)

#     # Repeat the input row to form a 5-day sequence
#     input_array = input_data.values[0]
#     X_pred_seq = np.tile(input_array, (sequence_length, 1))
#     X_pred_seq = X_pred_seq.reshape(1, sequence_length, num_features)

#     st.subheader("Model Input Shape")
#     st.code(f"Input data reshaped to: {X_pred_seq.shape} (Batch, Sequence, Features)")

#     # Run prediction (still scaled)
#     scaled_prediction = model.predict(X_pred_seq, verbose=0)[0, 0]

#     return scaled_prediction


# # --- 4. Streamlit UI ---
# st.set_page_config(
#     page_title="S&P 500 Daily Return Predictor (Hybrid LSTM-Transformer)",
#     layout="wide"
# )

# st.title("S&P 500 Daily Return Prediction")
# st.markdown("Forecasting the S&P 500 Daily Return (`Target` column) using a Hybrid **LSTM-Transformer** model.")

# if model is None:
#     st.stop()

# # --- User Input Section ---
# st.header("1. Input Features for Today")

# # Features used for training
# feature_names = ['Close', 'High', 'Low', 'Open', 'Volume', 'Daily_Return', 'MA_20', 'Lag_1']

# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     close = st.number_input("Close", value=1100.00, format="%.2f")
# with col2:
#     high = st.number_input("High", value=1105.00, format="%.2f")
# with col3:
#     low = st.number_input("Low", value=1090.00, format="%.2f")
# with col4:
#     open_price = st.number_input("Open", value=1092.00, format="%.2f")

# col5, col6, col7, col8 = st.columns(4)
# with col5:
#     volume = st.number_input("Volume", value=4_700_000_000)
# with col6:
#     daily_return = st.number_input("Daily_Return", value=0.005, format="%.4f")
# with col7:
#     ma_20 = st.number_input("MA_20", value=1100.00, format="%.2f")
# with col8:
#     lag_1 = st.number_input("Lag_1", value=1095.00, format="%.2f")

# # Build DataFrame for model input
# input_df = pd.DataFrame(
#     [[close, high, low, open_price, volume, daily_return, ma_20, lag_1]],
#     columns=feature_names
# )

# # --- Prediction Button ---
# st.markdown("---")
# if st.button("Predict Next Day's Daily Return", type="primary"):

#     with st.spinner("Processing data and running Hybrid Model..."):
#         predicted_scaled_return = preprocess_and_predict(input_df)

#         st.header("2. Prediction Result")

#         st.warning("WARNING: The prediction below is **SCALED** because the required 'y_scaler' was not loaded. The model was trained on scaled data.")

#         st.metric(
#             label="Predicted Scaled Daily Return",
#             value=f"{predicted_scaled_return:.6f}"
#         )

#         st.info(f"To get the true percentage return (e.g., +0.5%), you must load and use the 'y_scaler' to inverse-transform the predicted value of {predicted_scaled_return:.6f}.")
