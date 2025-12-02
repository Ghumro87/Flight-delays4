# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Flight Delay Predictor")
st.write("Enter flight details to predict if it will be delayed.")

# --- 2. Load Models and Preprocessor ---
@st.cache_resource
def load_resources():
    preprocessor = joblib.load('preprocessor.joblib')
    log_reg_model = joblib.load('log_reg_model.joblib')
    tree_model = joblib.load('tree_model.joblib')
    rf_model = joblib.load('rf_model.joblib')
    return preprocessor, log_reg_model, tree_model, rf_model

preprocessor, log_reg_model, tree_model, rf_model = load_resources()

# --- 3. Feature Definitions (Must match training script) ---
numeric_features = [
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DISTANCE",
    "DAY_WEEK",
    "DAY_OF_MONTH",
    "Weather"
]

categorical_features = [
    "CARRIER",
    "ORIGIN",
    "DEST"
]

# --- 4. Define Unique Categorical Values (from training data) ---
# These values are extracted from the original training dataframe.
carrier_options = ['OH', 'DH', 'DL', 'MQ', 'UA', 'US', 'RU', 'CO']
origin_options = ['BWI', 'DCA', 'IAD']
dest_options = ['JFK', 'LGA', 'EWR']

# For demonstration, creating a dictionary for easy access
cat_options = {
    'CARRIER': carrier_options,
    'ORIGIN': origin_options,
    'DEST': dest_options
}

# --- 5. Model Dictionary ---
models = {
    "Logistic Regression": log_reg_model,
    "Decision Tree": tree_model,
    "Random Forest": rf_model
}

st.write("Models and preprocessor loaded successfully!")

# --- 6. Input Widgets ---
st.header("Flight Details")

# Helper function to create input features
def get_user_inputs():
    # Numerical Inputs
    crs_dep_time = st.slider("Scheduled Departure Time (24h format e.g., 1400 for 2 PM)", 0, 2359, 1200, step=1)
    dep_time = st.slider("Actual Departure Time (24h format e.g., 1400 for 2 PM)", 0, 2359, 1200, step=1)
    distance = st.number_input("Distance (miles)", min_value=10, max_value=5000, value=200)
    day_week = st.selectbox("Day of Week", options=list(range(1, 8)), format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x-1], index=0) # Monday is 1, Sunday is 7
    day_of_month = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
    weather = st.selectbox("Weather Delay Indicator", options=[0, 1], format_func=lambda x: 'No Delay' if x == 0 else 'Delay', index=0)

    # Categorical Inputs
    carrier = st.selectbox("Carrier", options=cat_options['CARRIER'])
    origin = st.selectbox("Origin Airport", options=cat_options['ORIGIN'])
    dest = st.selectbox("Destination Airport", options=cat_options['DEST'])

    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        'CRS_DEP_TIME': [crs_dep_time],
        'DEP_TIME': [dep_time],
        'DISTANCE': [distance],
        'DAY_WEEK': [day_week],
        'DAY_OF_MONTH': [day_of_month],
        'Weather': [weather],
        'CARRIER': [carrier],
        'ORIGIN': [origin],
        'DEST': [dest]
    })
    return input_data

user_input_df = get_user_inputs()

# --- 7. Model Selection and Prediction ---
st.subheader("Prediction")

selected_model_name = st.selectbox("Select Model for Prediction", list(models.keys()))
selected_model = models[selected_model_name]

if st.button("Predict Flight Delay"):
    try:
        prediction_proba = selected_model.predict_proba(user_input_df)[:, 1][0]
        prediction_class = (prediction_proba > 0.5).astype(int)

        st.write(f"#### Selected Model: {selected_model_name}")
        st.write(f"Probability of Delay: {prediction_proba:.2f}")

        if prediction_class == 1:
            st.error("Prediction: Flight DELAYED")
        else:
            st.success("Prediction: Flight ON-TIME")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.write("--- All models are trained to predict flight delays based on the provided features. --- ")
