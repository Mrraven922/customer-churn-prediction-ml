# Encode Gender: Female = 1, Male = 0
# Encode Churn: Yes = 1, No = 0

# Exported files
#scaler_file = "scaler.pkl"   # Scaler
#model_file = "model.pkl"     # Model

# Feature order for X
#feature_order = ["Age", "Gender", "Tenure", "MonthlyCharges"]


import streamlit as st
import joblib
import numpy as np

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# App title
st.title("Churn Prediction App")
st.divider()
st.write("Please enter the values and hit the Predict button to get a prediction.")
st.divider()

# Input fields
age = st.number_input("Enter Age", min_value=10, max_value=100, value=38)
tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)
monthly_charge = st.number_input("Enter Monthly Charge", min_value=38, max_value=150, value=50)
gender = st.selectbox("Enter the Gender", ["Male", "Female"])
st.divider()

# Predict button
predict_button = st.button("Predict")

if predict_button:
    # Encode gender
    gender_selected = 1 if gender == "Female" else 0
    
    # Create input array
    X = [age, gender_selected, tenure, monthly_charge]
    X_array = np.array([X])
    
    # Scale inputs
    X_scaled = scaler.transform(X_array)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    # Convert prediction to readable output
    predicted = "Yes" if prediction == 1 else "No"
    
    # Display result
    st.balloons()
    st.write(f"Predicted Churn: {predicted}")
else:
    st.write("Please enter the values and click the Predict button.")


# run the app with: python -m streamlit run app.py