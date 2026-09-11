import streamlit as st
import pandas as pd
import joblib

# Load trained model, expected feature columns, and the scaler
model = joblib.load('gb_classifier.pkl')
model_columns = joblib.load('feature_names.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Cardiovascular Risk Assessment Tool")

# Collect ALL inputs present in the raw dataset
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
oldpeak = st.number_input("Oldpeak", value=0.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Assess Risk"):
    # Convert user input into a single-row DataFrame
    input_dict = {
        'Age': age, 'Sex': sex, 'ChestPainType': chest_pain,
        'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBS': fasting_bs, 'RestingECG': resting_ecg,
        'MaxHR': max_hr, 'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak, 'ST_Slope': st_slope
    }
    input_df = pd.DataFrame([input_dict])

    # Apply dummy encoding
    input_df_encoded = pd.get_dummies(input_df)

    # Reindex to match model features and fill missing dummies with 0[cite: 1]
    input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale the continuous numerical columns
    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    input_df_encoded[num_cols] = scaler.transform(input_df_encoded[num_cols])

    # Predict[cite: 1]
    prediction = model.predict(input_df_encoded)

    if prediction[0] == 1:
        st.error("Alert: Elevated Cardiovascular Risk Detected.")
    else:
        st.success("Result: Low Cardiovascular Risk.")