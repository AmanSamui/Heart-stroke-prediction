
import streamlit as st   # add UI
import pandas as pd
import joblib


model=joblib.load("KNN_heart.pkl")
scaler=joblib.load("scaler.pkl")
expected_columns=joblib.load("columns.pkl")

st.title("Heart stroke prediction")
# st.markdown()
st.markdown("Provide the following details")



# Numeric inputs
age = st.number_input("Age", 18, 100, 40)
resting_bp = st.number_input("Resting BP", 80, 220, 120)
cholesterol = st.number_input("Cholesterol", 0, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

# Categorical inputs
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)
resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)
exercise_angina = st.selectbox(
    "Exercise Angina",
    ["Y", "N"]
)
st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

if st.button("Predict"):
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }
        # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
