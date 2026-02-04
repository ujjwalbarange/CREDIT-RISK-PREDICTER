import streamlit as st
import pandas as pd
import joblib
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "best_extra_trees_model.pkl"))

encoders = {
    "Sex": joblib.load(os.path.join(BASE_DIR, "Sex_encoder.pkl")),
    "Housing": joblib.load(os.path.join(BASE_DIR, "Housing_encoder.pkl")),
    "Saving accounts": joblib.load(os.path.join(BASE_DIR, "Saving accounts_encoder.pkl")),
    
    "Checking account": joblib.load(os.path.join(BASE_DIR, "Checking account_encoder.pkl")),
    "Purpose": joblib.load(os.path.join(BASE_DIR, "Purpose_encoder.pkl")),
}


st.title("💳 Credit Risk Prediction")

st.write("Enter customer details to predict credit risk:")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.selectbox("Job", [0, 1, 2, 3])

housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich"])
checking_account = st.selectbox("Checking account", ["little", "moderate", "rich"])
credit_amount = st.number_input("Credit amount", min_value=100, max_value=100000, value=1000)
duration = st.number_input("Duration (months)", min_value=1, max_value=60, value=12)
purpose = st.selectbox(
    "Purpose",
    [
        "car", "furniture/equipment", "radio/TV",
        "domestic appliances", "repairs", "education",
        "vacation/others", "business", "retraining"
    ]
)


input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration],
    "Purpose": [encoders["Purpose"].transform([purpose])[0]],
})

# ================= PREDICTION =================
if st.button("Predict Credit Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.success(f"✅ Credit Risk: LOW (Confidence: {probability:.2%})")
    else:
        st.error(f"⚠️ Credit Risk: HIGH (Confidence: {probability:.2%})")
