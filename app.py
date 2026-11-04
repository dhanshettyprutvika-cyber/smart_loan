"""Smart Loan - Streamlit demo.  Run:  streamlit run app.py"""
import json

import joblib
import pandas as pd
import streamlit as st

# FeatureEngineer must be importable so joblib can unpickle the pipeline
from train import FeatureEngineer  # noqa: F401

st.set_page_config(page_title="Smart Loan", page_icon="💰", layout="centered")
st.title("💰 Smart Loan")
st.caption("ML-based loan approval prediction with risk score and reasons")


@st.cache_resource
def load():
    model = joblib.load("models/smart_loan_model.joblib")
    with open("models/meta.json") as f:
        meta = json.load(f)
    return model, meta


try:
    model, meta = load()
except FileNotFoundError:
    st.error("Model not found. Run `python generate_data.py` and `python train.py` first.")
    st.stop()

with st.sidebar:
    st.subheader("About the model")
    st.write(f"**Algorithm:** {meta['best_model']}")
    st.write(f"**F1:** {meta['metrics']['F1']}  |  **ROC-AUC:** {meta['metrics']['ROC-AUC']}")
    st.info("Educational project. Not for real lending decisions.")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self employed", ["No", "Yes"])
    area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
with col2:
    app_inc = st.number_input("Applicant monthly income", 0, 200000, 5000, step=500)
    co_inc = st.number_input("Co-applicant monthly income", 0, 100000, 0, step=500)
    loan_amt = st.number_input("Loan amount (in thousands)", 5, 1000, 150, step=5)
    term = st.selectbox("Loan term (months)", [360, 300, 240, 180, 120, 84, 60], index=0)
    credit = st.selectbox("Credit history meets guidelines?", [1.0, 0.0],
                          format_func=lambda v: "Yes" if v == 1.0 else "No")

if st.button("Check eligibility", type="primary"):
    row = pd.DataFrame([{
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_emp,
        "ApplicantIncome": app_inc, "CoapplicantIncome": co_inc,
        "LoanAmount": loan_amt, "Loan_Amount_Term": term,
        "Credit_History": credit, "Property_Area": area,
    }])
    p = float(model.predict_proba(row)[0, 1])

    total = app_inc + co_inc
    emi = loan_amt * 1000 / term
    ratio = emi / total if total else 1.0

    if p >= 0.5:
        st.success(f"✅ Likely APPROVED  (approval probability {p:.0%})")
    else:
        st.error(f"❌ Likely REJECTED  (approval probability {p:.0%})")
    st.progress(min(max(p, 0.0), 1.0))

    risk = "Low" if p >= 0.75 else "Medium" if p >= 0.5 else "High"
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk level", risk)
    c2.metric("Estimated EMI", f"{emi:,.0f}")
    c3.metric("EMI / income", f"{ratio:.0%}")

    st.subheader("Why?")
    reasons = []
    reasons.append("✔ Credit history is good." if credit == 1.0
                   else "✖ Credit history does not meet guidelines (the strongest negative factor).")
    if ratio > 0.4:
        reasons.append(f"✖ EMI is {ratio:.0%} of income, which is high (below 40% is safer).")
    else:
        reasons.append(f"✔ EMI is {ratio:.0%} of income, which is manageable.")
    if total < 3000:
        reasons.append("✖ Total household income is low.")
    for r in reasons:
        st.write(r)
    st.caption("Top model features: " + ", ".join(meta["top_features"][:5]))
