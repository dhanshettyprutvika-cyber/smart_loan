"""Generate a synthetic loan dataset with the same columns as the popular
Kaggle / Analytics Vidhya "Loan Prediction" dataset.

If you have the real dataset, save it as data/loan_data.csv with these columns
and skip this script:
Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
Property_Area, Loan_Status (Y/N)
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1500

gender = rng.choice(["Male", "Female"], N, p=[0.8, 0.2])
married = rng.choice(["Yes", "No"], N, p=[0.65, 0.35])
dependents = rng.choice(["0", "1", "2", "3+"], N, p=[0.57, 0.17, 0.16, 0.10])
education = rng.choice(["Graduate", "Not Graduate"], N, p=[0.78, 0.22])
self_emp = rng.choice(["No", "Yes"], N, p=[0.86, 0.14])
app_income = np.clip(rng.lognormal(8.5, 0.55, N), 1500, 80000).round()
co_income = np.where(
    rng.random(N) < 0.45, 0, np.clip(rng.lognormal(7.4, 0.7, N), 0, 40000)
).round()
loan_amount = np.clip(
    (app_income + co_income) * rng.uniform(0.02, 0.09, N) + rng.normal(0, 15, N), 9, 700
).round()
term = rng.choice([360, 180, 480, 300, 240, 120, 84, 60], N,
                  p=[0.80, 0.07, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01])
credit = rng.choice([1.0, 0.0], N, p=[0.84, 0.16])
area = rng.choice(["Urban", "Semiurban", "Rural"], N, p=[0.33, 0.38, 0.29])

# Hidden approval rule + noise (credit history dominates, like real data)
total_income = app_income + co_income
emi = loan_amount * 1000 / term
emi_ratio = emi / total_income
score = (
    3.2 * credit
    - 4.0 * emi_ratio * 10 / 10
    + 0.8 * (area == "Semiurban")
    + 0.3 * (education == "Graduate")
    + 0.25 * (married == "Yes")
    + 0.00003 * total_income
    - 2.4
    + rng.normal(0, 0.7, N)
)
prob = 1 / (1 + np.exp(-score))
status = np.where(rng.random(N) < prob, "Y", "N")

df = pd.DataFrame({
    "Gender": gender, "Married": married, "Dependents": dependents,
    "Education": education, "Self_Employed": self_emp,
    "ApplicantIncome": app_income, "CoapplicantIncome": co_income,
    "LoanAmount": loan_amount, "Loan_Amount_Term": term,
    "Credit_History": credit, "Property_Area": area, "Loan_Status": status,
})

# Inject realistic missing values
for col, frac in [("Gender", .02), ("Married", .01), ("Dependents", .02),
                  ("Self_Employed", .05), ("LoanAmount", .04),
                  ("Loan_Amount_Term", .02), ("Credit_History", .08)]:
    df.loc[rng.random(N) < frac, col] = np.nan

os.makedirs("data", exist_ok=True)
df.to_csv("data/loan_data.csv", index=False)
print(f"Saved data/loan_data.csv  shape={df.shape}")
print(df["Loan_Status"].value_counts(normalize=True).round(3).to_dict())
