# Smart Loan: ML Loan Approval Predictor

Predicts whether a loan application is likely to be approved, shows an approval
probability, a risk level, and plain-language reasons.

## Project structure

```
smart_loan/
├── generate_data.py   # builds a synthetic dataset (skip if you have the real CSV)
├── train.py           # EDA, preprocessing, model comparison, saves best model
├── app.py             # Streamlit demo
├── requirements.txt
├── data/              # loan_data.csv goes here
├── models/            # saved model + metadata (created by train.py)
└── outputs/           # plots and comparison table (created by train.py)
```

## How to run

```bash
pip install -r requirements.txt
python generate_data.py      # or place the real Kaggle CSV at data/loan_data.csv
python train.py              # trains 4 models, prints metrics, saves the best
streamlit run app.py         # opens the demo in your browser
```

### Using the real dataset
Download "Loan Prediction" from Kaggle / Analytics Vidhya and save it as
`data/loan_data.csv`. Keep the column names (Gender, Married, Dependents,
Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
Loan_Amount_Term, Credit_History, Property_Area, Loan_Status with Y/N).
A `Loan_ID` column is dropped automatically. Then just run `train.py`.

## Method

1. **Data cleaning**: median imputation (numeric), most-frequent (categorical).
2. **Feature engineering**: TotalIncome, EMI, EMI_to_Income, LogLoanAmount.
3. **Encoding and scaling**: one-hot encoding, standard scaling, all inside a single
   scikit-learn Pipeline (no data leakage).
4. **Models compared**: Logistic Regression (baseline), Decision Tree,
   Random Forest, Gradient Boosting, with 5-fold stratified CV.
5. **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix.
6. **Explainability**: feature importance plot plus rule-based reasons in the app.

## Limitations and future scope

- The bundled data is synthetic; results on real data will differ.
- Historical lending data can encode bias (e.g., gender, area). A production
  system would need fairness audits and should not use protected attributes.
- Future work: SHAP explanations, hyperparameter tuning, SMOTE, loan amount
  and interest-rate recommendation, deployment on Streamlit Cloud.
