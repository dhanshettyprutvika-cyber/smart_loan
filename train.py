"""Smart Loan - train, compare and save the best loan approval model."""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             f1_score, roc_auc_score, accuracy_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/loan_data.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds TotalIncome, EMI and EMI_to_Income. Used identically in the app."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TotalIncome"] = X["ApplicantIncome"] + X["CoapplicantIncome"]
        # LoanAmount is in thousands
        X["EMI"] = X["LoanAmount"] * 1000 / X["Loan_Amount_Term"]
        X["EMI_to_Income"] = X["EMI"] / X["TotalIncome"].replace(0, np.nan)
        X["LogLoanAmount"] = np.log1p(X["LoanAmount"])
        return X


NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
           "Loan_Amount_Term", "Credit_History", "TotalIncome", "EMI",
           "EMI_to_Income", "LogLoanAmount"]
CATEGORICAL = ["Gender", "Married", "Dependents", "Education",
               "Self_Employed", "Property_Area"]


def build_pipeline(model):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("fe", FeatureEngineer()), ("pre", pre), ("model", model)])


def main():
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
    print("\nTarget balance:\n", df["Loan_Status"].value_counts(normalize=True).round(3))

    X = df.drop(columns=["Loan_Status"])
    if "Loan_ID" in X.columns:
        X = X.drop(columns=["Loan_ID"])
    y = (df["Loan_Status"] == "Y").astype(int)

    # --- EDA plots ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.countplot(x="Loan_Status", data=df, ax=ax[0]); ax[0].set_title("Class balance")
    sns.countplot(x="Credit_History", hue="Loan_Status", data=df, ax=ax[1])
    ax[1].set_title("Credit history vs approval")
    sns.histplot(df["ApplicantIncome"], bins=40, ax=ax[2]); ax[2].set_title("Applicant income")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/eda.png", dpi=120); plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8,
                                                class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows, fitted = [], {}
    for name, m in models.items():
        pipe = build_pipeline(m)
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "CV F1": round(cv_f1, 4),
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred), 4),
            "Recall": round(recall_score(y_test, pred), 4),
            "F1": round(f1_score(y_test, pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, proba), 4),
        })
        fitted[name] = pipe

    results = pd.DataFrame(rows).sort_values("CV F1", ascending=False)
    print("\n=== Model comparison (test set) ===")
    print(results.to_string(index=False))
    results.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

    best_name = results.iloc[0]["Model"]
    best = fitted[best_name]
    print(f"\nBest model (by CV F1): {best_name}")
    print(classification_report(y_test, best.predict(X_test),
                                target_names=["Rejected", "Approved"]))

    ConfusionMatrixDisplay.from_estimator(best, X_test, y_test,
                                          display_labels=["Rejected", "Approved"])
    plt.title(f"Confusion matrix - {best_name}")
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=120); plt.close()

    # --- Feature importance (tree models or logistic coefficients) ---
    feat_names = best.named_steps["pre"].get_feature_names_out()
    est = best.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    else:
        imp = np.abs(est.coef_[0])
    imp_df = (pd.DataFrame({"feature": feat_names, "importance": imp})
              .sort_values("importance", ascending=False).head(12))
    plt.figure(figsize=(8, 5))
    sns.barplot(x="importance", y="feature", data=imp_df)
    plt.title(f"Top features - {best_name}"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=120); plt.close()

    joblib.dump(best, "models/smart_loan_model.joblib")
    meta = {"best_model": best_name,
            "metrics": results.iloc[0].to_dict(),
            "top_features": imp_df.head(8)["feature"].tolist()}
    with open("models/meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print("\nSaved: models/smart_loan_model.joblib, outputs/*.png, outputs/model_comparison.csv")


if __name__ == "__main__":
    main()
