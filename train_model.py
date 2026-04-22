import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import shap
import joblib
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

def generate_dataset(n=5000):
    credit_score   = np.random.normal(650, 100, n).clip(300, 850).astype(int)
    annual_income  = np.random.lognormal(mean=11.5, sigma=0.6, size=n).clip(120000, 5000000)
    dti            = np.random.beta(2, 5, n) * 70
    emp_years      = np.random.exponential(scale=5, size=n).clip(0, 40)
    existing_loans = np.random.poisson(lam=1.2, size=n).clip(0, 6)
    loan_amount    = np.random.lognormal(mean=12, sigma=0.8, size=n).clip(10000, 2000000)
    loan_term      = np.random.choice([12, 24, 36, 48, 60], n)
    age            = np.random.normal(38, 10, n).clip(21, 70).astype(int)
    loan_to_income = loan_amount / annual_income
    monthly_income = annual_income / 12
    emi_estimate   = loan_amount / loan_term
    emi_ratio      = emi_estimate / monthly_income
    default_prob = (
        - 0.004  * credit_score
        - 0.00000015 * annual_income
        + 0.025  * dti
        - 0.018  * emp_years
        + 0.12   * existing_loans
        + 0.8    * loan_to_income
        + 0.9    * emi_ratio
        + 0.003  * age
        + np.random.normal(0, 0.1, n)
    )
    prob_norm = 1 / (1 + np.exp(-default_prob))
    default   = (prob_norm > 0.45).astype(int)
    return pd.DataFrame({
        "credit_score": credit_score, "annual_income": annual_income.round(0),
        "dti": dti.round(2), "emp_years": emp_years.round(1),
        "existing_loans": existing_loans, "loan_amount": loan_amount.round(0),
        "loan_term": loan_term, "age": age,
        "loan_to_income": loan_to_income.round(4), "emi_ratio": emi_ratio.round(4),
        "default": default,
    })

print("Generating dataset...")
df = generate_dataset(5000)
df.to_csv("data/credit_data.csv", index=False)
print(f"Dataset: {df.shape}, Default rate: {df['default'].mean():.2%}")

FEATURES = ["credit_score","annual_income","dti","emp_years",
            "existing_loans","loan_amount","loan_term","age",
            "loan_to_income","emi_ratio"]

X, y = df[FEATURES], df["default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("Training XGBoost...")
model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)

y_pred  = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]
print(classification_report(y_test, y_pred, target_names=["Approved","Rejected"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

print("Building SHAP explainer...")
explainer = shap.TreeExplainer(model)

joblib.dump(model,    "models/credit_model.pkl")
joblib.dump(scaler,   "models/scaler.pkl")
joblib.dump(explainer,"models/shap_explainer.pkl")
joblib.dump(FEATURES, "models/features.pkl")
print("All done! Run: python backend/app.py")
