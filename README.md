# CreditAI — Credit Risk Prediction System

A loan approval engine that predicts credit risk using XGBoost and explains decisions using SHAP, served as a web app via Flask.

## What it does
- Takes applicant details (credit score, income, loan amount, etc.)
- Predicts whether a loan should be Approved, Rejected, or sent for Review
- Explains the decision using SHAP values (which factors helped or hurt)
- Lets you simulate different applicant profiles live

## Tech Stack
- ML Model:XGBoost (gradient boosting)
- Explainability:SHAP
- Backend:Flask (Python)
- Frontend: HTML, CSS, JavaScript

## Setup & Run

1. Install dependencies
   pip install flask flask-cors pandas numpy scikit-learn shap xgboost joblib

2. Train the model (generates .pkl files)
   python train_model.py

3. Start the server
   python backend/app.py

4. Open your browser at http://localhost:5000



## How the Risk Score works
Risk Score = 850 − (default probability × 550)
Higher is better. A score near 850 means very low default risk.
