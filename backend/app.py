import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")
CORS(app)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")

try:
    model     = joblib.load(os.path.join(MODEL_DIR, "credit_model.pkl"))
    scaler    = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.pkl"))
    FEATURES  = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    print("Model loaded.")
except:
    print("Run train_model.py first.")
    model = scaler = explainer = FEATURES = None

LABELS = {
    "credit_score":"Credit Score","annual_income":"Annual Income","dti":"Debt-to-Income Ratio",
    "emp_years":"Employment Duration","existing_loans":"Existing Loans","loan_amount":"Loan Amount",
    "loan_term":"Loan Term","age":"Age","loan_to_income":"Loan-to-Income Ratio","emi_ratio":"EMI-to-Income Ratio"
}

def get_recommendation(decision, prob, dti, credit_score):
    if decision == "Approved":
        rate = "8.5%" if prob < 0.15 else "11.0%"
        return f"Eligible for standard interest rate loan ({rate} p.a.). Strong credit profile with low default risk. Recommended loan term: up to 5 years."
    elif decision == "Under Review":
        return "Borderline profile. Consider reducing existing debt or providing additional income proof before reapplying."
    else:
        tips = []
        if credit_score < 650: tips.append("improve credit score (target 700+)")
        if dti > 40: tips.append("reduce debt-to-income ratio below 35%")
        tips.append("clear existing loans")
        return "Application rejected. To improve chances: " + ", ".join(tips) + "."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503
    d = request.get_json(force=True)
    try:
        credit_score   = float(d.get("credit_score", 650))
        annual_income  = float(d.get("annual_income", 500000))
        dti            = float(d.get("dti", 30))
        emp_years      = float(d.get("emp_years", 3))
        existing_loans = int(d.get("existing_loans", 1))
        loan_amount    = float(d.get("loan_amount", 300000))
        loan_term      = int(d.get("loan_term", 36))
        age            = int(d.get("age", 35))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    loan_to_income = loan_amount / max(annual_income, 1)
    emi_ratio      = (loan_amount / max(loan_term, 1)) / max(annual_income / 12, 1)

    row = pd.DataFrame([{
        "credit_score": credit_score, "annual_income": annual_income, "dti": dti,
        "emp_years": emp_years, "existing_loans": existing_loans, "loan_amount": loan_amount,
        "loan_term": loan_term, "age": age, "loan_to_income": loan_to_income, "emi_ratio": emi_ratio
    }])[FEATURES]

    row_scaled   = scaler.transform(row)
    prob_default = float(model.predict_proba(row_scaled)[0][1])
    risk_score   = int(round(850 - prob_default * 550))

    if prob_default < 0.35:   decision = "Approved"
    elif prob_default < 0.55: decision = "Under Review"
    else:                      decision = "Rejected"

    confidence = round(float(np.clip(1 - abs(prob_default - 0.35) / 0.65, 0, 1)) * 100, 1)

    bands = [(0.15,"Low","Very Low Risk"),(0.30,"Low-Medium","Low Risk"),
             (0.50,"Medium","Medium Risk"),(0.70,"High","High Risk"),(1.0,"Very High","Very High Risk")]
    risk_cat, risk_band = next((c,b) for t,c,b in bands if prob_default < t)

    shap_values = explainer.shap_values(row_scaled)
    sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    shap_list = sorted([{
        "feature": f, "label": LABELS.get(f, f),
        "shap_value": round(float(v), 4),
        "feature_value": round(float(row[f].iloc[0]), 4),
        "direction": "positive" if v > 0 else "negative"
    } for f, v in zip(FEATURES, sv)], key=lambda x: abs(x["shap_value"]), reverse=True)

    pos_factors = [s["label"] for s in shap_list if s["shap_value"] < 0][:4]
    neg_factors = [s["label"] for s in shap_list if s["shap_value"] > 0][:4]

    return jsonify({
        "decision": decision, "probability": round(prob_default * 100, 1),
        "risk_score": risk_score, "confidence": confidence,
        "risk_category": risk_cat, "risk_band": risk_band,
        "positive_factors": pos_factors, "negative_factors": neg_factors,
        "top_features": [s["label"] for s in shap_list[:3]],
        "recommendation": get_recommendation(decision, prob_default, dti, credit_score),
        "shap_values": shap_list
    })

@app.route("/feature-importance")
def feature_importance():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 503
    fi = model.feature_importances_
    total = fi.sum()
    result = sorted([{"feature": f, "label": LABELS.get(f, f), "importance": round(float(s/total)*100, 2)}
                     for f, s in zip(FEATURES, fi)], key=lambda x: x["importance"], reverse=True)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
