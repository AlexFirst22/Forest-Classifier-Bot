import joblib
import numpy as np
import pandas as pd


def load_model():
    rf = joblib.load("models/rf_model.pkl")
    label_encoders = joblib.load("models/label_encoders.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return rf, label_encoders, feature_names


def predict(data: dict):
    rf, label_encoders, feature_names = load_model()

    df = pd.DataFrame([data])

    # Auto-calculate loan_percent_income
    df['loan_percent_income'] = df['loan_amnt'] / df['person_income']

    # Encode categorical features
    cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    for col in cat_cols:
        le = label_encoders[col]
        df[col] = le.transform(df[col])

    # Reorder columns
    df = df[feature_names]

    prob_default = rf.predict_proba(df)[0][1]
    prob_approved = rf.predict_proba(df)[0][0]
    prediction = int(rf.predict(df)[0])

    # Feature importance for explanation
    importances = rf.feature_importances_
    top_features = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    return {
        "prediction": prediction,
        "prob_approved": round(prob_approved * 100, 1),
        "prob_default": round(prob_default * 100, 1),
        "top_features": top_features
    }