import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder


def train_and_save():
    # Load data
    df = pd.read_csv("data/credit_risk_dataset.csv")

    # Clean outliers
    df = df[df['person_age'] < 100]
    df = df[df['person_emp_length'] < 60]

    # Fill missing values
    df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
    df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)

    # Encode categorical features
    cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        oob_score=True,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Metrics
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print(f"OOB Score: {rf.oob_score_:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=['Approved', 'Default']))

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/rf_model.pkl")
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

    print("✅ Model saved!")
    return rf


if __name__ == "__main__":
    train_and_save()