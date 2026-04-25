import joblib
import numpy as np

def load_model():
    rf = joblib.load("models/rf_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    target_names = joblib.load("models/target_names.pkl")
    return rf, feature_names, target_names

def predict(input_values: list):
    rf, feature_names, target_names = load_model()

    X = np.array(input_values).reshape(1, -1)
    pred_class = rf.predict(X)[0]
    pred_proba = rf.predict_proba(X)[0]

    votes = [int(round(p * rf.n_estimators)) for p in pred_proba]

    return {
        "class_id": int(pred_class),
        "class_name": target_names[pred_class],
        "confidence": round(float(pred_proba[pred_class]) * 100, 1),
        "votes": votes,
        "target_names": target_names
    }