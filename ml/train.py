import joblib
import os
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

def train_and_save():
    # Загрузка данных
    wine = load_wine()
    X, y = wine.data, wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Обучение
    rf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    rf.fit(X_train, y_train)

    # Метрики
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "oob_score": rf.oob_score_,
        "roc_auc": roc_auc_score(y_test_bin, y_proba, multi_class='ovr'),
        "report": classification_report(y_test, y_pred, target_names=wine.target_names)
    }

    # Сохранение
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/rf_model.pkl")
    joblib.dump(wine.feature_names.tolist(), "models/feature_names.pkl")
    joblib.dump(wine.target_names.tolist(), "models/target_names.pkl")

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"OOB Score: {metrics['oob_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(metrics['report'])

    return rf, metrics

if __name__ == "__main__":
    train_and_save()