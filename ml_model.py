import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def train_ml_model(df):
    features = ["amount", "month", "day", "day_of_week", "hour", "weekend"]
    model_df = df.dropna(subset=features + ["category"]).copy()
    X = model_df[features]
    y = model_df["category"].astype(str)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    counts = pd.Series(y_encoded).value_counts()
    stratify = y_encoded if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=stratify
    )
    model = RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    labels = sorted(set(y_test) | set(y_pred))
    return {
        "model": model, "encoder": encoder, "features": features,
        "X_test": X_test, "y_test": y_test, "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "class_names": encoder.inverse_transform(labels),
        "classification_report": classification_report(
            y_test, y_pred, labels=labels,
            target_names=encoder.inverse_transform(labels), zero_division=0
        )
    }
