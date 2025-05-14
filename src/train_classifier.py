from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import pandas as pd
import numpy as np
from feature_extractor import extract_stylometric_features

def train_classifier(X, y, model_path="models/random_forest.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    # Save scaler
    joblib.dump(scaler, "models/random_forest_scaler.pkl")

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data/test_split.csv")
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Data file must contain 'text' and 'label' columns")
    df_features = df[[col for col in df.columns if col in ['text', 'label']]].copy()
    print("Extracting features...")
    features = extract_stylometric_features(df_features)
    feature_names = features.columns.values
    X = features.values
    y = df["label"].values
    train_classifier(X, y, model_path="models/random_forest.pkl")
    # Save feature names for evaluation
    os.makedirs("models/combined", exist_ok=True)
    np.save("models/combined/stacking_feature_names.npy", feature_names)
    print(f"Model and feature names saved. Total features: {len(feature_names)}")
