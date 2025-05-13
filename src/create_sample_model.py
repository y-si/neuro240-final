import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from feature_extractor import extract_stylometric_features

def create_sample_model():
    """Create a sample Random Forest model using the test data"""
    # Load the dataset
    try:
        df = pd.read_csv("data/test_samples.csv")
        print(f"Loaded dataset with {len(df)} samples")
    except FileNotFoundError:
        print("Dataset not found. Please run create_dataset.py first.")
        return
    
    # Select only the necessary columns for feature extraction
    df_features = df[["text", "label"]].copy()
    
    # Extract features
    print("Extracting stylometric features...")
    features = extract_stylometric_features(df_features)
    
    # Get labels
    labels = df_features["label"].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Train a simple Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create directory if it doesn't exist
    os.makedirs("models/stylometric", exist_ok=True)
    
    # Save the model
    model_path = "models/stylometric/random_forest.pkl"
    joblib.dump(model, model_path)
    print(f"Sample model saved to {model_path}")
    
    # Basic evaluation
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.4f}")
    
    return model

if __name__ == "__main__":
    create_sample_model() 