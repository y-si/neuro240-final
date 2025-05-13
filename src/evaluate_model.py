import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from feature_extractor import extract_stylometric_features, extract_roberta_embeddings

def load_model(model_path):
    """Load a trained model from disk"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    return joblib.load(model_path)

def evaluate_model(model, X, y, output_dir="results/evaluation"):
    """Evaluate a model on given data and save results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X)

    # Attempt to decode predictions if they are integers and true labels are strings
    if y.dtype == object and isinstance(y_pred[0], (int, np.integer)):
        # Try to load the label encoder
        encoder_path = 'models/combined/label_encoder.joblib'
        if os.path.exists(encoder_path):
            le = joblib.load(encoder_path)
            y_pred = le.inverse_transform(y_pred)
        else:
            # Fallback: try to infer mapping from training labels
            unique_y = np.unique(y)
            if set(y_pred).issubset(set(range(len(unique_y)))):
                y_pred = [unique_y[i] for i in y_pred]
            else:
                y_pred = y_pred.astype(str)
    elif y.dtype in [int, np.int32, np.int64] and isinstance(y_pred[0], str):
        y_pred = y_pred.astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    report = classification_report(y, y_pred)
    print("Classification Report:")
    print(report)
    
    # Save report to file
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    
    # Create confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    
    # Save predictions
    results_df = pd.DataFrame({
        "true_label": y,
        "predicted_label": y_pred,
        "correct": y == y_pred
    })
    results_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    
    return accuracy, report, results_df

def main():
    """Main function to evaluate a trained model on new data"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a trained model on new data")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--data", type=str, required=True, help="Path to the data file")
    parser.add_argument("--output", type=str, default="results/evaluation", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load new test split (sanity-checked, no leakage)
    df = pd.read_csv('data/test_split.csv')
    print(f"Loading data from data/test_split.csv (sanity checked)")
    
    # Check if the data has the required columns
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Data file must contain 'text' and 'label' columns")
    
    # Extract features - use only text and label columns to match training, exclude filename
    df_features = df[[col for col in df.columns if col in ['text', 'label']]].copy()
    
    print("Extracting features...")
    features = extract_stylometric_features(df_features)
    
    # === EXTRACT ROBERTA EMBEDDINGS AND APPLY PCA ===
    # Only if PCA model exists (i.e., model expects roberta_pca_* features)
    roberta_pca_path = 'models/combined/roberta_pca.joblib'
    if os.path.exists(roberta_pca_path):
        print("Extracting RoBERTa embeddings for test data...")
        texts = df_features['text'].tolist()
        roberta_embeddings = extract_roberta_embeddings(texts, pooling='cls')
        from sklearn.decomposition import PCA
        import joblib
        roberta_pca = joblib.load(roberta_pca_path)
        roberta_pca_features = roberta_pca.transform(roberta_embeddings)
        # Add to features DataFrame with correct column names
        pca_cols = [f'roberta_pca_{i}' for i in range(roberta_pca_features.shape[1])]
        roberta_pca_df = pd.DataFrame(roberta_pca_features, columns=pca_cols)
        features = pd.concat([features.reset_index(drop=True), roberta_pca_df.reset_index(drop=True)], axis=1)

    # Load feature names from training
    import numpy as np
    feature_names_path = 'models/combined/stacking_feature_names.npy'
    if os.path.exists(feature_names_path):
        feature_names = np.load(feature_names_path, allow_pickle=True)
    else:
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")

    # Ensure test features match training features
    for col in feature_names:
        if col not in features.columns:
            features[col] = 0
    # Reorder columns to match training
    features = features[list(feature_names)]
    
    # Ensure all feature columns are string type (for sklearn compatibility)
    features.columns = features.columns.astype(str)
    
    # Load model
    model = load_model(args.model)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, features, df["label"].values, output_dir=args.output)
    
    print(f"\nEvaluation completed. Results saved to {args.output}")

if __name__ == "__main__":
    main() 