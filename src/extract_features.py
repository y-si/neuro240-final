import os
import pandas as pd
import numpy as np
import torch
import traceback
from data_loader import load_dataset
from feature_extractor import (
    extract_stylometric_features,
    extract_roberta_embeddings,
    extract_xlnet_embeddings,
    combine_features
)

def main():
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} samples")
    
    # Extract stylometric features
    print("Extracting stylometric features...")
    style_features = extract_stylometric_features(df)
    print(f"Extracted {style_features.shape[1]} stylometric features")
    
    # Save stylometric features immediately
    style_features.to_csv("data/stylometric_features.csv", index=False)
    print("Saved stylometric features to data/stylometric_features.csv")
    
    # Sample a subset of data for model embeddings (to save time and memory)
    sample_size = min(30, len(df))  # Limit to 30 samples for demonstration
    sample_df = df.sample(sample_size, random_state=42)
    sample_texts = sample_df["text"].tolist()
    
    # Initialize empty dictionaries for embeddings
    model_embeddings = {}
    
    # Extract RoBERTa embeddings
    try:
        print("Extracting RoBERTa embeddings...")
        roberta_embeddings = extract_roberta_embeddings(sample_texts, pooling='cls')
        print(f"RoBERTa embedding shape: {roberta_embeddings.shape}")
        model_embeddings['roberta'] = roberta_embeddings
        
        # Save RoBERTa embeddings immediately
        np.save("data/roberta_embeddings.npy", roberta_embeddings)
        print("Saved RoBERTa embeddings to data/roberta_embeddings.npy")
    except Exception as e:
        print(f"Error extracting RoBERTa embeddings: {e}")
        traceback.print_exc()
    
    # Extract XLNet embeddings
    try:
        print("Extracting XLNet embeddings...")
        xlnet_embeddings = extract_xlnet_embeddings(sample_texts, pooling='last')
        print(f"XLNet embedding shape: {xlnet_embeddings.shape}")
        model_embeddings['xlnet'] = xlnet_embeddings
        
        # Save XLNet embeddings immediately
        np.save("data/xlnet_embeddings.npy", xlnet_embeddings)
        print("Saved XLNet embeddings to data/xlnet_embeddings.npy")
    except Exception as e:
        print(f"Error extracting XLNet embeddings: {e}")
        traceback.print_exc()
    
    # Combine features if we have any embeddings
    if model_embeddings:
        try:
            print("Combining features...")
            combined_features = combine_features(sample_df, model_embeddings)
            print(f"Combined features shape: {combined_features.shape}")
            
            # Save combined features
            combined_features.to_csv("data/combined_features_sample.csv", index=False)
            print("Saved combined features for sample to data/combined_features_sample.csv")
        except Exception as e:
            print(f"Error combining features: {e}")
            traceback.print_exc()
    
    # Show example of the extracted features
    print("\nExample of extracted stylometric features:")
    print(style_features.head())
    
    print("\nFeature names:")
    print(", ".join(style_features.columns))
    
if __name__ == "__main__":
    main() 