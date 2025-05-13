import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def load_samples():
    # Initialize lists to store data
    texts = []
    labels = []
    
    # Define the models and their directories
    models = ['gpt4', 'claude', 'llama']
    
    # Process each model's samples
    for model in models:
        # Get all .txt files in the model's directory
        sample_files = glob.glob(os.path.join('data', 'raw', model, '*.txt'))
        
        # Read each file
        for file_path in sample_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    texts.append(text)
                    labels.append(model)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    return df

def analyze_samples(df):
    # Basic statistics
    print("\n=== Sample Counts per Model ===")
    print(df['label'].value_counts())
    
    # Calculate text lengths (in words)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    
    print("\n=== Average Text Length (words) ===")
    print(df.groupby('label')['word_count'].mean())
    
    # Create histogram of text lengths
    plt.figure(figsize=(12, 6))
    for model in df['label'].unique():
        model_data = df[df['label'] == model]['word_count']
        plt.hist(model_data, bins=30, alpha=0.5, label=model)
    
    plt.title('Distribution of Text Lengths by Model')
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('text_length_distribution.png')
    plt.close()

def main():
    # Load samples
    print("Loading samples...")
    df = load_samples()
    
    # Analyze samples
    print("\nAnalyzing samples...")
    analyze_samples(df)
    
    print("\nAnalysis complete! Check 'text_length_distribution.png' for the visualization.")

if __name__ == "__main__":
    main() 