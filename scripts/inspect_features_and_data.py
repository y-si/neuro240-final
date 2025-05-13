import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
DF_PATH = 'data/test_samples.csv'
df = pd.read_csv(DF_PATH)

print("\n=== Feature Columns ===")
print(df.columns.tolist())

print("\n=== Class Distribution ===")
print(df['label'].value_counts())

print("\n=== Sample Rows by Class ===")
for label in df['label'].unique():
    print(f"\nClass: {label}")
    print(df[df['label'] == label].sample(2, random_state=42)[['text', 'label']])

# If there are numeric features, show their distribution by class
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    for col in numeric_cols:
        plt.figure(figsize=(6, 3))
        sns.histplot(data=df, x=col, hue='label', kde=True, element='step', stat='density')
        plt.title(f'Distribution of {col} by class')
        plt.tight_layout()
        plt.savefig(f'results/feature_dist_{col}.png')
        plt.close()
        print(f"Saved distribution plot for {col} to results/feature_dist_{col}.png")

print("\n=== Checking for Constant Features ===")
for col in numeric_cols:
    if df[col].nunique() == 1:
        print(f"Feature {col} is constant across all samples.")

print("\n=== Checking for Unique Features by Class ===")
for col in numeric_cols:
    nunique_per_class = df.groupby('label')[col].nunique()
    if (nunique_per_class == 1).all() and (df[col].nunique() > 1):
        print(f"Feature {col} is unique per class (potential leakage)")
