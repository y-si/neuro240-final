import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load combined features and labels
X = pd.read_csv('data/combined_features.csv')
y = pd.read_csv('data/labels.csv')['label']

print("\n=== Feature Columns ===")
print(X.columns.tolist())

print("\n=== Checking for Constant Features ===")
for col in X.columns:
    if X[col].nunique() == 1:
        print(f"Feature {col} is constant across all samples.")

print("\n=== Checking for Unique Features by Class ===")
for col in X.columns:
    nunique_per_class = X.groupby(y)[col].nunique()
    if (nunique_per_class == 1).all() and (X[col].nunique() > 1):
        print(f"Feature {col} is unique per class (potential leakage)")

print("\n=== Checking if any feature perfectly separates classes ===")
for col in X.columns:
    if X[col].nunique() == len(X):
        print(f"Feature {col} is unique for every sample (likely an index or ID, should be dropped)")

print("\n=== Sample Feature Values by Class ===")
for label in y.unique():
    print(f"\nClass: {label}")
    print(X[y == label].head(2))

# Plot distributions for first few features
for col in X.columns[:5]:
    plt.figure(figsize=(6, 3))
    sns.histplot(data=X, x=col, hue=y, kde=True, element='step', stat='density')
    plt.title(f'Distribution of {col} by class')
    plt.tight_layout()
    plt.savefig(f'results/feature_dist_combined_{col}.png')
    plt.close()
    print(f"Saved distribution plot for {col} to results/feature_dist_combined_{col}.png")
