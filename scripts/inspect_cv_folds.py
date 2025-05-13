import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Load data
DF_PATH = 'data/test_samples.csv'
df = pd.read_csv(DF_PATH)
X = df.drop(columns=['text', 'label', 'filename'], errors='ignore')
y = df['label']
le = LabelEncoder()
y_enc = le.fit_transform(y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

fold_accuracies = []
for i, (train_idx, test_idx) in enumerate(skf.split(X, y_enc)):
    model.fit(X.iloc[train_idx], y_enc[train_idx])
    acc = model.score(X.iloc[test_idx], y_enc[test_idx])
    print(f"Fold {i+1} accuracy: {acc:.6f}")
    fold_accuracies.append(acc)
print(f"\nAll fold accuracies: {fold_accuracies}")
print(f"Mean: {np.mean(fold_accuracies):.6f}, Std: {np.std(fold_accuracies):.6f}")
