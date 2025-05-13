import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from joblib import load

# Load features and labels
X = pd.read_csv('data/combined_features.csv')
y = pd.read_csv('data/labels.csv').values.ravel()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'RandomForest': RandomForestClassifier(n_estimators=500, random_state=42),
    'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
}

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    print(f'{name} CV accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
