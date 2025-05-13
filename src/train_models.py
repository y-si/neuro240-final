import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from data_loader import load_dataset
from feature_extractor import extract_stylometric_features, extract_roberta_embeddings, extract_xlnet_embeddings, extract_tfidf_features
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False
try:
    from lightgbm import LGBMClassifier
    lgbm_available = True
except ImportError:
    lgbm_available = False

def extract_and_save_features(df, output_dir="data"):
    """Extract and save all features"""
    print("Extracting stylometric features...")
    style_features = extract_stylometric_features(df)
    
    # Save stylometric features
    os.makedirs(output_dir, exist_ok=True)
    style_features_path = os.path.join(output_dir, "stylometric_features.csv")
    style_features.to_csv(style_features_path, index=False)
    print(f"Saved stylometric features to {style_features_path}")
    
    # Extract embeddings for ALL texts for RoBERTa (not just a sample)
    all_texts = df["text"].tolist()
    try:
        print("Extracting RoBERTa embeddings for ALL texts (this may take a while)...")
        roberta_embeddings = extract_roberta_embeddings(all_texts, pooling='cls')
        roberta_path = os.path.join(output_dir, "roberta_embeddings.npy")
        np.save(roberta_path, roberta_embeddings)
        print(f"Saved RoBERTa embeddings to {roberta_path}")
    except Exception as e:
        print(f"Error extracting RoBERTa embeddings: {e}")
        roberta_embeddings = None
    
    # Optionally: Keep sample logic for XLNet to avoid slowdowns
    sample_size = min(100, len(df))  # Limit sample size for memory constraints
    sample_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
    sample_texts = sample_df["text"].tolist()
    try:
        print("Extracting XLNet embeddings for sample texts...")
        xlnet_embeddings = extract_xlnet_embeddings(sample_texts, pooling='last')
        xlnet_path = os.path.join(output_dir, "xlnet_embeddings.npy")
        np.save(xlnet_path, xlnet_embeddings)
        print(f"Saved XLNet embeddings to {xlnet_path}")
    except Exception as e:
        print(f"Error extracting XLNet embeddings: {e}")
        xlnet_embeddings = None
    
    return style_features, roberta_embeddings, xlnet_embeddings, sample_df

def reduce_features(X_train, X_test, method="pca", n_components=50, y_train=None):
    if method == "pca":
        pca = PCA(n_components=n_components, random_state=42)
        X_train_red = pca.fit_transform(X_train)
        X_test_red = pca.transform(X_test)
        return X_train_red, X_test_red, pca
    elif method == "selectkbest":
        if y_train is None:
            raise ValueError("y_train must be provided for SelectKBest feature selection.")
        selector = SelectKBest(f_classif, k=n_components)
        X_train_red = selector.fit_transform(X_train, y_train)
        X_test_red = selector.transform(X_test)
        return X_train_red, X_test_red, selector
    else:
        return X_train, X_test, None

def run_model_with_feature_sets(models, X_train, X_test, y_train, y_test, text_test, results_dir, model_dir, feature_sets):
    results = {}
    for fs_name, (Xtr, Xte) in feature_sets.items():
        print(f"\n=== Feature set: {fs_name} ===")
        for name, model in models.items():
            print(f"Training {name} with {fs_name} features...")
            try:
                model.fit(Xtr, y_train)
                y_pred = model.predict(Xte)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                results[(name, fs_name)] = {
                    "accuracy": accuracy,
                    "report": report,
                    "model": model
                }
                # Save misclassified samples
                misclassified_idx = np.where(y_pred != y_test)[0]
                if len(misclassified_idx) > 0:
                    try:
                        misclassified_texts = text_test.iloc[misclassified_idx].values
                    except Exception:
                        misclassified_texts = text_test[misclassified_idx]
                    misclassified_df = pd.DataFrame({
                        "text": misclassified_texts,
                        "true_label": y_test[misclassified_idx],
                        "predicted_label": y_pred[misclassified_idx]
                    })
                    misclassified_csv = os.path.join(results_dir, f"{name}_{fs_name}_misclassified.csv")
                    misclassified_df.to_csv(misclassified_csv, index=False)
                # Save model
                model_path = os.path.join(model_dir, f"{name.lower()}_{fs_name}.pkl")
                joblib.dump(model, model_path)
                print(f"{name} ({fs_name}) accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"Model {name} ({fs_name}) failed: {e}")
    return results

def train_and_evaluate_models(X, y, model_dir="models", results_dir="results"):
    """Train and evaluate multiple models, save results"""
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Export full feature matrix and labels for cross-validation
    # Save BEFORE train/test split
    os.makedirs('data', exist_ok=True)
    pd.DataFrame(X).to_csv('data/combined_features.csv', index=False)
    pd.DataFrame(y, columns=['label']).to_csv('data/labels.csv', index=False)
    
    # Split data using stratified split for more accurate metrics
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Save scaler for later use
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    
    # Prepare feature sets
    feature_sets = {}
    # 1. Stylometric only
    if hasattr(X_train, 'columns'):
        stylometric_cols = [c for c in X_train.columns if c.startswith('flesch') or c.startswith('gunning') or c.endswith('_count') or c.endswith('_ratio') or c.endswith('_entropy') or c.endswith('_avg') or c.endswith('_length') or c.endswith('_k') or c.endswith('_stdev') or c.endswith('_var') or c.endswith('_mean') or c.endswith('_type') or c.startswith('yules') or c.startswith('hapax')]
        X_train_stylo = X_train[stylometric_cols].values if stylometric_cols else X_train.values
        X_test_stylo = X_test[stylometric_cols].values if stylometric_cols else X_test.values
    else:
        X_train_stylo = X_train
        X_test_stylo = X_test
    feature_sets['stylometric'] = (X_train_stylo, X_test_stylo)
    # 2. Embeddings only
    if hasattr(X_train, 'columns'):
        emb_cols = [c for c in X_train.columns if c.startswith('sent_emb_') or c.startswith('roberta_emb_') or c.startswith('xlnet_emb_')]
        X_train_emb = X_train[emb_cols].values if emb_cols else X_train.values
        X_test_emb = X_test[emb_cols].values if emb_cols else X_test.values
    else:
        X_train_emb = X_train
        X_test_emb = X_test
    feature_sets['embeddings'] = (X_train_emb, X_test_emb)
    # 3. All features
    feature_sets['all'] = (X_train, X_test)
    # 4. PCA reduction (all)
    X_train_pca, X_test_pca, _ = reduce_features(X_train, X_test, method="pca", n_components=min(50, X_train.shape[1]))
    feature_sets['pca'] = (X_train_pca, X_test_pca)
    # 5. SelectKBest (all)
    X_train_kbest, X_test_kbest, _ = reduce_features(X_train, X_test, method="selectkbest", n_components=min(50, X_train.shape[1]), y_train=y_train)
    feature_sets['kbest'] = (X_train_kbest, X_test_kbest)

    # Define base models
    base_models = {}
    base_models["LogisticRegression"] = LogisticRegression(max_iter=1000, random_state=42)
    base_models["RandomForest"] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    base_models["NeuralNetwork"] = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    if xgb_available:
        base_models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0, use_label_encoder=False)
    if lgbm_available:
        base_models["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Ensemble estimators: only base models (no ensembles)
    ensemble_estimators = [(name, mdl) for name, mdl in base_models.items()]
    
    # Define all models including ensembles
    models = dict(base_models)
    models["Voting"] = VotingClassifier(ensemble_estimators, voting='soft', n_jobs=1)
    models["Stacking"] = StackingClassifier(ensemble_estimators, final_estimator=LogisticRegression(max_iter=1000, random_state=42), n_jobs=1)

    # Encode string labels as integers for XGBoost and LightGBM
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Use 3-fold cross-validation for speed
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = {}
    best_accuracy = 0
    best_model_name = None
    for name, model in models.items():
        print(f"\nTraining {name}...")
        y_train = y_encoded if name in ["XGBoost", "LightGBM"] else y
        if name == "NeuralNetwork":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Ensure numeric dtype for MLPClassifier
            X_scaled = np.asarray(X_scaled, dtype=np.float32)
            # Extra safety: check for NaNs or infs
            assert not np.isnan(X_scaled).any(), "NaNs in NeuralNetwork features!"
            assert np.isfinite(X_scaled).all(), "Infs in NeuralNetwork features!"
            X_to_use = X_scaled
            # Label encode for NeuralNetwork
            nn_label_encoder = LabelEncoder()
            y_nn = nn_label_encoder.fit_transform(y)
            # Hyperparameter search for NeuralNetwork only
            param_grid = {
                'hidden_layer_sizes': [(128,), (64, 32), (256, 128, 64)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam'],
                'max_iter': [500, 1000, 2000],
                'early_stopping': [True],
                'learning_rate_init': [0.001, 0.01]
            }
            grid = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)
            grid.fit(X_to_use, y_nn)
            best_model = grid.best_estimator_
            print(f"Best NeuralNetwork params: {grid.best_params_}")
            scores = cross_val_score(best_model, X_to_use, y_nn, cv=cv, scoring='accuracy', n_jobs=-1)
            model_to_save = best_model
        else:
            X_to_use = X
            try:
                if name in ["XGBoost", "LightGBM"]:
                    scores = cross_val_score(model, X_to_use, y_train, cv=cv, scoring='accuracy', n_jobs=1)
                else:
                    scores = cross_val_score(model, X_to_use, y_train, cv=cv, scoring='accuracy', n_jobs=1 if name in ["Voting", "Stacking"] else -1)
                print(f"Fold scores: {scores}")
                print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
                model_to_save = model
            except Exception as e:
                print(f"Model {name} failed: {e}")
                results[name] = {
                    "mean_accuracy": None,
                    "std_accuracy": None,
                    "fold_scores": None,
                    "status": f"failed: {e}"
                }
                continue
        print(f"Fold scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        results[name] = {
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std(),
            "fold_scores": scores.tolist(),
            "status": "success"
        }
        if scores.mean() > best_accuracy:
            best_accuracy = scores.mean()
            best_model_name = name
        model_to_save.fit(X_to_use, y_train if name != "NeuralNetwork" else y_nn)
        model_path = os.path.join(model_dir, f"{name.lower()}.pkl")
        joblib.dump(model_to_save, model_path)
        if name == "NeuralNetwork":
            scaler_path = os.path.join(model_dir, f"{name.lower()}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            # Save label encoder for NeuralNetwork
            le_path = os.path.join(model_dir, f"{name.lower()}_label_encoder.pkl")
            joblib.dump(nn_label_encoder, le_path)
    # Save results
    results_path = os.path.join(results_dir, "model_comparison.csv")
    pd.DataFrame([{**{"model": k}, **v} for k, v in results.items()]).to_csv(results_path, index=False)
    print(f"\nBest model: {best_model_name} (accuracy: {best_accuracy:.4f})")
    print(f"Results saved to {results_path}")

    # Save accuracy bar plot (only successful models)
    import matplotlib.pyplot as plt
    model_names = [k for k, v in results.items() if v["mean_accuracy"] is not None]
    accuracies = [results[m]["mean_accuracy"] for m in model_names]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(model_names, accuracies, color='skyblue')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracies')
    plt.ylim(0, 1)
    plt.tight_layout()
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{acc * 100:.2f}%", ha='center', va='bottom', fontsize=11)
    acc_plot_path = os.path.join(results_dir, "model_accuracy.png")
    plt.savefig(acc_plot_path)
    print(f"Saved model accuracy plot to {acc_plot_path}")
    return results

def run_nn_on_features(X_feat, y, feature_name):
    print(f"\n[DIAGNOSTIC] Running NeuralNetwork on {feature_name} features only...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    X_scaled = np.asarray(X_scaled, dtype=np.float32)
    assert not np.isnan(X_scaled).any(), f"NaNs in {feature_name} features for NN!"
    assert np.isfinite(X_scaled).all(), f"Infs in {feature_name} features for NN!"
    nn_label_encoder = LabelEncoder()
    y_nn = nn_label_encoder.fit_transform(y)
    nn = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=1000, random_state=42)
    scores = cross_val_score(nn, X_scaled, y_nn, cv=3, scoring='accuracy')
    print(f"[DIAGNOSTIC] {feature_name} NN fold scores: {scores}")
    print(f"[DIAGNOSTIC] {feature_name} NN mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores.mean(), scores.std()

def run_nn_hyperparam_search(X_feat, y, feature_name):
    print(f"\n[HYPERPARAM SEARCH] NeuralNetwork on {feature_name} features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    X_scaled = np.asarray(X_scaled, dtype=np.float32)
    assert not np.isnan(X_scaled).any(), f"NaNs in {feature_name} features for NN!"
    assert np.isfinite(X_scaled).all(), f"Infs in {feature_name} features for NN!"
    nn_label_encoder = LabelEncoder()
    y_nn = nn_label_encoder.fit_transform(y)
    
    param_grid = {
        'hidden_layer_sizes': [(64,), (128,), (256,), (128, 64), (256, 128), (128, 128), (256, 256)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
    }
    nn = MLPClassifier(max_iter=1000, random_state=42)
    grid = GridSearchCV(nn, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid.fit(X_scaled, y_nn)
    print(f"[HYPERPARAM SEARCH] Best params: {grid.best_params_}")
    print(f"[HYPERPARAM SEARCH] Best NN mean accuracy: {grid.best_score_:.4f}")
    return grid.best_params_, grid.best_score_

def train_and_evaluate_models_with_best_nn(X, y, model_dir="models", results_dir="results"):
    import copy
    from sklearn.neural_network import MLPClassifier
    # Use original function, but replace NeuralNetwork with best params
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    pd.DataFrame(X).to_csv('data/combined_features.csv', index=False)
    pd.DataFrame(y, columns=['label']).to_csv('data/labels.csv', index=False)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    base_models = {}
    base_models["LogisticRegression"] = LogisticRegression(max_iter=1000, random_state=42)
    base_models["RandomForest"] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # Use best NN params
    nn_params_copy = copy.deepcopy(best_nn_params)
    base_models["NeuralNetwork"] = MLPClassifier(max_iter=1000, random_state=42, **nn_params_copy)
    if xgb_available:
        base_models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0, use_label_encoder=False)
    if lgbm_available:
        base_models["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    ensemble_estimators = [(name, mdl) for name, mdl in base_models.items()]
    models = dict(base_models)
    models["Voting"] = VotingClassifier(ensemble_estimators, voting='soft', n_jobs=1)
    models["Stacking"] = StackingClassifier(ensemble_estimators, final_estimator=LogisticRegression(max_iter=1000, random_state=42), n_jobs=1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        y_train_cv = y_encoded if name in ["XGBoost", "LightGBM"] else y
        if name == "NeuralNetwork":
            scaler_nn = StandardScaler()
            X_scaled = scaler_nn.fit_transform(X)
            X_scaled = np.asarray(X_scaled, dtype=np.float32)
            assert not np.isnan(X_scaled).any(), "NaNs in NeuralNetwork features!"
            assert np.isfinite(X_scaled).all(), "Infs in NeuralNetwork features!"
            scores = cross_val_score(model, X_scaled, label_encoder.transform(y), cv=cv, scoring='accuracy')
            print(f"[NeuralNetwork] fold scores: {scores}")
            print(f"[NeuralNetwork] mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
            results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'fold_scores': scores,
                'status': 'success'
            }
            continue
        scores = cross_val_score(model, X, y_train_cv, cv=cv, scoring='accuracy')
        print(f"[{name}] fold scores: {scores}")
        print(f"[{name}] mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        results[name] = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'fold_scores': scores,
            'status': 'success'
        }
    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'model': k,
            'mean_accuracy': v['mean_accuracy'],
            'std_accuracy': v['std_accuracy'],
            'fold_scores': list(v['fold_scores']),
            'status': v['status']
        } for k, v in results.items()
    ])
    results_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
    print(f"Results saved to {os.path.join(results_dir, 'model_comparison.csv')}")
    # Plot
    plt.figure(figsize=(10, 5))
    sns.barplot(x='model', y='mean_accuracy', data=results_df)
    plt.title('Model Mean Accuracy')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(results_dir, "model_accuracy.png"))
    print(f"Saved model accuracy plot to {os.path.join(results_dir, 'model_accuracy.png')}")
    return results

def feature_selection_nn_search(X, y, feature_name, k_list=[50, 100, 200]):
    from sklearn.feature_selection import SelectKBest, f_classif
    results = []
    for k in k_list:
        print(f"\n[FEATURE SELECTION] Running SelectKBest (k={k}) + NN grid search...")
        selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
        X_kbest = selector.fit_transform(X, y)
        best_params, best_score = run_nn_hyperparam_search(X_kbest, y, f"{feature_name}_kbest{k}")
        results.append({
            'k': k,
            'best_params': best_params,
            'best_score': best_score
        })
        print(f"[RESULT] k={k}, best NN accuracy: {best_score:.4f}, best params: {best_params}")
    return results

def main():
    """Main function to run the entire pipeline"""
    print("Starting model selection and training pipeline...")
    
    # Load new train split (sanity-checked, no leakage)
    df = pd.read_csv('data/train_split.csv')
    print(f"Loaded {len(df)} training samples from data/train_split.csv (sanity checked)")
    
    # Extract features (exclude filename column)
    feature_df = df.drop(columns=[col for col in ['filename'] if col in df.columns])
    style_features, roberta_embeddings, xlnet_embeddings, sample_df = extract_and_save_features(feature_df)
    
    # Get labels
    labels = feature_df["label"].values
    
    # === DIAGNOSTIC: Run NeuralNetwork on individual feature sets ===
    # Prepare individual feature sets
    tfidf_df, _ = extract_tfidf_features(df, max_features=100)
    
    # === REDUCE ROBERTA EMBEDDINGS WITH PCA ===
    from sklearn.decomposition import PCA
    roberta_pca = PCA(n_components=20, random_state=42)
    roberta_pca_features = roberta_pca.fit_transform(roberta_embeddings)
    # Save the fitted PCA model, not the features array
    import joblib
    joblib.dump(roberta_pca, 'models/combined/roberta_pca.joblib')
    print("[FINAL] Saved RoBERTa PCA model to models/combined/roberta_pca.joblib")
    
    # Run diagnostics
    run_nn_on_features(style_features, labels, "stylometric")
    run_nn_on_features(tfidf_df, labels, "tfidf")
    run_nn_on_features(roberta_pca_features, labels, "roberta_pca")
    
    # === MAIN MODELING: Use only stylometric + RoBERTa PCA for all models ===
    combined_features = pd.concat([
        style_features.reset_index(drop=True),
        pd.DataFrame(roberta_pca_features, columns=[f"roberta_pca_{i}" for i in range(roberta_pca_features.shape[1])]).reset_index(drop=True)
    ], axis=1)
    combined_features = combined_features.fillna(0)
    print(f"[INFO] Using stylometric features + top 20 PCA components of RoBERTa embeddings (NO TF-IDF).")

    # === FEATURE SELECTION + NN GRID SEARCH ===
    fs_results = feature_selection_nn_search(combined_features, labels, "stylometric+roberta_pca", k_list=[50, 100, 200])
    # Pick best k and params
    best_fs = max(fs_results, key=lambda x: x['best_score'])
    print(f"\n[FINAL NN] Retraining NN with k={best_fs['k']} features and best params: {best_fs['best_params']}")
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k=min(best_fs['k'], combined_features.shape[1]))
    X_kbest = selector.fit_transform(combined_features, labels)
    from sklearn.neural_network import MLPClassifier
    nn_final = MLPClassifier(max_iter=1000, random_state=42, **best_fs['best_params'])
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    scores = cross_val_score(nn_final, X_kbest, y_enc, cv=3, scoring='accuracy')
    print(f"[FINAL NN] fold scores: {scores}")
    print(f"[FINAL NN] mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    # Save to CSV (append or update NeuralNetwork_FS row)
    results_csv = 'results/combined/model_comparison.csv'
    df = pd.read_csv(results_csv)
    fs_row = {
        'model': f'NeuralNetwork_FS{best_fs["k"]}',
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std(),
        'fold_scores': list(scores),
        'status': 'success'
    }
    # Remove any existing NeuralNetwork_FS rows
    df = df[~df['model'].str.startswith('NeuralNetwork_FS')]
    df = pd.concat([df, pd.DataFrame([fs_row])], ignore_index=True)
    df.to_csv(results_csv, index=False)
    print(f"[FINAL NN] Results saved to {results_csv}")

    # === DIAGNOSTIC: Run NN on stylometric + RoBERTa PCA only ===
    print("\n[DIAGNOSTIC] Running NeuralNetwork on stylometric + RoBERTa PCA features only...")
    run_nn_on_features(combined_features, labels, "stylometric+roberta_pca")
    
    # === SAVE BEST MODEL (STACKING) ===
    # Retrain StackingClassifier on all data and save to models/combined/stacking.pkl
    print("\n[FINAL] Retraining StackingClassifier on full combined_features and saving model...")
    from sklearn.ensemble import StackingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except ImportError:
        xgb_available = False
    try:
        from lightgbm import LGBMClassifier
        lgbm_available = True
    except ImportError:
        lgbm_available = False
    base_models = {}
    base_models["LogisticRegression"] = LogisticRegression(max_iter=1000, random_state=42)
    base_models["RandomForest"] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    if xgb_available:
        base_models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0, use_label_encoder=False)
    if lgbm_available:
        base_models["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    ensemble_estimators = [(name, mdl) for name, mdl in base_models.items()]
    stacking = StackingClassifier(ensemble_estimators, final_estimator=LogisticRegression(max_iter=1000, random_state=42), n_jobs=1)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    stacking.fit(combined_features, y_enc)
    import joblib, os
    os.makedirs('models/combined', exist_ok=True)
    joblib.dump(stacking, 'models/combined/stacking.pkl')
    print("[FINAL] StackingClassifier saved to models/combined/stacking.pkl")
    
    # === SAVE FEATURE NAMES USED FOR TRAINING ===
    import numpy as np
    feature_names = combined_features.columns.to_list()
    np.save('models/combined/stacking_feature_names.npy', feature_names)
    print("[FINAL] Feature names saved to models/combined/stacking_feature_names.npy")
    
    # =====================
    # FEATURE ENGINEERING
    # =====================
    # Use stylometric features + RoBERTa PCA (no TF-IDF)
    
if __name__ == "__main__":
    main() 