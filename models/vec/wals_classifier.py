# models/vec/wals_classifier.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score, confusion_matrix

from models.vec.wals_features import get_or_compute_features

CLASS_NAMES = ["Worker", "Manager", "CEO"]

def load_dataset_split(seeds, data_dir="data", cache_dir="data/cache"):
    """
    Loads features and labels for a set of seeds, concatenating them.
    """
    all_features = []
    all_labels = []
    
    for seed in seeds:
        # Load computed/cached features
        feat, lbl = get_or_compute_features(data_dir, seed, cache_dir=cache_dir)
        all_features.append(feat)
        all_labels.append(lbl)
        
    return np.vstack(all_features), np.concatenate(all_labels)

def train_and_evaluate_classifier(
    train_seeds, 
    val_seeds, 
    test_seeds, 
    data_dir="data", 
    cache_dir="data/cache",
    model_type="extra_trees"
):
    """
    Trains a classifier using multi-scale physical features on training seeds,
    tunes/evaluates on validation seeds, and reports final test performance on unseen seeds.
    """
    print(f"\n==========================================")
    print(f"Loading datasets for {model_type.upper()} classifier...")
    print(f"==========================================")
    
    print("Loading Training set...")
    X_train, y_train = load_dataset_split(train_seeds, data_dir, cache_dir)
    print(f"  Training shape: {X_train.shape}, labels count: W={np.sum(y_train==0)}, M={np.sum(y_train==1)}, C={np.sum(y_train==2)}")
    
    print("Loading Validation set...")
    X_val, y_val = load_dataset_split(val_seeds, data_dir, cache_dir)
    print(f"  Validation shape: {X_val.shape}")
    
    print("Loading Test set...")
    X_test, y_test = load_dataset_split(test_seeds, data_dir, cache_dir)
    print(f"  Test shape: {X_test.shape}")
    
    # 1. Feature scaling (Robust vs Standard)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Model initialization with balanced class weights
    if model_type == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "svc":
        # Using linear or rbf kernel SVC. 
        # SVM is slower, so we can sub-sample training data if needed, or use linear.
        # Let's use LinearSVC or a small SVC.
        # SVC(class_weight='balanced') is highly robust.
        # Since we have 20,200 particles * 70 sims = 1.4 million training samples,
        # training a non-linear SVC will take forever! Let's use a linear SVC (LinearSVC) 
        # or extra_trees/random_forest which scale beautifully to large datasets.
        # Let's fallback to LinearSVC.
        from sklearn.svm import LinearSVC
        clf = LinearSVC(
            class_weight="balanced",
            random_state=42,
            dual=False,
            max_iter=5000
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    print(f"\nFitting {model_type} model on training set...")
    clf.fit(X_train_scaled, y_train)
    
    print("\n--- Training Set Evaluation ---")
    train_preds = clf.predict(X_train_scaled)
    print(classification_report(y_train, train_preds, target_names=CLASS_NAMES, digits=4))
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_train, train_preds):.4f}")
    
    print("\n--- Validation Set Evaluation ---")
    val_preds = clf.predict(X_val_scaled)
    print(classification_report(y_val, val_preds, target_names=CLASS_NAMES, digits=4))
    val_bal_acc = balanced_accuracy_score(y_val, val_preds)
    print(f"Balanced Accuracy: {val_bal_acc:.4f}")
    
    print("\n--- Test Set Evaluation (Unseen Seeds) ---")
    test_preds = clf.predict(X_test_scaled)
    print(classification_report(y_test, test_preds, target_names=CLASS_NAMES, digits=4))
    test_bal_acc = balanced_accuracy_score(y_test, test_preds)
    print(f"Balanced Accuracy: {test_bal_acc:.4f}")
    
    f1s = f1_score(y_test, test_preds, average=None)
    
    return {
        "model": clf,
        "scaler": scaler,
        "train_bal_acc": balanced_accuracy_score(y_train, train_preds),
        "val_bal_acc": val_bal_acc,
        "test_bal_acc": test_bal_acc,
        "test_f1_worker": f1s[0],
        "test_f1_manager": f1s[1],
        "test_f1_ceo": f1s[2],
        "y_test": y_test,
        "test_preds": test_preds
    }
