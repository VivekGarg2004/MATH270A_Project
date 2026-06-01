# run_hybrid_model_save.py
# Same pipeline as run_hybrid_model.py; additionally persists fitted RF, scaler, and
# threshold multipliers under weights/<YYYY-MM-DD>/<HH-MM-SS>/ for inference.
import sys
import os
import json
from datetime import datetime

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score

from configs.config import cfg
from models.vec.wals_features import get_or_compute_features
from run_wals_model import extract_baseline_rf_features_for_seed

CLASS_NAMES = ["Worker", "Manager", "CEO"]

def load_hybrid_dataset(seeds, data_dir="data", cache_dir="data/cache"):
    """
    Loads both physical multiscale features and kinematic baseline features for a set of seeds,
    and concatenates them into a hybrid feature matrix.
    """
    all_features = []
    all_labels = []
    
    for seed in seeds:
        # 1. Load physical multiscale features (21 features)
        phys_feat, lbl = get_or_compute_features(data_dir, seed, cache_dir=cache_dir)
        
        # 2. Load kinematic baseline features (14 features)
        kin_feat, _ = extract_baseline_rf_features_for_seed(
            seed, max_step=200, data_dir=data_dir
        )
        
        # 3. Create physical ratio features (each physical feature divided by its simulation median)
        # This provides scale-invariance to the physical coupling strengths
        medians = np.median(phys_feat, axis=0) + 1e-10
        phys_ratio_feat = phys_feat / medians
        
        # Stack them all: 21 (physical) + 14 (kinematic) + 21 (physical ratios) = 56 features
        hybrid_feat = np.hstack([phys_feat, kin_feat, phys_ratio_feat])
        
        all_features.append(hybrid_feat)
        all_labels.append(lbl)
        
    return np.vstack(all_features), np.concatenate(all_labels)

def optimize_thresholds(probs, y_true):
    """
    Optimizes class probability multipliers on the validation set to maximize the macro F1-score.
    This corrects the precision/recall imbalance caused by the extreme class ratio.
    """
    print("\nOptimizing decision thresholds on validation set...")
    best_score = 0
    best_multipliers = np.array([1.0, 1.0, 1.0])
    
    # Worker multiplier is fixed at 1.0
    # Grid search Manager and CEO multipliers to find the optimal decision boundary scaling
    m_range = np.linspace(0.2, 4.0, 20)
    c_range = np.linspace(0.5, 10.0, 39)
    
    for m_mult in m_range:
        for c_mult in c_range:
            multipliers = np.array([1.0, m_mult, c_mult])
            preds = np.argmax(probs * multipliers, axis=1)
            score = f1_score(y_true, preds, average='macro')
            if score > best_score:
                best_score = score
                best_multipliers = multipliers
                
    print(f"  Optimal class multipliers found: Worker=1.0, Manager={best_multipliers[1]:.2f}, CEO={best_multipliers[2]:.2f}")
    print(f"  Best Validation Macro F1: {best_score:.4f}")
    return best_multipliers

def save_inference_artifacts(out_dir, clf, scaler, multipliers, train_seeds, val_seeds, test_seeds):
    """Persist hybrid RF, scaler, and class multipliers for reload + inference."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, out_dir / "hybrid_rf.joblib")
    joblib.dump(scaler, out_dir / "hybrid_robust_scaler.joblib")
    np.save(out_dir / "hybrid_class_multipliers.npy", multipliers)

    import sklearn
    meta = {
        "class_names": CLASS_NAMES,
        "n_features_in": int(clf.n_features_in_),
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "test_seeds": test_seeds,
        "sklearn_version": sklearn.__version__,
        "random_forest": {
            "n_estimators": int(clf.n_estimators),
            "max_depth": clf.max_depth,
            "class_weight": clf.class_weight,
            "random_state": clf.random_state,
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved inference artifacts to: {out_dir}")
    print("  - hybrid_rf.joblib")
    print("  - hybrid_robust_scaler.joblib")
    print("  - hybrid_class_multipliers.npy")
    print("  - manifest.json")

def main():
    train_seeds = list(cfg.vec.TRAIN_SEEDS)
    val_seeds = list(cfg.vec.VAL_SEEDS)
    test_seeds = list(cfg.vec.TEST_SEEDS)
    
    cache_dir = "data/cache"

    now = datetime.now()
    date_part = now.strftime("%Y-%m-%d")
    time_part = now.strftime("%H-%M-%S")
    weights_run_dir = Path(project_root) / "weights" / date_part / time_part
    
    print("==========================================")
    print("Loading datasets for HYBRID physical-kinematic model...")
    print("==========================================")
    
    print("Loading Hybrid Train Set...")
    X_train, y_train = load_hybrid_dataset(train_seeds, cache_dir=cache_dir)
    print(f"  Train shape: {X_train.shape}")
    
    print("Loading Hybrid Validation Set...")
    X_val, y_val = load_hybrid_dataset(val_seeds, cache_dir=cache_dir)
    print(f"  Validation shape: {X_val.shape}")
    
    print("Loading Hybrid Test Set...")
    X_test, y_test = load_hybrid_dataset(test_seeds, cache_dir=cache_dir)
    print(f"  Test shape: {X_test.shape}")
    
    # 1. Standardize features robustly
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Train Hybrid Random Forest with balanced class weights
    print("\nTraining Hybrid Random Forest Classifier (Kinematic + Physical + Ratios)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)
    
    # 3. Evaluate Hybrid RF with default thresholds on test set
    print("\n--- Hybrid RF Test Evaluation (Default Thresholds) ---")
    test_preds_default = clf.predict(X_test_scaled)
    print(classification_report(y_test, test_preds_default, target_names=CLASS_NAMES, digits=4))
    
    # 4. Tune decision thresholds on the validation set
    val_probs = clf.predict_proba(X_val_scaled)
    multipliers = optimize_thresholds(val_probs, y_val)
    
    # 5. Evaluate Hybrid RF with optimized thresholds on test set
    print("\n--- Hybrid RF Test Evaluation (Optimized Thresholds) ---")
    test_probs = clf.predict_proba(X_test_scaled)
    test_preds_opt = np.argmax(test_probs * multipliers, axis=1)
    print(classification_report(y_test, test_preds_opt, target_names=CLASS_NAMES, digits=4))

    save_inference_artifacts(
        weights_run_dir, clf, scaler, multipliers, train_seeds, val_seeds, test_seeds
    )
    
    # Compute metrics for comparison
    bal_acc_default = balanced_accuracy_score(y_test, test_preds_default)
    f1s_default = f1_score(y_test, test_preds_default, average=None)
    
    bal_acc_opt = balanced_accuracy_score(y_test, test_preds_opt)
    f1s_opt = f1_score(y_test, test_preds_opt, average=None)
    
    # Compare with baseline RF from run_wals_model (which we run here for direct reference)
    from run_wals_model import run_baseline_rf
    baseline_results = run_baseline_rf(train_seeds, test_seeds)
    
    # Print side-by-side comparison table
    print("\n" + "="*95)
    print(f"{'Final Scientific Comparison Table':^95}")
    print("="*95)
    print(f"{'Model':40s} | {'Bal Acc':8s} | {'Worker F1':9s} | {'Manager F1':10s} | {'CEO F1':8s}")
    print("-"*95)
    print(f"{'Baseline RF (Kinematic Stats Only)':40s} | {baseline_results['bal_acc']:.4f}  | {baseline_results['f1_worker']:.4f}   | {baseline_results['f1_manager']:.4f}    | {baseline_results['f1_ceo']:.4f}")
    print(f"{'Hybrid RF (Default Thresholds)':40s} | {bal_acc_default:.4f}  | {f1s_default[0]:.4f}   | {f1s_default[1]:.4f}    | {f1s_default[2]:.4f}")
    print(f"{'Hybrid RF + Optimized Thresholds (Ours)':40s} | {bal_acc_opt:.4f}  | {f1s_opt[0]:.4f}   | {f1s_opt[1]:.4f}    | {f1s_opt[2]:.4f}")
    print("="*95)
    
    # Generate final performance comparison plot
    fig, ax = plt.subplots(figsize=(11, 6))
    
    classes = ["Worker", "Manager", "CEO"]
    x = np.arange(len(classes))
    width = 0.25
    
    baseline_f1s = [baseline_results['f1_worker'], baseline_results['f1_manager'], baseline_results['f1_ceo']]
    hybrid_default_f1s = [f1s_default[0], f1s_default[1], f1s_default[2]]
    hybrid_opt_f1s = [f1s_opt[0], f1s_opt[1], f1s_opt[2]]
    
    rects1 = ax.bar(x - width, baseline_f1s, width, label='Baseline RF (Kinematic Stats)', color='#7f7f7f', alpha=0.8)
    rects2 = ax.bar(x, hybrid_default_f1s, width, label='Hybrid RF (Default Boundaries)', color='#4e79a7', alpha=0.9)
    rects3 = ax.bar(x + width, hybrid_opt_f1s, width, label='Hybrid RF + Threshold Tuning (Ours)', color='#e15759', alpha=0.9)
    
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Generalization F1-Scores on Unseen Test Seeds (85-99)\nScientific Comparison of Physical Hybrid Features and Boundary Calibration', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plot_path = "analysis/final_hybrid_vs_baseline_comparison.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nSaved scientific comparison plot to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
