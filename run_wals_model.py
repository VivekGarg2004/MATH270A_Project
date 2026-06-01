# run_wals_model.py
import sys
import os
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from pathlib import Path
from typing import Optional, Union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score

from configs.config import cfg
from models.vec.wals_features import get_or_compute_features
from models.vec.wals_classifier import train_and_evaluate_classifier, load_dataset_split

# Parameters for Baseline RF Features (matching models/random_forest/model.py)
DT = 1e-3
N1, N2, N3 = 16000, 4000, 200
N_TOTAL = N1 + N2 + N3
DATA_DIR = Path("data")

def extract_baseline_rf_features_for_seed(
    seed,
    max_step=200,
    data_dir: Optional[Union[str, Path]] = None,
):
    """
    Loads raw simulation data and extracts the trajectory statistical features
    used in the baseline Random Forest classifier.

    data_dir defaults to DATA_DIR (typically ``data/``). Use e.g.
    ``data_distribution/data`` for distribution-generated trajectories.
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    w = np.load(root / f"opinion_workers_mpi_{seed}.npy").squeeze()[:max_step]
    m = np.load(root / f"opinion_managers_mpi_{seed}.npy").squeeze()[:max_step]
    c = np.load(root / f"opinion_ceos_mpi_{seed}.npy").squeeze()[:max_step]

    def feats(o):
        v = np.diff(o, axis=0) / DT
        a = np.diff(v, axis=0) / DT
        if a.shape[0] == 0:
            a = np.zeros_like(v)
        raw = np.stack([
            np.mean(np.abs(v), axis=0),
            np.std(v,          axis=0),
            np.mean(v,         axis=0),
            np.max(np.abs(v),  axis=0),
            np.mean(np.abs(a), axis=0),
            np.max(np.abs(a),  axis=0),
            np.std(a,          axis=0),
            np.sum(np.abs(v),  axis=0),
            np.std(o,          axis=0),
            np.max(o,          axis=0) - np.min(o, axis=0),
        ], axis=1)
        return raw

    X_raw = np.concatenate([feats(w), feats(m), feats(c)], axis=0)
    sim_median = np.median(X_raw, axis=0) + 1e-10
    ratio = np.stack([
        X_raw[:, 6] / sim_median[6],
        X_raw[:, 5] / sim_median[5],
        X_raw[:, 4] / sim_median[4],
        X_raw[:, 1] / sim_median[1],
    ], axis=1)

    X = np.concatenate([X_raw, ratio], axis=1)
    n1, n2, n3 = w.shape[1], m.shape[1], c.shape[1]
    y = np.concatenate(
        [
            np.zeros(n1, dtype=int),
            np.ones(n2, dtype=int),
            2 * np.ones(n3, dtype=int),
        ]
    )
    return X, y

def load_baseline_dataset(seeds, max_step=200):
    all_X, all_y = [], []
    for seed in seeds:
        X, y = extract_baseline_rf_features_for_seed(seed, max_step=max_step)
        all_X.append(X)
        all_y.append(y)
    return np.vstack(all_X), np.concatenate(all_y)

def run_baseline_rf(train_seeds, test_seeds, max_step=200):
    print("\n==========================================")
    print("Training Baseline RF (Trajectory Statistics)...")
    print("==========================================")
    
    print("Loading Baseline Train Set...")
    X_train, y_train = load_baseline_dataset(train_seeds, max_step=max_step)
    print("Loading Baseline Test Set...")
    X_test, y_test = load_baseline_dataset(test_seeds, max_step=max_step)
    
    clf = RandomForestClassifier(
        n_estimators=100,  # 100 for parity with our physical models
        max_depth=15,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    test_preds = clf.predict(X_test)
    print("\n--- Baseline RF Test Set Evaluation ---")
    print(classification_report(y_test, test_preds, target_names=["Worker", "Manager", "CEO"], digits=4))
    
    bal_acc = balanced_accuracy_score(y_test, test_preds)
    f1s = f1_score(y_test, test_preds, average=None)
    
    return {
        "bal_acc": bal_acc,
        "f1_worker": f1s[0],
        "f1_manager": f1s[1],
        "f1_ceo": f1s[2]
    }

def main():
    # Load configuration seeds
    train_seeds = list(cfg.vec.TRAIN_SEEDS)
    val_seeds = list(cfg.vec.VAL_SEEDS)
    test_seeds = list(cfg.vec.TEST_SEEDS)
    
    print(f"Project configuration loaded.")
    print(f"Train seeds:      {train_seeds[0]} to {train_seeds[-1]} ({len(train_seeds)} seeds)")
    print(f"Validation seeds: {val_seeds[0]} to {val_seeds[-1]} ({len(val_seeds)} seeds)")
    print(f"Test seeds:       {test_seeds[0]} to {test_seeds[-1]} ({len(test_seeds)} seeds)")
    
    cache_dir = "data/cache"
    
    # Pre-extract physical features with JIT compilation & caching
    print("\nPre-extracting / loading physical multiscale features (with caching)...", flush=True)
    # Loop over first few seeds to compile the Numba functions so they are fast
    for seed in train_seeds + val_seeds + test_seeds:
        get_or_compute_features("data", seed, cache_dir=cache_dir)
    print("All physical features loaded/computed successfully!", flush=True)

    # 1. Train and evaluate our Physical Extra Trees Classifier
    et_results = train_and_evaluate_classifier(
        train_seeds, val_seeds, test_seeds, 
        data_dir="data", cache_dir=cache_dir, model_type="extra_trees"
    )

    # 2. Train and evaluate our Physical Random Forest Classifier
    rf_results = train_and_evaluate_classifier(
        train_seeds, val_seeds, test_seeds, 
        data_dir="data", cache_dir=cache_dir, model_type="random_forest"
    )

    # 3. Train and evaluate the Baseline RF model (Trajectory Statistics)
    baseline_results = run_baseline_rf(train_seeds, test_seeds)

    # Print clean side-by-side performance summary
    print("\n" + "="*80)
    print(f"{'Classifier Model Comparison':^80}")
    print("="*80)
    print(f"{'Model':30s} | {'Bal Acc':8s} | {'Worker F1':9s} | {'Manager F1':10s} | {'CEO F1':8s}")
    print("-"*80)
    print(f"{'Baseline RF (Trajectory Stats)':30s} | {baseline_results['bal_acc']:.4f}  | {baseline_results['f1_worker']:.4f}   | {baseline_results['f1_manager']:.4f}    | {baseline_results['f1_ceo']:.4f}")
    print(f"{'Physical RF (Ours)':30s} | {rf_results['test_bal_acc']:.4f}  | {rf_results['test_f1_worker']:.4f}   | {rf_results['test_f1_manager']:.4f}    | {rf_results['test_f1_ceo']:.4f}")
    print(f"{'Physical Extra Trees (Ours)':30s} | {et_results['test_bal_acc']:.4f}  | {et_results['test_f1_worker']:.4f}   | {et_results['test_f1_manager']:.4f}    | {et_results['test_f1_ceo']:.4f}")
    print("="*80)

    # Generate beautiful performance comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = ["Worker", "Manager", "CEO"]
    x = np.arange(len(classes))
    width = 0.25
    
    baseline_f1s = [baseline_results['f1_worker'], baseline_results['f1_manager'], baseline_results['f1_ceo']]
    phys_rf_f1s = [rf_results['test_f1_worker'], rf_results['test_f1_manager'], rf_results['test_f1_ceo']]
    phys_et_f1s = [et_results['test_f1_worker'], et_results['test_f1_manager'], et_results['test_f1_ceo']]
    
    rects1 = ax.bar(x - width, baseline_f1s, width, label='Baseline RF (Trajectory Stats)', color='#7f7f7f', alpha=0.8)
    rects2 = ax.bar(x, phys_rf_f1s, width, label='Physical RF (Ours)', color='#f28e2b', alpha=0.9)
    rects3 = ax.bar(x + width, phys_et_f1s, width, label='Physical Extra Trees (Ours)', color='#e15759', alpha=0.9)
    
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Generalization F1-Scores on Unseen Test Seeds (85-99)\nSevere Class Imbalance (Workers: 16k, Managers: 4k, CEOs: 200)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plot_path = "analysis/physical_vs_baseline_comparison.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nSaved performance comparison plot to: {plot_path}")
    plt.close()

    # Generate a beautiful physical feature distribution plot
    # We will load a single test seed and plot the physical interaction density at scale R=1.0
    print("\nGenerating physical feature separation plot...")
    X_test_single, y_test_single = get_or_compute_features("data", seed=85, cache_dir=cache_dir)
    
    # Feature 6 in our multiscale features corresponds to r=1.0_count (after 6 RBF bases)
    # Let's find columns:
    # Scale 1.0: 6 RBF bases (cols 0-5), 1 neighbor count (col 6)
    # Scale 2.5: 6 RBF bases (cols 7-12), 1 neighbor count (col 13)
    # Scale 5.0: 6 RBF bases (cols 14-19), 1 neighbor count (col 20)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Neighbor count at Scale 1.0 (Local neighborhood density)
    ax = axes[0]
    colors = ["#4e79a7", "#f28e2b", "#e15759"]
    for c in [0, 1, 2]:
        mask = y_test_single == c
        ax.hist(X_test_single[mask, 6], bins=50, alpha=0.6, label=classes[c], color=colors[c], density=True)
    ax.set_title("Physical Feature Invariant: Local Density (Scale R=1.0)\nSeparates Worker, Manager, and CEO neighborhoods", fontsize=12, fontweight='bold')
    ax.set_xlabel("Mean Neighbor Count", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Key RBF Coupling Coefficient (Scale R=1.0, Basis 0)
    ax = axes[1]
    for c in [0, 1, 2]:
        mask = y_test_single == c
        ax.hist(X_test_single[mask, 0], bins=50, alpha=0.6, label=classes[c], color=colors[c], density=True)
    ax.set_title("Physical Feature Invariant: Interaction Coupling (Scale R=1.0, Basis 0)\nReveals discrete physical coupling bounds", fontsize=12, fontweight='bold')
    ax.set_xlabel("RBF Coefficient Value", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    feat_plot_path = "analysis/physical_feature_distributions.png"
    plt.savefig(feat_plot_path, dpi=200)
    print(f"Saved feature distribution plot to: {feat_plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
