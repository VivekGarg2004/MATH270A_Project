"""
LDA diagnostic to project multi-scale RBF features into a class-separating 2D space.
Compares PCA (unsupervised) vs LDA (supervised) and evaluates a simple linear model.
"""
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.io import load_simulation_processed
# Import JIT features directly from pca_multiscale to reuse optimized code
from analysis.pca_multiscale import compute_velocities, multi_scale_features

SEEDS = [0, 1, 2]
CLASS_NAMES  = ["Worker", "Manager", "CEO"]
CLASS_COLORS = ["#4e79a7", "#f28e2b", "#e15759"]

def main():
    print("Loading simulations and extracting physical features...", flush=True)
    all_features = []
    all_labels = []

    for seed in SEEDS:
        print(f"  Processing seed {seed} ...", flush=True)
        X, labels = load_simulation_processed("data", seed, max_step=200, subsample=5)
        vel = compute_velocities(X)
        X = X[:vel.shape[0], :]
        
        mf = multi_scale_features(X, vel)
        all_features.append(mf)
        all_labels.append(labels)

    F_raw = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    # Clean NaNs/Infs
    F_clean = np.nan_to_num(F_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale the features
    scaler = StandardScaler()
    F_scaled = scaler.fit_transform(F_clean)

    print("\nFitting Linear Discriminant Analysis (LDA)...", flush=True)
    lda = LinearDiscriminantAnalysis(n_components=2)
    Z_lda = lda.fit_transform(F_scaled, labels)

    print("\nEvaluating LDA classification performance on training seeds (0, 1, 2):")
    preds = lda.predict(F_scaled)
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
    print(f"Balanced Accuracy: {balanced_accuracy_score(labels, preds):.4f}")

    # Plot the comparison: PCA (from pca_multiscale) vs LDA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(F_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot PCA
    ax = axes[0]
    for c in [0, 1, 2]:
        idx = np.where(labels == c)[0]
        s, alpha, zorder = (2, 0.15, 1) if c == 0 else ((6, 0.4, 2) if c == 1 else (30, 0.9, 3))
        ax.scatter(Z_pca[idx, 0], Z_pca[idx, 1],
                   s=s, alpha=alpha, zorder=zorder,
                   c=CLASS_COLORS[c], label=CLASS_NAMES[c],
                   edgecolors='none')
    ax.set_title("PCA Projection (Unsupervised)\nMaximizes global variance (dominated by Workers)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=3)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Plot LDA
    ax = axes[1]
    for c in [0, 1, 2]:
        idx = np.where(labels == c)[0]
        s, alpha, zorder = (2, 0.15, 1) if c == 0 else ((6, 0.4, 2) if c == 1 else (30, 0.9, 3))
        ax.scatter(Z_lda[idx, 0], Z_lda[idx, 1],
                   s=s, alpha=alpha, zorder=zorder,
                   c=CLASS_COLORS[c], label=CLASS_NAMES[c],
                   edgecolors='none')
    ax.set_title("LDA Projection (Supervised)\nMaximizes class separability")
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.legend(markerscale=3)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(project_root, "analysis", "lda_vs_pca_diagnostic.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved beautiful diagnostic plot to {out_path}", flush=True)

if __name__ == "__main__":
    main()
