"""
PCA diagnostics for multi-scale RBF embedding.

Compares:
  1. Original single-scale embedding (r_max=5, 10 RBF bases)
  2. Multi-scale embedding: separate RBF bases at each physical
     interaction radius R1=1.0, R2=2.5, R3=5.0

Uses a small subsample for speed, runs on 3 seeds.
"""
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import numba as nb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.io import load_simulation_processed

# ── config ──
DT        = 1e-3
SUBSAMPLE = 5
MAX_STEP  = 200
DATA_DIR  = "data"

# Physical interaction radii from simulation.py
R_SCALES = [1.0, 2.5, 5.0]
M_PER_SCALE = 6          # RBF bases per scale
M_SINGLE    = 10          # bases for the single-scale baseline

SEEDS = [0, 1, 2]        # 3 sims for diagnostics

CLASS_NAMES  = ["Worker", "Manager", "CEO"]
CLASS_COLORS = ["#4e79a7", "#f28e2b", "#e15759"]


# ─────────────────────────────────────────────
# Numba kernel: embedding at a SINGLE r_max
# ─────────────────────────────────────────────
@nb.njit(parallel=True, fastmath=True)
def _embedding_at_scale(X, velocities, r_max, sigma_scales):
    """Compute RBF embedding + neighbor count at one scale."""
    T_v, N = velocities.shape
    M = sigma_scales.shape[0]
    embeddings  = np.zeros((N, M))
    mean_counts = np.zeros(N)
    inv_denom = 1.0 / (2.0 * sigma_scales**2)

    X_sorted = np.zeros((T_v, N))
    for t in range(T_v):
        X_sorted[t] = np.sort(X[t])

    for i in nb.prange(N):
        Phi = np.zeros((T_v, M))
        total_count = 0
        for t in range(T_v):
            xi = X[t, i]
            xs = X_sorted[t]
            left  = np.searchsorted(xs, xi - r_max, side='left')
            right = np.searchsorted(xs, xi + r_max, side='right')

            count = 0
            for idx in range(left, right):
                xj = xs[idx]
                diff = abs(xi - xj)
                if diff > 1e-12:
                    diff_sq = diff * diff
                    count += 1
                    for m in range(M):
                        Phi[t, m] += np.exp(-diff_sq * inv_denom[m])

            if count > 0:
                for m in range(M):
                    Phi[t, m] /= count
            total_count += count

        # Normal equations: (Phi.T @ Phi + λI) β = Phi.T @ v_i
        AtA = np.zeros((M, M))
        for m1 in range(M):
            for m2 in range(M):
                for t in range(T_v):
                    AtA[m1, m2] += Phi[t, m1] * Phi[t, m2]
        for m in range(M):
            AtA[m, m] += 1e-6

        Atb = np.zeros(M)
        v_i = velocities[:, i]
        for m in range(M):
            for t in range(T_v):
                Atb[m] += Phi[t, m] * v_i[t]

        embeddings[i]  = np.linalg.solve(AtA, Atb)
        mean_counts[i] = total_count / T_v

    return embeddings, mean_counts


# ─────────────────────────────────────────────
# High-level feature builders
# ─────────────────────────────────────────────
def compute_velocities(X):
    return np.diff(X, axis=0) / (DT * SUBSAMPLE)

def single_scale_features(X, vel):
    """Original approach: one r_max=5 with 10 RBF bases."""
    sigmas = np.linspace(0.5, 5.0, M_SINGLE)
    emb, counts = _embedding_at_scale(X, vel, 5.0, sigmas)
    return np.column_stack([emb, counts])

def multi_scale_features(X, vel):
    """NEW: separate embedding at each physical radius."""
    parts = []
    for r in R_SCALES:
        sigmas = np.linspace(0.1 * r, r, M_PER_SCALE)
        emb, counts = _embedding_at_scale(X, vel, r, sigmas)
        parts.append(emb)
        parts.append(counts[:, None])
    return np.hstack(parts)   # (N, M_PER_SCALE*3 + 3)


# ─────────────────────────────────────────────
# Main diagnostic
# ─────────────────────────────────────────────
def run_diagnostic():
    print("Loading simulations...")
    all_single = []
    all_multi  = []
    all_labels = []

    for seed in SEEDS:
        print(f"  seed {seed} ...", flush=True)
        X, labels = load_simulation_processed(DATA_DIR, seed, MAX_STEP, SUBSAMPLE)
        vel = compute_velocities(X)
        # trim X to match vel
        X = X[:vel.shape[0], :]

        sf = single_scale_features(X, vel)
        mf = multi_scale_features(X, vel)

        all_single.append(sf)
        all_multi.append(mf)
        all_labels.append(labels)

    F_single = np.vstack(all_single)
    F_multi  = np.vstack(all_multi)
    labels   = np.concatenate(all_labels)

    print(f"\nSingle-scale feature shape: {F_single.shape}")
    print(f"Multi-scale feature shape:  {F_multi.shape}")
    print(f"Label counts: W={np.sum(labels==0)}, M={np.sum(labels==1)}, C={np.sum(labels==2)}")

    # ── PCA ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for row, (F_raw, title_prefix) in enumerate([
        (F_single, "Single-scale (r=5)"),
        (F_multi,  "Multi-scale (r=1,2.5,5)")
    ]):
        # Standardize
        F = StandardScaler().fit_transform(np.nan_to_num(F_raw, nan=0.0, posinf=0.0, neginf=0.0))

        pca = PCA()
        Z = pca.fit_transform(F)

        # (a) PC1 vs PC2
        ax = axes[row, 0]
        for c in [0, 1, 2]:
            mask = labels == c
            idx = np.where(mask)[0]
            if c == 0:
                s, alpha, zorder = 1, 0.15, 1
            elif c == 1:
                s, alpha, zorder = 4, 0.35, 2
            else:
                s, alpha, zorder = 25, 0.9, 3
            ax.scatter(Z[idx, 0], Z[idx, 1],
                       s=s, alpha=alpha, zorder=zorder,
                       c=CLASS_COLORS[c], label=CLASS_NAMES[c],
                       edgecolors='none')
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(f"{title_prefix} — PC1 vs PC2")
        ax.legend(markerscale=3)

        # (b) PC1 vs PC3
        ax = axes[row, 1]
        for c in [0, 1, 2]:
            mask = labels == c
            idx = np.where(mask)[0]
            if c == 0:
                s, alpha, zorder = 1, 0.15, 1
            elif c == 1:
                s, alpha, zorder = 4, 0.35, 2
            else:
                s, alpha, zorder = 25, 0.9, 3
            ax.scatter(Z[idx, 0], Z[idx, 2],
                       s=s, alpha=alpha, zorder=zorder,
                       c=CLASS_COLORS[c], label=CLASS_NAMES[c],
                       edgecolors='none')
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
        ax.set_title(f"{title_prefix} — PC1 vs PC3")
        ax.legend(markerscale=3)

        # (c) Explained variance (scree plot)
        ax = axes[row, 2]
        n_show = min(15, len(pca.explained_variance_ratio_))
        cumvar = np.cumsum(pca.explained_variance_ratio_[:n_show]) * 100
        ax.bar(range(1, n_show+1), pca.explained_variance_ratio_[:n_show]*100,
               color="#4e79a7", alpha=0.7, label="Individual")
        ax.plot(range(1, n_show+1), cumvar, 'o-', color="#e15759", label="Cumulative")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance (%)")
        ax.set_title(f"{title_prefix} — Scree Plot")
        ax.legend()
        ax.set_xticks(range(1, n_show+1))

        # Print per-class centroid in PC space
        print(f"\n{'='*60}")
        print(f"{title_prefix}")
        print(f"{'='*60}")
        print(f"Explained variance (first 5): {pca.explained_variance_ratio_[:5]}")
        print(f"Cumulative at PC5: {np.sum(pca.explained_variance_ratio_[:5])*100:.1f}%")
        for c in range(3):
            mask = labels == c
            centroid = Z[mask, :3].mean(axis=0)
            spread   = Z[mask, :3].std(axis=0)
            print(f"  {CLASS_NAMES[c]:10s} centroid=({centroid[0]:+.3f}, {centroid[1]:+.3f}, {centroid[2]:+.3f})  "
                  f"spread=({spread[0]:.3f}, {spread[1]:.3f}, {spread[2]:.3f})")

    plt.tight_layout()
    out_path = os.path.join(project_root, "analysis", "pca_multiscale_diagnostic.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved figure to {out_path}")
    plt.close()

    # ── Per-class feature distributions (quick summary) ──
    print("\n\nMulti-scale feature column means by class:")
    col_names = []
    for r in R_SCALES:
        for j in range(M_PER_SCALE):
            col_names.append(f"r={r:.1f}_rbf{j}")
        col_names.append(f"r={r:.1f}_count")

    F_multi_clean = np.nan_to_num(F_multi, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"{'Feature':25s} {'Worker':>10s} {'Manager':>10s} {'CEO':>10s}")
    print("-" * 58)
    for j, name in enumerate(col_names):
        vals = [F_multi_clean[labels == c, j].mean() for c in range(3)]
        print(f"{name:25s} {vals[0]:10.4f} {vals[1]:10.4f} {vals[2]:10.4f}")


if __name__ == "__main__":
    run_diagnostic()
