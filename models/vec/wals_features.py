# models/vec/wals_features.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import numba as nb
from utils.io import load_simulation_processed

DT = 1e-3
SUBSAMPLE = 5
MAX_STEP = 200
R_SCALES = np.array([1.0, 2.5, 5.0])
M_PER_SCALE = 6

@nb.njit(parallel=True, fastmath=True)
def _embedding_at_scale(X, velocities, r_max, sigma_scales):
    """Compute RBF embedding + neighbor count at one scale."""
    T_v, N = velocities.shape
    M = sigma_scales.shape[0]
    embeddings  = np.zeros((N, M))
    mean_counts = np.zeros(N)
    inv_denom = 1.0 / (2.0 * sigma_scales**2)

    # Pre-sort X at each timestep once to enable binary neighbor search
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

        # Solve Normal equations: (Phi.T @ Phi + λI) β = Phi.T @ v_i
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

def compute_velocities(X):
    return np.diff(X, axis=0) / (DT * SUBSAMPLE)

def multi_scale_features(X, vel):
    """Compute and stack separate embedding features at each physical scale."""
    parts = []
    for r in R_SCALES:
        sigmas = np.linspace(0.1 * r, r, M_PER_SCALE)
        emb, counts = _embedding_at_scale(X, vel, r, sigmas)
        parts.append(emb)
        parts.append(counts[:, None])
    return np.hstack(parts)   # returns (N, M_PER_SCALE * len(R_SCALES) + len(R_SCALES)) -> (N, 21)

def get_or_compute_features(data_dir, seed, max_step=MAX_STEP, subsample=SUBSAMPLE, cache_dir=None):
    """
    Loads simulation for seed, computes multi-scale physical features and labels, 
    caching results to avoid re-computation.
    """
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        feat_path = os.path.join(cache_dir, f"multiscale_features_seed_{seed}.npy")
        label_path = os.path.join(cache_dir, f"multiscale_labels_seed_{seed}.npy")
        if os.path.exists(feat_path) and os.path.exists(label_path):
            return np.load(feat_path), np.load(label_path)

    # If cache miss or caching disabled, load and compute
    X, labels = load_simulation_processed(data_dir, seed, max_step=max_step, subsample=subsample)
    vel = compute_velocities(X)
    X_trimmed = X[:vel.shape[0], :]
    
    features = multi_scale_features(X_trimmed, vel)
    features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if cache_dir is not None:
        np.save(feat_path, features_clean)
        np.save(label_path, labels)

    return features_clean, labels
