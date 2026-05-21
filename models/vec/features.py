# models/vec/features.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import numpy as np
import numba as nb
from configs.config import cfg as _cfg
from utils.io import load_simulation_processed

# Extract configuration tokens at the module level
cfg          = _cfg.vec
DATA_DIR     = cfg.DATA_DIR
MAX_STEP     = cfg.MAX_STEP
SUBSAMPLE    = cfg.SUBSAMPLE
DT           = cfg.DT
SIGMA_SCALES = cfg.SIGMA_SCALES
M            = cfg.M
r_max        = cfg.r_max

def load_simulation(seed):
    return load_simulation_processed(DATA_DIR, seed, MAX_STEP, SUBSAMPLE)

def compute_velocities(X):
    dX = np.diff(X, axis=0) / (DT * SUBSAMPLE)
    return dX
@nb.njit(parallel=True, fastmath=True)
def _compute_embedding_kernel(X, velocities, M, r_max, sigma_scales):
    T_v, N = velocities.shape
    embeddings = np.zeros((N, M))
    mean_counts = np.zeros(N) 
    denom = 2.0 * sigma_scales**2
    
    # 1. Sort X at each timestep once to enable binary neighbor search
    X_sorted = np.zeros((T_v, N))
    for t in range(T_v):
        X_sorted[t] = np.sort(X[t])
        
    # 2. Parallel loop over each particle
    for i in nb.prange(N):
        Phi = np.zeros((T_v, M))
        total_count = 0 
        for t in range(T_v):
            xi = X[t, i]
            xs = X_sorted[t]
            
            # Binary search for neighbor range
            left = np.searchsorted(xs, xi - r_max, side='left')
            right = np.searchsorted(xs, xi + r_max, side='right')
            
            # Accumulate RBF basis over actual neighbors
            count = 0
            for idx in range(left, right):
                xj = xs[idx]
                diff = abs(xi - xj)
                if diff > 1e-12:
                    diff_sq = diff * diff
                    count += 1
                    for m in range(M):
                        Phi[t, m] += np.exp(-diff_sq / denom[m])
            
            # normalize by neighbor count
            if count > 0:
                for m in range(M):
                    Phi[t, m] /= count
            total_count += count
                        
        # 3. Direct evaluation of AtA = Phi.T @ Phi
        AtA = np.zeros((M, M))
        for m1 in range(M):
            for m2 in range(M):
                for t in range(T_v):
                    AtA[m1, m2] += Phi[t, m1] * Phi[t, m2]
                    
        # Add tiny ridge regularization to diagonal
        for m in range(M):
            AtA[m, m] += 1e-6
            
        # 4. Direct evaluation of Atb = Phi.T @ velocities[:, i]
        Atb = np.zeros(M)
        v_i = velocities[:, i]
        for m in range(M):
            for t in range(T_v):
                Atb[m] += Phi[t, m] * v_i[t]
        
        # 5. Solve the linear system
        embeddings[i] = np.linalg.solve(AtA, Atb)
        mean_counts[i] = total_count / T_v
        
    return embeddings, mean_counts

def compute_embedding(X, velocities):
    embeddings, mean_counts = _compute_embedding_kernel(X, velocities, M, r_max, SIGMA_SCALES)
    return np.column_stack([embeddings, mean_counts])


@nb.njit(parallel=True, fastmath=True)
def _compute_simple_features(X, velocities, r_max):
    T_v, N = velocities.shape
    features = np.zeros((N, 3))
    
    X_sorted = np.zeros((T_v, N))
    for t in range(T_v):
        X_sorted[t] = np.sort(X[t])
    
    for i in nb.prange(N):
        mean_abs_vel = 0.0
        mean_count   = 0.0
        mean_dist    = 0.0
        
        for t in range(T_v):
            xi  = X[t, i]
            xs  = X_sorted[t]
            
            mean_abs_vel += abs(velocities[t, i])
            
            left  = np.searchsorted(xs, xi - r_max, side='left')
            right = np.searchsorted(xs, xi + r_max, side='right')
            
            count     = 0
            dist_sum  = 0.0
            for idx in range(left, right):
                xj   = xs[idx]
                diff = abs(xi - xj)
                if diff > 1e-12:
                    count    += 1
                    dist_sum += diff
            
            mean_count += count
            if count > 0:
                mean_dist += dist_sum / count
        
        features[i, 0] = mean_abs_vel / T_v
        features[i, 1] = mean_count   / T_v
        features[i, 2] = mean_dist    / T_v
    return features

def compute_simple_features(X, velocities):
    return _compute_simple_features(X, velocities, r_max)

def compute_autocorr_features(X, velocities, n_lags=20):
    # velocities shape: (T_v, N)
    # returns (N, n_lags)
    T_v, N = velocities.shape
    features = np.zeros((N, n_lags))
    
    for i in range(N):
        v = velocities[:, i]
        # normalize by variance
        v_centered = v - v.mean()
        var = np.var(v_centered)
        if var < 1e-4:  # particle has essentially stopped
            features[i] = 0.0
            continue
        for lag in range(1, n_lags + 1):
            features[i, lag-1] = np.clip(
                    np.mean(v_centered[lag:] * v_centered[:-lag]) / var, 
                    -1.0, 1.0
                )
    
    return features

@nb.njit(parallel=True, fastmath=True)
def _compute_neighbor_vel_corr(X, velocities, r_scales):
    T_v, N = velocities.shape
    n_scales = r_scales.shape[0]
    features = np.zeros((N, n_scales))

    X_sorted  = np.zeros((T_v, N))
    sort_idx  = np.zeros((T_v, N), dtype=np.int64)
    for t in range(T_v):
        idx = np.argsort(X[t])
        sort_idx[t] = idx
        X_sorted[t] = X[t, idx]

    for i in nb.prange(N):
        v_i = velocities[:, i]
        # center and normalize particle i velocity
        v_mean = 0.0
        for t in range(T_v):
            v_mean += v_i[t]
        v_mean /= T_v
        v_var = 0.0
        for t in range(T_v):
            v_var += (v_i[t] - v_mean) ** 2
        v_std = np.sqrt(v_var / T_v)

        for s in range(n_scales):
            r = r_scales[s]
            dot    = 0.0
            f_var  = 0.0
            f_mean = 0.0

            # first pass: compute mean neighbor velocity at each timestep
            F = np.zeros(T_v)
            counts = np.zeros(T_v)
            for t in range(T_v):
                xi   = X[t, i]
                xs   = X_sorted[t]
                left  = np.searchsorted(xs, xi - r, side='left')
                right = np.searchsorted(xs, xi + r, side='right')
                count = 0
                vsum  = 0.0
                for idx in range(left, right):
                    j = sort_idx[t, idx]
                    if j != i:
                        vsum  += velocities[t, j]
                        count += 1
                if count > 0:
                    F[t] = vsum / count  # mean neighbor velocity
                counts[t] = count

            # center F
            f_mean = 0.0
            for t in range(T_v):
                f_mean += F[t]
            f_mean /= T_v

            # compute correlation
            num   = 0.0
            f_var = 0.0
            for t in range(T_v):
                num   += (v_i[t] - v_mean) * (F[t] - f_mean)
                f_var += (F[t] - f_mean) ** 2
            f_std = np.sqrt(f_var / T_v)

            if v_std > 1e-6 and f_std > 1e-6:
                features[i, s] = num / (T_v * v_std * f_std)
            else:
                features[i, s] = 0.0

    return features

def compute_neighbor_vel_corr(X, velocities):
    r_scales = np.linspace(0.5, r_max, 10)
    return _compute_neighbor_vel_corr(X, velocities, r_scales)