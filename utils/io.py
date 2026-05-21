# utils/io.py
import os
import numpy as np

def load_simulation_raw(data_dir, seed):
    """
    Loads raw worker, manager, and CEO opinion arrays for a given seed.
    """
    w = np.load(os.path.join(data_dir, f"opinion_workers_mpi_{seed}.npy"))
    m = np.load(os.path.join(data_dir, f"opinion_managers_mpi_{seed}.npy"))
    c = np.load(os.path.join(data_dir, f"opinion_ceos_mpi_{seed}.npy"))
    return w, m, c

def load_simulation_processed(data_dir, seed, max_step=None, subsample=1):
    """
    Loads, stacks species opinions into a single matrix, and generates labels.
    """
    w, m, c = load_simulation_raw(data_dir, seed)
    
    # Stack all particles along the particle axis (axis 1)
    X = np.concatenate([w, m, c], axis=1)
    
    # Apply optional time trimming and subsampling
    if max_step is not None:
        X = X[:max_step:subsample]
    elif subsample > 1:
        X = X[::subsample]
        
    N1, N2, N3 = w.shape[1], m.shape[1], c.shape[1]
    labels = np.array([0]*N1 + [1]*N2 + [2]*N3)
    
    return X, labels