import sys
import os
import time
import numpy as np
import joblib

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Add project root to sys path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from MVNN_Model.model import create_model
from run_hybrid_model import load_hybrid_dataset

DT = 1e-3
FEATURE_NET_LAYER = [12, 24, 48]
DESCRIBE_NET_LAYER = [128, 128, 128, 1]
DIM = 1
CKPT_PATH = "MVNN_Model/jax_ckpt_000370.npz"
CACHE_DIR = "data/cache"

def load_mvnn():
    _, fcn_f = create_model(FEATURE_NET_LAYER, DESCRIBE_NET_LAYER, DIM)
    with open(CKPT_PATH, 'rb') as f:
        data = np.load(f, allow_pickle=True)
        params = data['params'].item()
    return fcn_f, params

def evaluate_mvnn_global(params, fcn_f, x_w, v_w, x_m, v_m, x_c, v_c):
    total_residual_sq = 0.0
    total_v_sq = 0.0
    
    chunk_size = 20
    num_chunks = int(np.ceil(x_w.shape[0] / chunk_size))
    
    for i in range(num_chunks):
        start, end = i * chunk_size, (i + 1) * chunk_size
        
        xw_c, vw_c = jnp.array(x_w[start:end]), jnp.array(v_w[start:end])
        xm_c, vm_c = jnp.array(x_m[start:end]), jnp.array(v_m[start:end])
        xc_c, vc_c = jnp.array(x_c[start:end]), jnp.array(v_c[start:end])
        
        f1, f2, f3 = fcn_f(params, xw_c, xm_c, xc_c)
        
        # Combine all residuals and velocities
        total_residual_sq += float(jnp.sum((vw_c - f1)**2) + jnp.sum((vm_c - f2)**2) + jnp.sum((vc_c - f3)**2))
        total_v_sq += float(jnp.sum(vw_c**2) + jnp.sum(vm_c**2) + jnp.sum(vc_c**2))
        
    global_relative_error = np.sqrt(total_residual_sq) / np.sqrt(total_v_sq)
    return global_relative_error

def main():
    fcn_f, mvnn_params = load_mvnn()
    clf = joblib.load("models/saved/hybrid_rf_model.joblib")
    scaler = joblib.load("models/saved/robust_scaler.joblib")
    multipliers = np.load("models/saved/optimal_multipliers.npy")
    
    test_seeds = list(range(70, 80))  # Test first 10 seeds
    
    print(f"{'Seed':<6} | {'GT Global Error':<16} | {'Pred Global Error':<16} | {'Δ Error':<10}")
    print("-" * 60)
    
    gt_errors = []
    pred_errors = []
    
    for seed in test_seeds:
        data_workers = np.load(f"data/opinion_workers_mpi_{seed}.npy")[..., None]
        data_managers = np.load(f"data/opinion_managers_mpi_{seed}.npy")[..., None]
        data_ceos = np.load(f"data/opinion_ceos_mpi_{seed}.npy")[..., None]
        
        # 1. Ground Truth
        x_w_gt = data_workers[:-1]
        v_w_gt = (data_workers[1:] - data_workers[:-1]) / DT
        x_m_gt = data_managers[:-1]
        v_m_gt = (data_managers[1:] - data_managers[:-1]) / DT
        x_c_gt = data_ceos[:-1]
        v_c_gt = (data_ceos[1:] - data_ceos[:-1]) / DT
        
        err_gt = evaluate_mvnn_global(mvnn_params, fcn_f, x_w_gt, v_w_gt, x_m_gt, v_m_gt, x_c_gt, v_c_gt)
        gt_errors.append(err_gt)
        
        # 2. Predicted
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            X, _ = load_hybrid_dataset([seed], cache_dir=CACHE_DIR)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
            
        X_scaled = scaler.transform(X)
        probs = clf.predict_proba(X_scaled)
        preds = np.argmax(probs * multipliers, axis=1)
        
        w_mask = (preds == 0)
        m_mask = (preds == 1)
        c_mask = (preds == 2)
        
        data_all = np.concatenate([data_workers, data_managers, data_ceos], axis=1)
        data_w_pred = data_all[:, w_mask, :]
        data_m_pred = data_all[:, m_mask, :]
        data_c_pred = data_all[:, c_mask, :]
        
        if data_w_pred.shape[1] == 0 or data_m_pred.shape[1] == 0 or data_c_pred.shape[1] == 0:
            continue
            
        x_w_pred = data_w_pred[:-1]
        v_w_pred = (data_w_pred[1:] - data_w_pred[:-1]) / DT
        x_m_pred = data_m_pred[:-1]
        v_m_pred = (data_m_pred[1:] - data_m_pred[:-1]) / DT
        x_c_pred = data_c_pred[:-1]
        v_c_pred = (data_c_pred[1:] - data_c_pred[:-1]) / DT
        
        err_pred = evaluate_mvnn_global(mvnn_params, fcn_f, x_w_pred, v_w_pred, x_m_pred, v_m_pred, x_c_pred, v_c_pred)
        pred_errors.append(err_pred)
        
        delta = err_pred - err_gt
        print(f"{seed:<6} | {err_gt:<16.5f} | {err_pred:<16.5f} | {delta:<10.5f}", flush=True)

if __name__ == "__main__":
    main()
