import sys
import os
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

# Network settings
DT = 1e-3
FEATURE_NET_LAYER = [12, 24, 48]
DESCRIBE_NET_LAYER = [128, 128, 128, 1]
DIM = 1
CKPT_PATH = "MVNN_Model/jax_ckpt_000370.npz"
CACHE_DIR = "data/cache"

def load_mvnn():
    print(f"Loading MVNN Checkpoint from {CKPT_PATH}...")
    _, fcn_f = create_model(FEATURE_NET_LAYER, DESCRIBE_NET_LAYER, DIM)
    with open(CKPT_PATH, 'rb') as f:
        data = np.load(f, allow_pickle=True)
        params = data['params'].item()
    return fcn_f, params

def evaluate_mvnn(params, fcn_f, x_w, v_w, x_m, v_m, x_c, v_c):
    total_loss1, total_v1_sq = 0.0, 0.0
    total_loss2, total_v2_sq = 0.0, 0.0
    total_loss3, total_v3_sq = 0.0, 0.0
    
    chunk_size = 20
    num_chunks = int(np.ceil(x_w.shape[0] / chunk_size))
    
    for i in range(num_chunks):
        start, end = i * chunk_size, (i + 1) * chunk_size
        
        xw_c, vw_c = jnp.array(x_w[start:end]), jnp.array(v_w[start:end])
        xm_c, vm_c = jnp.array(x_m[start:end]), jnp.array(v_m[start:end])
        xc_c, vc_c = jnp.array(x_c[start:end]), jnp.array(v_c[start:end])
        
        f1, f2, f3 = fcn_f(params, xw_c, xm_c, xc_c)
        
        total_loss1 += float(jnp.sum((vw_c - f1)**2))
        total_v1_sq += float(jnp.sum(vw_c**2))
        
        total_loss2 += float(jnp.sum((vm_c - f2)**2))
        total_v2_sq += float(jnp.sum(vm_c**2))
        
        total_loss3 += float(jnp.sum((vc_c - f3)**2))
        total_v3_sq += float(jnp.sum(vc_c**2))
        
    loss1 = np.sqrt(total_loss1) / np.sqrt(total_v1_sq)
    loss2 = np.sqrt(total_loss2) / np.sqrt(total_v2_sq)
    loss3 = np.sqrt(total_loss3) / np.sqrt(total_v3_sq)
    
    error_total = (loss1 + loss2 + loss3) / 3.0
    return float(error_total), float(loss1), float(loss2), float(loss3)

def load_hybrid_rf():
    print("Loading Hybrid RF Model Artifacts...")
    clf = joblib.load("models/saved/hybrid_rf_model.joblib")
    scaler = joblib.load("models/saved/robust_scaler.joblib")
    multipliers = np.load("models/saved/optimal_multipliers.npy")
    return clf, scaler, multipliers

def get_predicted_masks(seed, clf, scaler, multipliers):
    # Suppress internal prints of load_hybrid_dataset
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        X, y_true = load_hybrid_dataset([seed], cache_dir=CACHE_DIR)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
        
    X_scaled = scaler.transform(X)
    probs = clf.predict_proba(X_scaled)
    preds = np.argmax(probs * multipliers, axis=1)
    
    w_mask = (preds == 0)
    m_mask = (preds == 1)
    c_mask = (preds == 2)
    return w_mask, m_mask, c_mask, preds, y_true

def main():
    fcn_f, mvnn_params = load_mvnn()
    clf, scaler, multipliers = load_hybrid_rf()
    
    # Test seeds as per our previous evaluation
    test_seeds = list(range(70, 100))
    
    print("\nStarting MVNN Downstream Evaluation on 30 Test Seeds...")
    print("="*80)
    print(f"{'Seed':<6} | {'GT Total Error':<16} | {'Pred Total Error':<16} | {'Δ Error':<10}")
    print("-" * 80)
    
    gt_errors = []
    pred_errors = []
    
    for seed in test_seeds:
        # Load raw data
        try:
            data_workers = np.load(f"data/opinion_workers_mpi_{seed}.npy")[..., None]
            data_managers = np.load(f"data/opinion_managers_mpi_{seed}.npy")[..., None]
            data_ceos = np.load(f"data/opinion_ceos_mpi_{seed}.npy")[..., None]
        except FileNotFoundError:
            print(f"Skipping seed {seed} (data not found)")
            continue
            
        # 1. Ground Truth setup
        x_w_gt = data_workers[:-1]
        v_w_gt = (data_workers[1:] - data_workers[:-1]) / DT
        x_m_gt = data_managers[:-1]
        v_m_gt = (data_managers[1:] - data_managers[:-1]) / DT
        x_c_gt = data_ceos[:-1]
        v_c_gt = (data_ceos[1:] - data_ceos[:-1]) / DT
        
        # Evaluate Ground Truth
        err_gt, _, _, _ = evaluate_mvnn(mvnn_params, fcn_f, x_w_gt, v_w_gt, x_m_gt, v_m_gt, x_c_gt, v_c_gt)
        gt_errors.append(err_gt)
        
        # 2. Predicted setup
        w_mask, m_mask, c_mask, preds, _ = get_predicted_masks(seed, clf, scaler, multipliers)
        
        data_all = np.concatenate([data_workers, data_managers, data_ceos], axis=1)
        
        data_w_pred = data_all[:, w_mask, :]
        data_m_pred = data_all[:, m_mask, :]
        data_c_pred = data_all[:, c_mask, :]
        
        # Safety check in case a class has 0 predictions
        if data_w_pred.shape[1] == 0 or data_m_pred.shape[1] == 0 or data_c_pred.shape[1] == 0:
            print(f"{seed:<6} | {'---':<16} | {'---':<16} | (Missing predicted class)")
            continue
            
        x_w_pred = data_w_pred[:-1]
        v_w_pred = (data_w_pred[1:] - data_w_pred[:-1]) / DT
        x_m_pred = data_m_pred[:-1]
        v_m_pred = (data_m_pred[1:] - data_m_pred[:-1]) / DT
        x_c_pred = data_c_pred[:-1]
        v_c_pred = (data_c_pred[1:] - data_c_pred[:-1]) / DT
        
        # Evaluate Predicted
        err_pred, _, _, _ = evaluate_mvnn(mvnn_params, fcn_f, x_w_pred, v_w_pred, x_m_pred, v_m_pred, x_c_pred, v_c_pred)
        pred_errors.append(err_pred)
        
        # Print row
        delta = err_pred - err_gt
        print(f"{seed:<6} | {err_gt:<16.5f} | {err_pred:<16.5f} | {delta:<10.5f}", flush=True)

    print("=" * 80)
    print(f"MEAN GT ERROR:   {np.mean(gt_errors):.5f}")
    print(f"MEAN PRED ERROR: {np.mean(pred_errors):.5f}")
    print(f"AVG DIFFERENCE:  {np.mean(pred_errors) - np.mean(gt_errors):.5f}")
    print("=" * 80)
    
if __name__ == "__main__":
    main()
