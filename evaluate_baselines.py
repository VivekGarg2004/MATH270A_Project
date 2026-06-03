import sys
import os
import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from MVNN_Model.model import create_model

DT = 1e-3
FEATURE_NET_LAYER = [12, 24, 48]
DESCRIBE_NET_LAYER = [128, 128, 128, 1]
DIM = 1
CKPT_PATH = "MVNN_Model/jax_ckpt_000370.npz"

def load_mvnn():
    _, fcn_f = create_model(FEATURE_NET_LAYER, DESCRIBE_NET_LAYER, DIM)
    with open(CKPT_PATH, 'rb') as f:
        data = np.load(f, allow_pickle=True)
        params = data['params'].item()
    return fcn_f, params

def evaluate_mvnn_safe(params, fcn_f, x_w, v_w, x_m, v_m, x_c, v_c):
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
        
        if xw_c.shape[1] > 0:
            total_loss1 += float(jnp.sum((vw_c - f1)**2))
            total_v1_sq += float(jnp.sum(vw_c**2))
            
        if xm_c.shape[1] > 0:
            total_loss2 += float(jnp.sum((vm_c - f2)**2))
            total_v2_sq += float(jnp.sum(vm_c**2))
            
        if xc_c.shape[1] > 0:
            total_loss3 += float(jnp.sum((vc_c - f3)**2))
            total_v3_sq += float(jnp.sum(vc_c**2))
            
    loss1 = np.sqrt(total_loss1) / np.sqrt(total_v1_sq) if total_v1_sq > 0 else 0.0
    loss2 = np.sqrt(total_loss2) / np.sqrt(total_v2_sq) if total_v2_sq > 0 else 0.0
    loss3 = np.sqrt(total_loss3) / np.sqrt(total_v3_sq) if total_v3_sq > 0 else 0.0
    
    active_classes = (total_v1_sq > 0) + (total_v2_sq > 0) + (total_v3_sq > 0)
    error_total = (loss1 + loss2 + loss3) / active_classes
    return float(error_total)

def get_random_masks(seed, n_total):
    np.random.seed(seed)
    p = [16000/20200, 4000/20200, 200/20200]
    preds = np.random.choice([0, 1, 2], size=n_total, p=p)
    return (preds == 0), (preds == 1), (preds == 2)

def get_all_worker_masks(n_total):
    return np.ones(n_total, dtype=bool), np.zeros(n_total, dtype=bool), np.zeros(n_total, dtype=bool)

def main():
    fcn_f, mvnn_params = load_mvnn()
    test_seeds = list(range(70, 100))
    
    print("\nStarting Baseline Evaluations (Random vs All-Worker)")
    print("="*80)
    print(f"{'Seed':<6} | {'Random Lbl Error':<16} | {'All-Worker Error':<16}")
    print("-" * 80)
    
    rand_errors = []
    homog_errors = []
    
    for seed in test_seeds:
        try:
            data_workers = np.load(f"data/opinion_workers_mpi_{seed}.npy")[..., None]
            data_managers = np.load(f"data/opinion_managers_mpi_{seed}.npy")[..., None]
            data_ceos = np.load(f"data/opinion_ceos_mpi_{seed}.npy")[..., None]
        except FileNotFoundError:
            continue
            
        data_all = np.concatenate([data_workers, data_managers, data_ceos], axis=1)
        n_total = data_all.shape[1]
        
        # 1. Random Labels
        w_mask_r, m_mask_r, c_mask_r = get_random_masks(seed, n_total)
        data_w_r, data_m_r, data_c_r = data_all[:, w_mask_r], data_all[:, m_mask_r], data_all[:, c_mask_r]
        
        v_w_r = (data_w_r[1:] - data_w_r[:-1]) / DT
        v_m_r = (data_m_r[1:] - data_m_r[:-1]) / DT
        v_c_r = (data_c_r[1:] - data_c_r[:-1]) / DT
        
        err_rand = evaluate_mvnn_safe(mvnn_params, fcn_f, data_w_r[:-1], v_w_r, data_m_r[:-1], v_m_r, data_c_r[:-1], v_c_r)
        rand_errors.append(err_rand)
        
        # 2. All-Worker (Homogeneous)
        w_mask_h, m_mask_h, c_mask_h = get_all_worker_masks(n_total)
        data_w_h, data_m_h, data_c_h = data_all[:, w_mask_h], data_all[:, m_mask_h], data_all[:, c_mask_h]
        
        v_w_h = (data_w_h[1:] - data_w_h[:-1]) / DT
        v_m_h = (data_m_h[1:] - data_m_h[:-1]) / DT
        v_c_h = (data_c_h[1:] - data_c_h[:-1]) / DT
        
        err_homog = evaluate_mvnn_safe(mvnn_params, fcn_f, data_w_h[:-1], v_w_h, data_m_h[:-1], v_m_h, data_c_h[:-1], v_c_h)
        homog_errors.append(err_homog)
        
        print(f"{seed:<6} | {err_rand:<16.5f} | {err_homog:<16.5f}", flush=True)

    print("=" * 80)
    print(f"MEAN RANDOM LBL ERROR: {np.mean(rand_errors):.5f}")
    print(f"MEAN ALL-WORKER ERROR: {np.mean(homog_errors):.5f}")
    print("=" * 80)
    
if __name__ == "__main__":
    main()
