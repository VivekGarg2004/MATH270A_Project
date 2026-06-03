import sys
import os
import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from MVNN_Model.model import MLP

DT = 1e-3
FEATURE_NET_LAYER = [12, 24, 48]
DESCRIBE_NET_LAYER = [128, 128, 128, 1]
DIM = 1
CKPT_PATH = "MVNN_Model/jax_ckpt_001970.npz"

def create_single_model(feature_net_layer, describe_net_layer, dim):
    feature_net_mlp = MLP(feature_net_layer)
    describe_net_mlp = MLP(describe_net_layer)
    
    def fcn_f(params, x):
        feature = feature_net_mlp.apply(params["feature_net_param"], x)
        mu_mean = jnp.mean(feature, axis=-2)[..., None, :]
        
        def broadcast_to_target(mu_mean, target_shape):
            return jnp.broadcast_to(mu_mean, (*mu_mean.shape[:-2], target_shape[-2], mu_mean.shape[-1]))
            
        force = describe_net_mlp.apply(params["describe_net_param"], jnp.concatenate([
            x,
            broadcast_to_target(mu_mean, x.shape)
        ], axis=-1))
        
        return force
        
    return fcn_f

def load_single_mvnn():
    fcn_f = create_single_model(FEATURE_NET_LAYER, DESCRIBE_NET_LAYER, DIM)
    with open(CKPT_PATH, 'rb') as f:
        data = np.load(f, allow_pickle=True)
        params = data['params'].item()
    return fcn_f, params

def evaluate_single_mvnn_safe(params, fcn_f, x_all, v_all):
    total_loss, total_v_sq = 0.0, 0.0
    
    chunk_size = 20
    num_chunks = int(np.ceil(x_all.shape[0] / chunk_size))
    
    for i in range(num_chunks):
        start, end = i * chunk_size, (i + 1) * chunk_size
        
        x_c, v_c = jnp.array(x_all[start:end]), jnp.array(v_all[start:end])
        
        f_pred = fcn_f(params, x_c)
        
        if x_c.shape[1] > 0:
            total_loss += float(jnp.sum((v_c - f_pred)**2))
            total_v_sq += float(jnp.sum(v_c**2))
            
    loss = np.sqrt(total_loss) / np.sqrt(total_v_sq) if total_v_sq > 0 else 0.0
    return float(loss)

def main():
    fcn_f, params = load_single_mvnn()
    test_seeds = list(range(70, 100))
    
    print("\nStarting Baseline Evaluation: 1-Class Homogeneous MVNN")
    print("="*80)
    print(f"{'Seed':<6} | {'Single-Class MVNN Error':<25}")
    print("-" * 80)
    
    errors = []
    
    for seed in test_seeds:
        try:
            data_workers = np.load(f"data/opinion_workers_mpi_{seed}.npy")[..., None]
            data_managers = np.load(f"data/opinion_managers_mpi_{seed}.npy")[..., None]
            data_ceos = np.load(f"data/opinion_ceos_mpi_{seed}.npy")[..., None]
        except FileNotFoundError:
            continue
            
        data_all = np.concatenate([data_workers, data_managers, data_ceos], axis=1)
        
        x_all = data_all[:-1]
        v_all = (data_all[1:] - data_all[:-1]) / DT
        
        err = evaluate_single_mvnn_safe(params, fcn_f, x_all, v_all)
        errors.append(err)
        
        print(f"{seed:<6} | {err:<25.5f}", flush=True)

    print("=" * 80)
    print(f"MEAN 1-CLASS MVNN ERROR: {np.mean(errors):.5f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
