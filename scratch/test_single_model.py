import sys
import os
import numpy as np

import jax
import jax.numpy as jnp

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from MVNN_Model.model import MLP

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

ckpt = np.load("MVNN_Model/jax_ckpt_001970.npz", allow_pickle=True)
params = ckpt['params'].item()

print("Loaded 1-class params successfully.")

fcn_f = create_single_model([12, 24, 48], [128, 128, 128, 1], 1)

x = jnp.ones((20, 100, 1))
try:
    f = fcn_f(params, x)
    print("Success! Output shape:", f.shape)
except Exception as e:
    print("Error:", e)
