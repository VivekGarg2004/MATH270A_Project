import numpy as np

ckpt1 = np.load("MVNN_Model/jax_ckpt_000370.npz", allow_pickle=True)
ckpt2 = np.load("MVNN_Model/jax_ckpt_001970.npz", allow_pickle=True)

params1 = ckpt1['params'].item()
params2 = ckpt2['params'].item()

print("Keys in 3-class checkpoint (000370):")
print(params1.keys())
print("\nKeys in 1-class checkpoint (001970):")
print(params2.keys())
