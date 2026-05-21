# # main.py
# from features import load_simulation, compute_velocities, compute_autocorr_features, compute_neighbor_vel_corr
# import os
# import sys
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# import numpy as np
# from configs.config import cfg as _cfg
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# # diagnostic only — 2 seeds
# all_embeddings = []
# all_labels = []
# for seed in range(0, 2):
#     X, labels = load_simulation(seed)
#     velocities = compute_velocities(X)
#     embeddings = compute_neighbor_vel_corr(X, velocities)
#     all_embeddings.append(embeddings)
#     all_labels.append(labels)

# embeddings = np.concatenate(all_embeddings)
# labels = np.concatenate(all_labels)

# # print raw feature stats per species
# feature_names = [f"r_scale_{i}" for i in range(embeddings.shape[1])]
# for cls, name in [(0, 'workers'), (1, 'managers'), (2, 'CEOs')]:
#     mask = labels == cls
#     for f, fname in enumerate(feature_names):
#         vals = embeddings[mask, f]
#         print(f"{name:10s} | {fname:15s} | mean: {vals.mean():.4f}  std: {vals.std():.4f}")
#     print()

# # PCA plot
# scaler_diag = StandardScaler()
# emb_scaled = scaler_diag.fit_transform(embeddings)

# pca = PCA(n_components=2)
# emb_2d = pca.fit_transform(emb_scaled)

# plt.figure(figsize=(8, 6))
# for cls, name in [(0, 'workers'), (1, 'managers'), (2, 'CEOs')]:
#     mask = labels == cls
#     plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=1, alpha=0.3, label=name)
# plt.legend()
# plt.title('PCA of correlation features')
# plt.savefig('embedding_pca.png', dpi=150)
# print("saved embedding_pca.png")


# pick one seed, look at worker opinion distribution over time
import numpy as np
import matplotlib.pyplot as plt

seed = 0
w = np.load(f"data/opinion_workers_mpi_{seed}.npy")  # (T, N1)

plt.figure(figsize=(12, 4))
for t in [0, 100, 200, 300, 400, 499]:
    plt.plot(sorted(w[t]), label=f"t={t}")
plt.legend()
plt.title("Worker opinion distribution over time")
plt.savefig("worker_dist.png", dpi=150)
print("opinion range:", w[0].min(), w[0].max())
print("final range:", w[-1].min(), w[-1].max())