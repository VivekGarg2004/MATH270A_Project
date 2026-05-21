# train.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from features import load_simulation, compute_velocities, compute_embedding

def get_embeddings_for_seeds(seeds):
    all_embeddings = []
    all_labels     = []
    for seed in seeds:
        print(f"processing seed {seed}")
        X, labels          = load_simulation(seed)
        velocities         = compute_velocities(X)
        embeddings         = compute_embedding(X, velocities)
        # shuffle particles
        idx                = np.random.permutation(len(labels))
        all_embeddings.append(embeddings[idx])
        all_labels.append(labels[idx])
    return np.concatenate(all_embeddings), np.concatenate(all_labels)

def train(train_seeds):
    embeddings, labels = get_embeddings_for_seeds(train_seeds)
    # standardize before clustering
    scaler     = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    kmeans     = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    return kmeans, scaler