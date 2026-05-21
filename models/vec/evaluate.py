# evaluate.py
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from scipy.optimize import linear_sum_assignment

def align_labels(true, pred, n_clusters=3):
    # K-means labels are arbitrary so we need to find best permutation
    cost = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost[i, j] = -np.sum((true == i) & (pred == j))
    _, col_ind = linear_sum_assignment(cost)
    aligned = np.zeros_like(pred)
    for i, j in enumerate(col_ind):
        aligned[pred == j] = i
    return aligned

def evaluate(kmeans, scaler, seeds, get_embeddings_fn):
    embeddings, labels = get_embeddings_fn(seeds)
    embeddings         = scaler.transform(embeddings)
    pred               = kmeans.predict(embeddings)
    pred               = align_labels(labels, pred)
    bal_acc            = balanced_accuracy_score(labels, pred)
    f1                 = f1_score(labels, pred, average=None)
    print(f"balanced accuracy: {bal_acc:.3f}")
    print(f"per-class F1 \u2014 workers: {f1[0]:.3f}  managers: {f1[1]:.3f}  CEOs: {f1[2]:.3f}")
    return bal_acc, f1