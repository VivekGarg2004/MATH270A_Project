import sys
import os
import numpy as np
import joblib

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluate_mvnn_pipeline import get_predicted_masks

def main():
    clf = joblib.load("models/saved/hybrid_rf_model.joblib")
    scaler = joblib.load("models/saved/robust_scaler.joblib")
    multipliers = np.load("models/saved/optimal_multipliers.npy")

    w_mask, m_mask, c_mask, preds, y_true = get_predicted_masks(97, clf, scaler, multipliers)

    w_wrong = np.sum((y_true == 0) & (preds != 0))
    m_wrong = np.sum((y_true == 1) & (preds != 1))
    c_wrong = np.sum((y_true == 2) & (preds != 2))

    print(f"Seed 97 Misclassifications out of 20,200 total agents:")
    print(f"Workers Misclassified:  {w_wrong} out of 16000")
    print(f"Managers Misclassified: {m_wrong} out of 4000")
    print(f"CEOs Misclassified:     {c_wrong} out of 200")

    print("\nWorker Misclassifications:")
    print(f"  -> Predicted as Manager: {np.sum((y_true == 0) & (preds == 1))}")
    print(f"  -> Predicted as CEO:     {np.sum((y_true == 0) & (preds == 2))}")

    print("\nManager Misclassifications:")
    print(f"  -> Predicted as Worker: {np.sum((y_true == 1) & (preds == 0))}")
    print(f"  -> Predicted as CEO:    {np.sum((y_true == 1) & (preds == 2))}")

    print("\nCEO Misclassifications:")
    print(f"  -> Predicted as Worker:  {np.sum((y_true == 2) & (preds == 0))}")
    print(f"  -> Predicted as Manager: {np.sum((y_true == 2) & (preds == 1))}")

    DT = 1e-3
    w = np.load(f"data/opinion_workers_mpi_{97}.npy")
    m = np.load(f"data/opinion_managers_mpi_{97}.npy")
    c = np.load(f"data/opinion_ceos_mpi_{97}.npy")

    v_w = np.abs(w[1:] - w[:-1]) / DT
    v_m = np.abs(m[1:] - m[:-1]) / DT
    v_c = np.abs(c[1:] - c[:-1]) / DT

    print("\nTrue Velocities on Seed 97:")
    print(f"Worker Mean Vel:  {np.mean(v_w):.5f} | Max: {np.max(v_w):.5f}")
    print(f"Manager Mean Vel: {np.mean(v_m):.5f} | Max: {np.max(v_m):.5f}")
    print(f"CEO Mean Vel:     {np.mean(v_c):.5f} | Max: {np.max(v_c):.5f}")

    w_to_c_indices = np.where((y_true == 0) & (preds == 2))[0]
    m_to_c_indices = np.where((y_true == 1) & (preds == 2))[0]
    
    if len(w_to_c_indices) > 0:
        w_to_c_vels = v_w[:, w_to_c_indices, :]
        print(f"\nMean Velocity of the {len(w_to_c_indices)} Workers misclassified as CEOs: {np.mean(w_to_c_vels):.5f}")
    
    if len(m_to_c_indices) > 0:
        m_idx_raw = m_to_c_indices - 16000
        m_to_c_vels = v_m[:, m_idx_raw, :]
        print(f"Mean Velocity of the {len(m_to_c_indices)} Managers misclassified as CEOs: {np.mean(m_to_c_vels):.5f}")

if __name__ == "__main__":
    main()
