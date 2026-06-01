import sys
import os
import joblib
import numpy as np
from sklearn.metrics import recall_score

# Add project root to sys path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from run_hybrid_model import load_hybrid_dataset

def main():
    clf = joblib.load("models/saved/hybrid_rf_model.joblib")
    scaler = joblib.load("models/saved/robust_scaler.joblib")
    multipliers = np.load("models/saved/optimal_multipliers.npy")
    
    CACHE_DIR = "data/cache"
    seeds = list(range(70, 100))
    
    print(f"{'Seed':<6} | {'Worker Acc':<12} | {'Manager Acc':<12} | {'CEO Acc':<12}")
    print("-" * 55)
    
    worker_accs, manager_accs, ceo_accs = [], [], []
    
    for seed in seeds:
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
        
        # Calculate recall (accuracy per true class)
        w_acc = np.mean(preds[y_true == 0] == 0)
        m_acc = np.mean(preds[y_true == 1] == 1)
        c_acc = np.mean(preds[y_true == 2] == 2)
        
        worker_accs.append(w_acc)
        manager_accs.append(m_acc)
        ceo_accs.append(c_acc)
        
        print(f"{seed:<6} | {w_acc:10.2%} | {m_acc:10.2%} | {c_acc:10.2%}")

    print("=" * 55)
    print(f"{'MEAN':<6} | {np.mean(worker_accs):10.2%} | {np.mean(manager_accs):10.2%} | {np.mean(ceo_accs):10.2%}")
    print("=" * 55)

if __name__ == "__main__":
    main()
