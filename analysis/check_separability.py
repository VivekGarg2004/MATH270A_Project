# analysis/check_separability.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from models.vec.wals_features import get_or_compute_features

def main():
    print("Loading simulation seed 0 and extracting physical multiscale features...")
    feat, labels = get_or_compute_features("data", 0, cache_dir="data/cache")
    
    classes = ["Worker", "Manager", "CEO"]
    
    print("\n" + "=" * 105)
    print(f"{'Feature Name':30s} | {'Worker (16,000 particles)':23s} | {'Manager (4,000 particles)':23s} | {'CEO (200 particles)':23s}")
    print("=" * 105)
    
    # Let's inspect some RBF coefficients and neighbor counts across the three scales
    features_to_check = [
        (0, "R=1.0 RBF Basis 0 (Coupling)"),
        (1, "R=1.0 RBF Basis 1"),
        (6, "R=1.0 Mean Neighbor Count"),
        (7, "R=2.5 RBF Basis 0 (Coupling)"),
        (13, "R=2.5 Mean Neighbor Count"),
        (14, "R=5.0 RBF Basis 0 (Coupling)"),
        (20, "R=5.0 Mean Neighbor Count"),
    ]
    
    for idx, name in features_to_check:
        w_vals = feat[labels == 0, idx]
        m_vals = feat[labels == 1, idx]
        c_vals = feat[labels == 2, idx]
        
        print(f"{name:30s} | {w_vals.mean():9.4f} ± {w_vals.std():6.4f} | {m_vals.mean():9.4f} ± {m_vals.std():6.4f} | {c_vals.mean():9.4f} ± {c_vals.std():6.4f}")
    
    print("=" * 105)
    print("\nLet's also look at the standard deviation ratio (Max/Min) of these class means to check separation:")
    for idx, name in features_to_check:
        means = [feat[labels == c, idx].mean() for c in range(3)]
        stds = [feat[labels == c, idx].std() for c in range(3)]
        # Distance between CEO and Worker mean in units of joint std
        joint_std = np.sqrt(stds[0]**2 + stds[2]**2)
        dist = abs(means[2] - means[0]) / (joint_std + 1e-10)
        print(f"  {name:30s}: CEO vs Worker mean distance = {dist:.2f} standard deviations")

if __name__ == "__main__":
    main()
