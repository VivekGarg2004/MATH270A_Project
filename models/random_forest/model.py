"""
Random Forest with ratio features added to raw features.
Key insight: CEO detection needs scale-invariant features
but Worker/Manager separation needs absolute magnitudes.
Solution: include BOTH raw features AND ratio-to-median features.
Labels: 0=Worker, 1=Manager, 2=CEO
"""

import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR   = Path("../DATA_GENERATION4/data")
DT         = 1e-3
N1, N2, N3 = 16000, 4000, 200
N_TOTAL    = N1 + N2 + N3
TIMESTEP_WINDOWS = [5, 10, 20, 50, 100, 200, 350, 500]

TRAIN_SIMS = list(range(0,  70))
VAL_SIMS   = list(range(70, 85))
TEST_SIMS  = list(range(85, 101))

FEAT_NAMES = [
    # raw features (10)
    "mean_abs_drift", "std_drift", "mean_drift",
    "peak_drift", "mean_abs_accel", "peak_accel",
    "std_accel", "total_path", "opinion_variance", "opinion_range",
    # ratio features (4) — scale invariant, CEO specific
    "ratio_std_accel",      # std_accel_i / median(std_accel_all)
    "ratio_peak_accel",     # peak_accel_i / median(peak_accel_all)
    "ratio_mean_abs_accel", # mean_abs_accel_i / median(mean_abs_accel_all)
    "ratio_std_drift",      # std_drift_i / median(std_drift_all)
]

def extract_features(workers, managers, ceos, T_end):
    # truncate to first T_end timesteps
    w = workers[:T_end]
    m = managers[:T_end]
    c = ceos[:T_end]

    def feats(o):
        v = np.diff(o, axis=0) / DT
        a = np.diff(v, axis=0) / DT
        # guard against T too small for acceleration
        if a.shape[0] == 0:
            a = np.zeros_like(v)
        raw = np.stack([
            np.mean(np.abs(v), axis=0),
            np.std(v,          axis=0),
            np.mean(v,         axis=0),
            np.max(np.abs(v),  axis=0),
            np.mean(np.abs(a), axis=0),
            np.max(np.abs(a),  axis=0),
            np.std(a,          axis=0),
            np.sum(np.abs(v),  axis=0),
            np.std(o,          axis=0),
            np.max(o,          axis=0) - np.min(o, axis=0),
        ], axis=1)
        return raw

    X_raw = np.concatenate([feats(w), feats(m), feats(c)], axis=0)
    sim_median = np.median(X_raw, axis=0) + 1e-10
    ratio = np.stack([
        X_raw[:, 6] / sim_median[6],
        X_raw[:, 5] / sim_median[5],
        X_raw[:, 4] / sim_median[4],
        X_raw[:, 1] / sim_median[1],
    ], axis=1)

    X = np.concatenate([X_raw, ratio], axis=1)
    y = np.concatenate([
        np.zeros(N1, dtype=int),
        np.ones(N2,  dtype=int),
        2*np.ones(N3, dtype=int),
    ])
    return X, y

def load_simulation(idx):
    w = np.load(DATA_DIR / f"opinion_workers_mpi_{idx}.npy").squeeze()
    m = np.load(DATA_DIR / f"opinion_managers_mpi_{idx}.npy").squeeze()
    c = np.load(DATA_DIR / f"opinion_ceos_mpi_{idx}.npy").squeeze()
    return w, m, c

def train(sim_indices, timestep):
    print(f"\n── Training on sims {sim_indices[0]}-{sim_indices[-1]} with timestep {timestep} ──")
    all_X, all_y = [], []

    for idx in sim_indices:
        try:
            w, m, c = load_simulation(idx)
            X, y    = extract_features(w, m, c, timestep)
            all_X.append(X); all_y.append(y)
            print(f"  Sim {idx} loaded")
        except FileNotFoundError:
            print(f"  Sim {idx} not found, skipping")

    X_tr = np.concatenate(all_X)
    y_tr = np.concatenate(all_y)

    print(f"\n  Total: {X_tr.shape[0]} particles, {X_tr.shape[1]} features")
    print(f"  Class counts: Worker={( y_tr==0).sum()}, "
          f"Manager={(y_tr==1).sum()}, CEO={(y_tr==2).sum()}")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_tr, y_tr)

    print("\n  Train report:")
    print(classification_report(y_tr, clf.predict(X_tr),
          target_names=["Worker","Manager","CEO"], digits=3))

    print("  Feature importances:")
    for i in np.argsort(clf.feature_importances_)[::-1]:
        print(f"    {FEAT_NAMES[i]:30s}: {clf.feature_importances_[i]:.4f}")

    return clf

def evaluate(clf, sim_indices, split_name, timestep):
    print(f"\n{'='*60}\n{split_name}\n{'='*60}")
    all_true, all_pred, f1s = [], [], []

    for idx in sim_indices:
        try:
            w, m, c   = load_simulation(idx)
            X, y_true = extract_features(w, m, c, timestep)
        except FileNotFoundError:
            print(f"  Sim {idx} not found"); continue

        y_pred = clf.predict(X)
        f1     = f1_score(y_true, y_pred, average=None, labels=[0,1,2])
        print(f"  Sim {idx} — Worker:{f1[0]:.3f}  Manager:{f1[1]:.3f}  CEO:{f1[2]:.3f}")

        all_true.append(y_true); all_pred.append(y_pred); f1s.append(f1)

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    f1s      = np.array(f1s)

    print(f"\nAggregate {split_name}:")
    print(classification_report(all_true, all_pred,
          target_names=["Worker","Manager","CEO"], digits=3))

    print("Per-sim F1 mean ± std:")
    for c, name in enumerate(["Worker","Manager","CEO"]):
        print(f"  {name:10s}: {f1s[:,c].mean():.3f} ± {f1s[:,c].std():.3f}")

if __name__ == "__main__":
    # sanity check ratio features on sims 48 and 49
    # for check_idx in [48, 49]:
    #     print(f"\n── Sanity check sim {check_idx} ──")
    #     w, m, c = load_simulation(check_idx)
    #     X, y    = extract_features(w, m, c)
    #     print(f"  Feature shape: {X.shape}")
    #     ratio_feats = [10, 11, 12, 13]
    #     for k, name in enumerate(["Worker","Manager","CEO"]):
    #         vals = [X[y==k, i].mean() for i in ratio_feats]
    #         print(f"  {name:10s}: "
    #               f"ratio_std_accel={vals[0]:.3f}  "
    #               f"ratio_peak_accel={vals[1]:.3f}  "
    #               f"ratio_mean_abs_accel={vals[2]:.3f}")
    for timestep in TIMESTEP_WINDOWS:
        clf = train(TRAIN_SIMS, timestep)
        evaluate(clf, VAL_SIMS,  "VALIDATION", timestep)
        evaluate(clf, TEST_SIMS, "TEST", timestep)