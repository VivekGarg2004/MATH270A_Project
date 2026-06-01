#!/usr/bin/env python3
"""
Evaluate saved hybrid RF artifacts (from run_hybrid_model_save.py) on a trajectory dataset.

All settings are hyperparameters inside main() — no CLI / argparse.
Run from project root: python eval_hybrid_weights.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.vec.wals_features import get_or_compute_features
from run_wals_model import extract_baseline_rf_features_for_seed


def main() -> None:
    # ----- hyperparameters (edit here) ---------------------------------------------------------
    WEIGHTS_DIR = Path(project_root) / "weights" / "2026-05-22" / "22-15-59"
    DATA_DIR = os.path.join(project_root, "data_distribution_trial2", "data")
    CACHE_DIR = os.path.join(project_root, "data_distribution_trial2", "cache")
    EVAL_SEEDS = list(range(0, 50))
    MAX_STEP = 200
    EPS_MEDIAN = 1e-10
    CLASS_NAMES = ["Worker", "Manager", "CEO"]
    VERBOSE = True

    # --------------------------------------------------------------------------------------------
    weights_dir = Path(WEIGHTS_DIR)
    clf_path = weights_dir / "hybrid_rf.joblib"
    scaler_path = weights_dir / "hybrid_robust_scaler.joblib"
    mult_path = weights_dir / "hybrid_class_multipliers.npy"
    manifest_path = weights_dir / "manifest.json"

    if not clf_path.is_file():
        raise FileNotFoundError(f"Missing classifier: {clf_path}")
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")
    if not mult_path.is_file():
        raise FileNotFoundError(f"Missing multipliers: {mult_path}")

    clf = joblib.load(clf_path)
    scaler = joblib.load(scaler_path)
    multipliers = np.load(mult_path)

    if manifest_path.is_file() and VERBOSE:
        with open(manifest_path, encoding="utf-8") as f:
            meta = json.load(f)
        print("[eval] manifest:", json.dumps(meta, indent=2)[:2000], flush=True)

    if VERBOSE:
        print(f"[eval] weights_dir={weights_dir}", flush=True)
        print(f"[eval] DATA_DIR={DATA_DIR}", flush=True)
        print(f"[eval] CACHE_DIR={CACHE_DIR}", flush=True)
        print(f"[eval] seeds={EVAL_SEEDS}", flush=True)

    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []

    for seed in EVAL_SEEDS:
        phys, y_phys = get_or_compute_features(
            DATA_DIR,
            seed,
            max_step=MAX_STEP,
            cache_dir=CACHE_DIR,
        )
        kin, y_kin = extract_baseline_rf_features_for_seed(
            seed,
            max_step=MAX_STEP,
            data_dir=DATA_DIR,
        )
        if y_phys.shape[0] != y_kin.shape[0] or not np.array_equal(y_phys, y_kin):
            raise ValueError(
                f"seed={seed}: label mismatch between physical and kinematic paths "
                f"(phys {y_phys.shape}, kin {y_kin.shape})"
            )
        medians = np.median(phys, axis=0) + EPS_MEDIAN
        phys_ratio = phys / medians
        X = np.hstack([phys, kin, phys_ratio])

        if X.shape[1] != int(clf.n_features_in_):
            raise ValueError(
                f"seed={seed}: feature dim {X.shape[1]} != clf.n_features_in_={clf.n_features_in_}"
            )

        Xs = scaler.transform(X)
        probs = clf.predict_proba(Xs)
        pred = np.argmax(probs * multipliers, axis=1)

        all_y_true.append(y_phys)
        all_y_pred.append(pred)

        if VERBOSE:
            print(f"\n--- seed {seed} ---", flush=True)
            print(
                classification_report(
                    y_phys, pred, target_names=CLASS_NAMES, digits=4
                ),
                flush=True,
            )
            print(
                f"balanced_accuracy={balanced_accuracy_score(y_phys, pred):.4f}",
                flush=True,
            )

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    print("\n" + "=" * 60)
    print("[eval] pooled over all seeds")
    print("=" * 60)
    print(
        classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4),
        flush=True,
    )
    print(f"balanced_accuracy={balanced_accuracy_score(y_true, y_pred):.4f}", flush=True)
    f1_per = f1_score(y_true, y_pred, average=None)
    print(
        f"f1_worker={f1_per[0]:.4f}  f1_manager={f1_per[1]:.4f}  f1_ceo={f1_per[2]:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
