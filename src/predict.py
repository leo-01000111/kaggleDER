"""
predict.py
----------
Loads per-device-type fold models and generates test predictions.
Splits test set by device subtype, applies the appropriate models,
then recombines in original row order before applying threshold.

Usage:
    py src/predict.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import add_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data-source")
MODEL_DIR = os.path.join(ROOT, "models")
SUBMISSIONS_DIR = os.path.join(ROOT, "submissions")
TEST_PARQUET = os.path.join(DATA_DIR, "test.parquet")


def main():
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

    # ------------------------------------------------------------------ meta
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    threshold    = meta["optimal_threshold"]
    n_folds      = meta["n_folds"]
    device_col   = meta["device_col"]
    dev_values   = meta["dev_values"]
    print(f"OOF F2={meta['oof_f2']:.4f} | threshold={threshold:.3f} | "
          f"devices={dev_values} | folds={n_folds}")

    # ------------------------------------------------------------------ load test
    print("\nLoading test data ...")
    test = pd.read_parquet(TEST_PARQUET)
    test = add_features(test)
    print(f"  Shape: {test.shape}")

    ids       = test["Id"].to_numpy()
    dev_col   = test[device_col].to_numpy()
    avg_proba = np.zeros(len(test), dtype=np.float32)

    # ------------------------------------------------------------------ predict per device
    for dev_val in dev_values:
        mask = (dev_col == dev_val)
        n_dev = mask.sum()
        print(f"\nDevice {dev_val}: {n_dev:,} rows")

        X_dev = test.loc[mask, feature_cols].to_numpy(dtype=np.float32)

        fold_probas = []
        for fold in range(1, n_folds + 1):
            model_path = os.path.join(MODEL_DIR, f"lgbm_dev{dev_val}_fold{fold}.pkl")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            proba = model.predict(X_dev, num_iteration=model.best_iteration)
            fold_probas.append(proba)
            print(f"  Fold {fold} predicted (mean={proba.mean():.4f})")

        avg_proba[mask] = np.mean(fold_probas, axis=0)

    # ------------------------------------------------------------------ submission
    print(f"\nEnsemble mean proba: {avg_proba.mean():.4f}")
    binary_labels = (avg_proba >= threshold).astype(np.int8)
    print(f"Positive predictions: {binary_labels.sum():,} / {len(binary_labels):,} "
          f"({binary_labels.mean()*100:.1f}%)")

    submission = pd.DataFrame({"Id": ids, "Label": binary_labels})
    out_path = os.path.join(SUBMISSIONS_DIR, "submission_lgbm.csv")
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved: {out_path} ({len(submission):,} rows)")
    print(submission.head())


if __name__ == "__main__":
    main()
