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

    feature_cols    = meta["feature_cols"]
    global_threshold = meta["optimal_threshold"]
    n_folds         = meta["n_folds"]
    device_col      = meta["device_col"]
    dev_values      = meta["dev_values"]

    # Per-device thresholds (dev1_threshold added by train_dev1.py)
    dev_thresholds = {dv: global_threshold for dv in dev_values}
    if "dev1_threshold" in meta:
        dev_thresholds[1] = meta["dev1_threshold"]
        print(f"Using Device 1 threshold: {meta['dev1_threshold']:.4f} "
              f"(dev1 OOF F2={meta.get('dev1_oof_f2', 'n/a')})")

    print(f"OOF F2={meta['oof_f2']:.4f} | global_threshold={global_threshold:.3f} | "
          f"devices={dev_values} | folds={n_folds} | "
          f"per-device thresholds={dev_thresholds}")

    # ------------------------------------------------------------------ load test
    print("\nLoading test data ...")
    test = pd.read_parquet(TEST_PARQUET)
    test = add_features(test)
    print(f"  Shape: {test.shape}")

    ids       = test["Id"].to_numpy()
    dev_col   = test[device_col].to_numpy()
    avg_proba = np.zeros(len(test), dtype=np.float32)

    def predict_for_mask(mask, dev_vals_to_use):
        """Average predictions from all fold models of the given device types."""
        X = test.loc[mask, feature_cols].to_numpy(dtype=np.float32)
        all_probas = []
        for dv in dev_vals_to_use:
            for fold in range(1, n_folds + 1):
                model_path = os.path.join(MODEL_DIR, f"lgbm_dev{dv}_fold{fold}.pkl")
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                all_probas.append(model.predict(X, num_iteration=model.best_iteration))
        return np.mean(all_probas, axis=0)

    # ------------------------------------------------------------------ predict per device
    for dev_val in dev_values:
        mask = (dev_col == dev_val)
        n_dev = mask.sum()
        print(f"\nDevice {dev_val}: {n_dev:,} rows")
        avg_proba[mask] = predict_for_mask(mask, [dev_val])
        print(f"  mean proba={avg_proba[mask].mean():.4f}")

    # Handle unknown device type (-1): average over all device models
    unknown_mask = (dev_col == -1)
    n_unknown = unknown_mask.sum()
    if n_unknown > 0:
        print(f"\nUnknown device (-1): {n_unknown:,} rows — using average of all models")
        avg_proba[unknown_mask] = predict_for_mask(unknown_mask, dev_values)

    # ------------------------------------------------------------------ submission
    print(f"\nEnsemble mean proba: {avg_proba.mean():.4f}")

    # Apply per-device threshold (falls back to global_threshold for unknowns)
    binary_labels = np.zeros(len(avg_proba), dtype=np.int8)
    for dev_val in dev_values:
        mask = (dev_col == dev_val)
        t    = dev_thresholds.get(dev_val, global_threshold)
        binary_labels[mask] = (avg_proba[mask] >= t).astype(np.int8)
    unknown_mask = (dev_col == -1)
    if unknown_mask.any():
        binary_labels[unknown_mask] = (
            avg_proba[unknown_mask] >= global_threshold
        ).astype(np.int8)

    print(f"Positive predictions: {binary_labels.sum():,} / {len(binary_labels):,} "
          f"({binary_labels.mean()*100:.1f}%)")

    submission = pd.DataFrame({"Id": ids, "Label": binary_labels})
    out_path = os.path.join(SUBMISSIONS_DIR, "submission_lgbm.csv")
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved: {out_path} ({len(submission):,} rows)")
    print(submission.head())


if __name__ == "__main__":
    main()
