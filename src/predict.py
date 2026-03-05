"""
predict.py
----------
Loads trained fold models and generates test predictions.
Averages probabilities across folds, then applies the optimal threshold.
Outputs a submission CSV.

Usage:
    py src/predict.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data-source")
MODEL_DIR = os.path.join(ROOT, "models")
SUBMISSIONS_DIR = os.path.join(ROOT, "submissions")
TEST_PARQUET = os.path.join(DATA_DIR, "test.parquet")


def main():
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

    # ------------------------------------------------------------------ meta
    meta_path = os.path.join(MODEL_DIR, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    threshold = meta["optimal_threshold"]
    n_folds = meta["n_folds"]
    print(f"Loaded meta: {n_folds} folds, threshold={threshold:.3f}, OOF F2={meta['oof_f2']:.4f}")

    # ------------------------------------------------------------------ load test
    print("Loading test data ...")
    test = pd.read_parquet(TEST_PARQUET)
    print(f"  Shape: {test.shape}")

    X_test = test[feature_cols].to_numpy(dtype=np.float32)
    ids = test["Id"].to_numpy()

    # ------------------------------------------------------------------ predict
    fold_probas = []
    for fold in range(1, n_folds + 1):
        model_path = os.path.join(MODEL_DIR, f"lgbm_fold{fold}.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        proba = model.predict(X_test, num_iteration=model.best_iteration)
        fold_probas.append(proba)
        print(f"  Fold {fold} predicted (mean proba={proba.mean():.4f})")

    # Average across folds
    avg_proba = np.mean(fold_probas, axis=0)
    print(f"\nEnsemble mean proba: {avg_proba.mean():.4f}")
    print(f"Predicted positive rate @ threshold={threshold:.3f}: "
          f"{(avg_proba >= threshold).mean()*100:.1f}%")

    # ------------------------------------------------------------------ submission
    # Despite format docs saying "probability", Kaggle's scorer calls
    # sklearn.fbeta_score directly and requires binary 0/1 predictions.
    binary_labels = (avg_proba >= threshold).astype(np.int8)
    print(f"Positive predictions: {binary_labels.sum():,} / {len(binary_labels):,} "
          f"({binary_labels.mean()*100:.1f}%)")
    submission = pd.DataFrame({"Id": ids, "Label": binary_labels})

    out_path = os.path.join(SUBMISSIONS_DIR, "submission_lgbm.csv")
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved: {out_path} ({len(submission)} rows)")
    print(submission.head())


if __name__ == "__main__":
    main()
