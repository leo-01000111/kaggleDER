"""
train.py
--------
Trains a LightGBM model with stratified K-Fold CV.
Tunes classification threshold for F2 score.
Saves model, OOF predictions, and optimal threshold.

Memory strategy:
  - Loads train_sample.parquet (~826k rows, ~1.8GB float32)
  - Converts DataFrame to float32 numpy before CV (bypasses LightGBM's
    Windows-specific float64 conversion of pandas DataFrames)
  - Frees DataFrame and full numpy array before each fold to minimise peak RAM

Usage:
    py src/sample_train.py   # run once first if not done
    py src/train.py
"""

import gc
import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data-source")
MODEL_DIR = os.path.join(ROOT, "models")
TRAIN_PARQUET = os.path.join(DATA_DIR, "train_sample.parquet")

N_FOLDS = 5
SEED = 42
BETA = 2  # F-beta score beta

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 255,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": SEED,
}

NUM_BOOST_ROUND = 3000
EARLY_STOPPING = 150


def f2_score(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return fbeta_score(y_true, y_pred, beta=BETA, zero_division=0)


def tune_threshold(y_true, y_pred_proba, n_steps=200):
    """Grid search threshold to maximise F2."""
    best_thresh, best_score = 0.5, 0.0
    for thresh in np.linspace(0.01, 0.99, n_steps):
        score = f2_score(y_true, y_pred_proba, threshold=thresh)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh, best_score


def sanitize_names(names: list[str]) -> list[str]:
    """Replace JSON-special chars that LightGBM rejects in feature names."""
    return [re.sub(r'[\[\]\.\{\}:,"<> ]', '_', n) for n in names]


def df_to_float32_numpy(df: pd.DataFrame) -> np.ndarray:
    """
    Convert a mixed-dtype DataFrame to a contiguous float32 numpy array.
    Builds column by column to avoid a float64 intermediate allocation.
    """
    n_rows, n_cols = df.shape
    arr = np.empty((n_rows, n_cols), dtype=np.float32)
    for i, col in enumerate(df.columns):
        arr[:, i] = df.iloc[:, i].to_numpy(dtype=np.float32, na_value=np.nan)
    return arr


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_PARQUET):
        raise FileNotFoundError(
            f"{TRAIN_PARQUET} not found.\n"
            "Run:  py src/sample_train.py  first."
        )

    # ------------------------------------------------------------------ load
    print("Loading sampled train data ...")
    train = pd.read_parquet(TRAIN_PARQUET)
    print(f"  Shape: {train.shape}")

    feature_cols = [c for c in train.columns if c not in ("Id", "Label")]
    y = train["Label"].to_numpy(dtype=np.int8)
    ids = train["Id"].to_numpy()

    print(f"  Features: {len(feature_cols)}")
    print(f"  Positive rate: {y.mean()*100:.1f}%")

    # Convert to float32 numpy — bypasses LightGBM's Windows float64 conversion.
    # Column-by-column build avoids a full float64 intermediate.
    print("  Converting features to float32 numpy ...")
    X = df_to_float32_numpy(train[feature_cols])
    del train
    gc.collect()
    print(f"  X shape: {X.shape}, dtype: {X.dtype}, "
          f"size: {X.nbytes/1e9:.2f} GB")

    # ------------------------------------------------------------------ CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_proba = np.zeros(len(y), dtype=np.float32)
    fold_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*50}")
        print(f"Fold {fold}/{N_FOLDS}")
        print(f"{'='*50}")

        # Explicit copies so we can free X during training
        X_tr = X[train_idx].copy()
        X_val = X[val_idx].copy()
        y_tr, y_val = y[train_idx], y[val_idx]

        safe_names = sanitize_names(feature_cols)
        # lgb.Dataset accepts float32 numpy directly on Windows (no float64 cast)
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=safe_names,
                             free_raw_data=True)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=safe_names,
                           reference=dtrain, free_raw_data=True)

        del X_tr  # LightGBM has ingested it; free before training
        gc.collect()

        callbacks = [
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(100),
        ]

        model = lgb.train(
            LGBM_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        val_proba = model.predict(X_val, num_iteration=model.best_iteration)
        oof_proba[val_idx] = val_proba

        thresh, score = tune_threshold(y_val, val_proba)
        fold_scores.append(score)
        print(f"  Fold {fold} F2={score:.4f} @ threshold={thresh:.3f}")

        del X_val, dtrain, dval
        gc.collect()
        models.append(model)

    # ------------------------------------------------------------------ OOF evaluation
    print(f"\n{'='*50}")
    print("OOF Results")
    print(f"{'='*50}")
    oof_thresh, oof_f2 = tune_threshold(y, oof_proba)
    print(f"  Per-fold F2: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"  Mean fold F2: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"  OOF F2 (global): {oof_f2:.4f} @ threshold={oof_thresh:.3f}")

    # ------------------------------------------------------------------ save
    print(f"\nSaving models and OOF predictions ...")

    for i, model in enumerate(models):
        model_path = os.path.join(MODEL_DIR, f"lgbm_fold{i+1}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    oof_df = pd.DataFrame({"Id": ids, "oof_proba": oof_proba, "Label": y})
    oof_df.to_parquet(os.path.join(MODEL_DIR, "oof_predictions.parquet"), index=False)

    meta = {
        "feature_cols": feature_cols,
        "optimal_threshold": float(oof_thresh),
        "oof_f2": float(oof_f2),
        "fold_f2_scores": [float(s) for s in fold_scores],
        "n_folds": N_FOLDS,
        "best_iterations": [m.best_iteration for m in models],
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved {N_FOLDS} models to {MODEL_DIR}/")
    print(f"  Saved OOF predictions")
    print(f"  Saved meta.json (threshold={oof_thresh:.3f}, OOF F2={oof_f2:.4f})")


if __name__ == "__main__":
    main()
