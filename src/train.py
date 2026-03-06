"""
train.py
--------
Trains one LightGBM model per device subtype with stratified K-Fold CV.
Device types are split on common[0].Md (0=DER Simulator 10kW, 1=DER Simulator 100kW).
Each subtype has structurally different active features, so separate models
learn cleaner decision boundaries without cross-device noise.

Memory strategy:
  - Reads parquet once per device type (~1.3GB per subset vs 2.6GB full)
  - Converts to float32 numpy column-by-column (no float64 intermediate)
  - Frees each subset before moving to the next

Usage:
    py src/sample_train.py   # run once first if not done
    py src/train.py
"""

import gc
import os
import re
import sys
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import add_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data-source")
MODEL_DIR = os.path.join(ROOT, "models")
TRAIN_PARQUET = os.path.join(DATA_DIR, "train_sample.parquet")

DEVICE_COL = "common[0].Md"   # encoded: 0 = 10kW simulator, 1 = 100kW simulator
DEVICE_NAMES = {0: "DER Simulator 10kW", 1: "DER Simulator 100kW"}
# -1 = NaN rows (unknown device type) — excluded from per-device training

N_FOLDS = 5
SEED = 42
BETA = 2

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
    best_thresh, best_score = 0.5, 0.0
    for thresh in np.linspace(0.01, 0.99, n_steps):
        score = f2_score(y_true, y_pred_proba, threshold=thresh)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh, best_score


def sanitize_names(names: list[str]) -> list[str]:
    return [re.sub(r'[\[\]\.\{\}:,"<> ]', '_', n) for n in names]


def df_to_float32_numpy(df: pd.DataFrame) -> np.ndarray:
    """Column-by-column conversion to float32 — avoids float64 intermediate."""
    arr = np.empty((len(df), len(df.columns)), dtype=np.float32)
    for i, col in enumerate(df.columns):
        arr[:, i] = df.iloc[:, i].to_numpy(dtype=np.float32, na_value=np.nan)
    return arr


def train_device(dev_val: int, global_indices: np.ndarray,
                 feature_cols: list[str], oof_proba: np.ndarray,
                 all_fold_scores: list, all_best_iters: list):
    """Train N_FOLDS models for one device subtype. Updates oof_proba in-place."""

    print(f"\n{'#'*60}")
    print(f"  Device {dev_val}: {DEVICE_NAMES[dev_val]}")
    print(f"  Training rows: {len(global_indices):,}")
    print(f"{'#'*60}")

    # Re-read parquet and filter to this device type
    train_full = pd.read_parquet(TRAIN_PARQUET)
    mask = train_full[DEVICE_COL].to_numpy() == dev_val
    train_dev = train_full[mask].reset_index(drop=True)
    del train_full
    gc.collect()

    train_dev = add_features(train_dev)
    y_dev = train_dev["Label"].to_numpy(dtype=np.int8)

    print(f"  Converting to float32 numpy ...")
    X_dev = df_to_float32_numpy(train_dev[feature_cols])
    del train_dev
    gc.collect()
    print(f"  X shape: {X_dev.shape} | size: {X_dev.nbytes/1e9:.2f} GB")

    safe_names = sanitize_names(feature_cols)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    dev_fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), 1):
        print(f"\n  {'='*46}")
        print(f"  Fold {fold}/{N_FOLDS}")
        print(f"  {'='*46}")

        X_tr = X_dev[tr_idx].copy()
        X_val = X_dev[val_idx].copy()
        y_tr, y_val = y_dev[tr_idx], y_dev[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=safe_names,
                             free_raw_data=True)
        dval   = lgb.Dataset(X_val, label=y_val, feature_name=safe_names,
                             reference=dtrain, free_raw_data=True)
        del X_tr
        gc.collect()

        model = lgb.train(
            LGBM_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(100),
            ],
        )

        val_proba = model.predict(X_val, num_iteration=model.best_iteration)

        # Store OOF at the correct global positions
        oof_proba[global_indices[val_idx]] = val_proba

        thresh, score = tune_threshold(y_val, val_proba)
        dev_fold_scores.append(score)
        all_fold_scores.append(score)
        all_best_iters.append(model.best_iteration)
        print(f"  Fold {fold} F2={score:.4f} @ threshold={thresh:.3f}")

        # Save model
        model_path = os.path.join(MODEL_DIR, f"lgbm_dev{dev_val}_fold{fold}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        del X_val, dtrain, dval
        gc.collect()

    print(f"\n  Dev {dev_val} mean F2: "
          f"{np.mean(dev_fold_scores):.4f} ± {np.std(dev_fold_scores):.4f}")
    del X_dev
    gc.collect()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_PARQUET):
        raise FileNotFoundError(
            f"{TRAIN_PARQUET} not found.\n"
            "Run:  py src/sample_train.py  first."
        )

    # ------------------------------------------------------------------ layout pass
    # Read only the columns needed to set up indexing — cheap
    print("Reading dataset layout ...")
    layout = pd.read_parquet(TRAIN_PARQUET, columns=["Id", "Label", DEVICE_COL])
    ids_all   = layout["Id"].to_numpy()
    y_all     = layout["Label"].to_numpy(dtype=np.int8)
    dev_all   = layout[DEVICE_COL].to_numpy()
    n_total   = len(layout)

    # Determine feature cols from one full row read (with FE applied)
    sample_df = pd.read_parquet(TRAIN_PARQUET).head(10)
    sample_df = add_features(sample_df)
    feature_cols = [c for c in sample_df.columns if c not in ("Id", "Label")]
    del sample_df, layout
    gc.collect()

    all_dev_values = sorted(np.unique(dev_all).tolist())
    dev_values = [v for v in all_dev_values if v >= 0]  # exclude -1 (NaN device type)
    n_unknown = (dev_all == -1).sum()
    print(f"  Total rows: {n_total:,} | Devices found: {all_dev_values} "
          f"| Training on: {dev_values} | Unknown (-1): {n_unknown:,} | Features: {len(feature_cols)}")
    print(f"  Positive rate: {y_all.mean()*100:.1f}%")

    # ------------------------------------------------------------------ per-device training
    oof_proba       = np.zeros(n_total, dtype=np.float32)
    all_fold_scores = []
    all_best_iters  = []

    for dev_val in dev_values:
        global_indices = np.where(dev_all == dev_val)[0]
        train_device(dev_val, global_indices, feature_cols,
                     oof_proba, all_fold_scores, all_best_iters)

    # ------------------------------------------------------------------ OOF evaluation
    print(f"\n{'='*60}")
    print("Combined OOF Results")
    print(f"{'='*60}")
    oof_thresh, oof_f2 = tune_threshold(y_all, oof_proba)
    print(f"  All fold F2s: {[f'{s:.4f}' for s in all_fold_scores]}")
    print(f"  Mean fold F2: {np.mean(all_fold_scores):.4f} ± {np.std(all_fold_scores):.4f}")
    print(f"  OOF F2 (global): {oof_f2:.4f} @ threshold={oof_thresh:.3f}")

    # ------------------------------------------------------------------ save
    oof_df = pd.DataFrame({"Id": ids_all, "oof_proba": oof_proba, "Label": y_all})
    oof_df.to_parquet(os.path.join(MODEL_DIR, "oof_predictions.parquet"), index=False)

    meta = {
        "feature_cols":      feature_cols,
        "device_col":        DEVICE_COL,
        "dev_values":        dev_values,
        "optimal_threshold": float(oof_thresh),
        "oof_f2":            float(oof_f2),
        "fold_f2_scores":    [float(s) for s in all_fold_scores],
        "n_folds":           N_FOLDS,
        "best_iterations":   all_best_iters,
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved 10 models, OOF predictions, and meta.json")
    print(f"  threshold={oof_thresh:.3f} | OOF F2={oof_f2:.4f}")


if __name__ == "__main__":
    main()
