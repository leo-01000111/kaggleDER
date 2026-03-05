"""
leak_hunt.py
------------
Finds features that perfectly or near-perfectly predict the label.
Checks:
  1. Feature importances from trained models (dominant features)
  2. Per-feature correlation and AUC with label
  3. Id column as a predictor
  4. Any single feature that gives F2 > 0.99

Usage:
    py src/leak_hunt.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, fbeta_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data-source")
MODEL_DIR = os.path.join(ROOT, "models")
TRAIN_PARQUET = os.path.join(DATA_DIR, "train.parquet")

TOP_N = 30  # how many top features to report


def f2(y_true, y_pred_binary):
    return fbeta_score(y_true, y_pred_binary, beta=2, zero_division=0)


def main():
    # ------------------------------------------------------------------ load meta + models
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    print("Loading fold 1 model for feature importances ...")
    with open(os.path.join(MODEL_DIR, "lgbm_fold1.pkl"), "rb") as f:
        model = pickle.load(f)

    # ------------------------------------------------------------------ feature importance
    print("\n=== Top feature importances (gain) ===")
    importances = model.feature_importance(importance_type="gain")
    safe_names = [n.replace("[", "_").replace("]", "_").replace(".", "_") for n in feature_cols]
    imp_df = pd.DataFrame({"feature": feature_cols, "safe": safe_names, "gain": importances})
    imp_df = imp_df.sort_values("gain", ascending=False).reset_index(drop=True)
    print(imp_df.head(TOP_N).to_string(index=False))

    top_features = imp_df.head(TOP_N)["feature"].tolist()

    # ------------------------------------------------------------------ load a sample of train for correlation check
    print(f"\nLoading train sample for correlation analysis ...")
    # Read only the top features + Label to stay memory-efficient
    cols_to_load = list(set(top_features + ["Id", "Label"]))
    train = pd.read_parquet(TRAIN_PARQUET, columns=cols_to_load)
    y = train["Label"].to_numpy()
    print(f"  Rows: {len(train):,}")

    # ------------------------------------------------------------------ Id correlation
    print("\n=== Id column analysis ===")
    id_vals = train["Id"].to_numpy()
    id_auc = roc_auc_score(y, id_vals)
    print(f"  AUC of Id vs Label: {id_auc:.4f}")
    for thresh in [id_vals.mean(), np.median(id_vals)]:
        pred = (id_vals >= thresh).astype(int)
        print(f"  F2 @ Id >= {thresh:.0f}: {f2(y, pred):.4f}")

    # ------------------------------------------------------------------ per-feature AUC
    print(f"\n=== Per-feature AUC vs Label (top {TOP_N} by importance) ===")
    results = []
    for col in top_features:
        if col not in train.columns:
            continue
        vals = train[col].to_numpy(dtype=np.float32)
        # Skip constant or all-null columns
        valid = ~np.isnan(vals)
        if valid.sum() < 100 or np.nanstd(vals) == 0:
            continue
        try:
            auc = roc_auc_score(y[valid], vals[valid])
            auc = max(auc, 1 - auc)  # flip if inverted
            results.append((col, auc))
        except Exception:
            pass

    results.sort(key=lambda x: x[1], reverse=True)
    print(f"{'Feature':<60} {'AUC':>6}")
    print("-" * 68)
    for col, auc in results[:TOP_N]:
        flag = " *** LEAK CANDIDATE ***" if auc > 0.99 else ""
        print(f"{col:<60} {auc:>6.4f}{flag}")

    # ------------------------------------------------------------------ perfect single-feature check
    print("\n=== Scanning ALL features for perfect/near-perfect predictors ===")
    all_cols = [c for c in pd.read_parquet(TRAIN_PARQUET, columns=["Id"]).columns
                if c not in ("Id", "Label")]
    # Re-load full feature set in chunks via the parquet
    pf_cols = [c for c in feature_cols if c not in top_features][:100]  # check next 100 too
    extra = pd.read_parquet(TRAIN_PARQUET, columns=pf_cols + ["Label"])
    y2 = extra["Label"].to_numpy()

    leaks = []
    for col in pf_cols:
        vals = extra[col].to_numpy(dtype=np.float32)
        valid = ~np.isnan(vals)
        if valid.sum() < 100 or np.nanstd(vals) == 0:
            continue
        try:
            auc = roc_auc_score(y2[valid], vals[valid])
            auc = max(auc, 1 - auc)
            if auc > 0.98:
                leaks.append((col, auc))
        except Exception:
            pass

    if leaks:
        leaks.sort(key=lambda x: x[1], reverse=True)
        print("LEAK CANDIDATES FOUND:")
        for col, auc in leaks:
            print(f"  {col}: AUC={auc:.5f}")
    else:
        print("  No additional leaks found in next 100 features by importance.")


if __name__ == "__main__":
    main()
