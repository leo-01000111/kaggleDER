"""
find_leak.py
------------
Scans Device 0 (and optionally Device 1) on the NEW dataset for remaining
leakage sources now that Id / DERMode / DeviceType / DerSimControls are fixed.

Three checks:
  1. AUC scan — any single feature with AUC > 0.90 is a direct leak
  2. Null-rate diff — columns whose null rate differs significantly between
     normal and anomaly rows (NaN-pattern leak, like DERMode was)
  3. Constant-value diff — columns that are constant in normal rows but not
     in anomaly rows, or vice versa

Usage:
    py src/find_leak.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data-source")
MODEL_DIR = os.path.join(ROOT, "models")
TRAIN_PQ  = os.path.join(DATA_DIR, "train_sample.parquet")

DEVICE_COL = "common[0].Md"
LABEL_COL  = "Label"

AUC_THRESHOLD      = 0.70   # flag as suspicious
NULL_DIFF_THRESHOLD = 0.10  # flag if null rate differs by > 10 pp
TOP_N_AUC          = 30


def check_device(df_dev, label, device_name):
    y = df_dev[LABEL_COL].to_numpy()
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    print(f"\n{'#'*60}")
    print(f"  {device_name}  |  n={len(df_dev):,}  pos={n_pos:,}  neg={n_neg:,}")
    print(f"{'#'*60}")

    feature_cols = [c for c in df_dev.columns if c not in (LABEL_COL, "Id")]

    # ── 1. AUC scan ────────────────────────────────────────────────────────────
    print(f"\n--- AUC scan (top {TOP_N_AUC} features) ---")
    aucs = {}
    for col in feature_cols:
        vals = df_dev[col].to_numpy(dtype=np.float32)
        if np.isnan(vals).all():
            continue
        # fill NaN with a sentinel for AUC (nan → -999 makes nans a group)
        filled = np.where(np.isnan(vals), -999.0, vals)
        try:
            auc = roc_auc_score(y, filled)
            aucs[col] = max(auc, 1 - auc)  # always ≥ 0.5
        except Exception:
            pass

    top = sorted(aucs.items(), key=lambda x: -x[1])[:TOP_N_AUC]
    print(f"  {'Feature':<55} {'AUC':>7}")
    print(f"  {'-'*63}")
    for col, auc in top:
        flag = "  *** LEAK ***" if auc >= AUC_THRESHOLD else ""
        print(f"  {col:<55} {auc:>7.4f}{flag}")

    # ── 2. Null-rate difference between normal and anomaly ─────────────────────
    print(f"\n--- Null-rate leak (|null_rate_normal - null_rate_anomaly| > "
          f"{NULL_DIFF_THRESHOLD:.0%}) ---")
    null_leaks = []
    for col in feature_cols:
        vals = df_dev[col].to_numpy()
        null_pos = np.isnan(vals[pos_mask].astype(float)).mean() if n_pos else 0
        null_neg = np.isnan(vals[neg_mask].astype(float)).mean() if n_neg else 0
        diff = abs(null_pos - null_neg)
        if diff >= NULL_DIFF_THRESHOLD:
            null_leaks.append((col, null_neg, null_pos, diff))

    if null_leaks:
        null_leaks.sort(key=lambda x: -x[3])
        print(f"  {'Feature':<55} {'NullNorm':>9} {'NullAnom':>9} {'Diff':>7}")
        print(f"  {'-'*83}")
        for col, nn, na, d in null_leaks[:30]:
            print(f"  {col:<55} {nn:>9.4f} {na:>9.4f} {d:>7.4f}")
    else:
        print("  None found.")

    # ── 3. Constant-value difference ───────────────────────────────────────────
    print(f"\n--- Constant-value leak (constant in normal XOR anomaly) ---")
    const_leaks = []
    for col in feature_cols:
        vals_pos = df_dev.loc[pos_mask, col].dropna()
        vals_neg = df_dev.loc[neg_mask, col].dropna()
        if len(vals_pos) == 0 or len(vals_neg) == 0:
            continue
        pos_unique = vals_pos.nunique()
        neg_unique = vals_neg.nunique()
        # Flag if one class is constant (1 unique value) but the other varies
        if (pos_unique == 1 and neg_unique > 5) or (neg_unique == 1 and pos_unique > 5):
            const_leaks.append((col, neg_unique, pos_unique))

    if const_leaks:
        const_leaks.sort(key=lambda x: -(max(x[1], x[2])))
        print(f"  {'Feature':<55} {'UniqueNorm':>10} {'UniqueAnom':>10}")
        print(f"  {'-'*78}")
        for col, nu, au in const_leaks[:20]:
            print(f"  {col:<55} {nu:>10} {au:>10}")
    else:
        print("  None found.")


def main():
    print(f"Loading {TRAIN_PQ} ...")
    df = pd.read_parquet(TRAIN_PQ)
    print(f"  Shape: {df.shape}")
    print(f"  Label dist: {df[LABEL_COL].value_counts().sort_index().to_dict()}")

    devices = sorted(df[DEVICE_COL].dropna().unique().astype(int).tolist())
    device_names = {0: "Device 0 — DER Simulator 10kW",
                    1: "Device 1 — DER Simulator 100kW"}

    for dev_val in devices:
        if dev_val < 0:
            continue
        df_dev = df[df[DEVICE_COL] == dev_val].reset_index(drop=True)
        check_device(df_dev, dev_val, device_names.get(dev_val, f"Device {dev_val}"))

    print(f"\n{'='*60}")
    print("Done. Features with AUC ≥ 0.70 or large null-rate diffs are likely leaks.")
    print("Report them to the competition organisers.")


if __name__ == "__main__":
    main()
