"""
features.py
-----------
Physics-based feature engineering for DER telemetry data.
Imported by train.py and predict.py — not run directly.

Features added:
  - Alarm activity: combined alarm register signals
  - Power quality: reactive ratio, apparent vs real power gap
  - Capacity violation: how far measurements exceed rated limits
  - Voltage/current imbalance across phases
  - Frequency deviation from nominal (60 Hz)
  - Operating mode flags
"""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Works on both train and test DataFrames."""

    # ------------------------------------------------------------------ helpers
    def col(name):
        """Return series if column exists, else zeros."""
        return df[name] if name in df.columns else pd.Series(0.0, index=df.index)

    # ------------------------------------------------------------------ alarm activity
    # AL1 and AL2 are IEEE 2030.5 alarm status registers (bitmasks)
    al1 = col("DERMeasureAC[0].AL1")
    al2 = col("DERMeasureAC[0].AL2")
    df["fe_alarm_sum"]    = al1 + al2
    df["fe_alarm_any"]    = ((al1 != 0) | (al2 != 0)).astype(np.float32)
    df["fe_alarm_both"]   = ((al1 != 0) & (al2 != 0)).astype(np.float32)

    # ------------------------------------------------------------------ power quality
    w   = col("DERMeasureAC[0].W")
    va  = col("DERMeasureAC[0].VA")
    var = col("DERMeasureAC[0].Var")

    # Reactive power ratio (var/VA) — anomalies shift power factor
    df["fe_reactive_ratio"] = (var / va.replace(0, np.nan)).fillna(0).astype(np.float32)

    # Apparent vs real power gap (VA - W)
    df["fe_power_gap"] = (va - w).astype(np.float32)

    # Power factor deviation from 1.0
    pf = col("DERMeasureAC[0].PF")
    df["fe_pf_deviation"] = (1.0 - pf.abs()).astype(np.float32)

    # ------------------------------------------------------------------ capacity violations
    w_max_ovr = col("DERCapacity[0].WMaxOvrExt")
    w_max_und = col("DERCapacity[0].WMaxUndExt")
    va_max    = col("DERCapacity[0].VAMaxRtg")

    # How much W exceeds its overextension limit
    df["fe_w_ovr_ratio"] = (w / w_max_ovr.replace(0, np.nan)).fillna(0).astype(np.float32)
    df["fe_w_und_ratio"] = (w / w_max_und.replace(0, np.nan)).fillna(0).astype(np.float32)
    df["fe_va_ratio"]    = (va / va_max.replace(0, np.nan)).fillna(0).astype(np.float32)

    # Asymmetry between over/under extension limits
    df["fe_wext_asymmetry"] = (w_max_ovr - w_max_und).astype(np.float32)

    # ------------------------------------------------------------------ voltage imbalance (3-phase)
    vl1 = col("DERMeasureAC[0].VL1")
    vl2 = col("DERMeasureAC[0].VL2")
    vl3 = col("DERMeasureAC[0].VL3")
    v_mean = (vl1 + vl2 + vl3) / 3.0
    df["fe_v_imbalance"] = (
        ((vl1 - v_mean).abs() + (vl2 - v_mean).abs() + (vl3 - v_mean).abs()) / v_mean.replace(0, np.nan)
    ).fillna(0).astype(np.float32)
    df["fe_v_mean"] = v_mean.astype(np.float32)

    # ------------------------------------------------------------------ current imbalance
    al1_a = col("DERMeasureAC[0].AL1")  # phase A alarm (reuse)
    a = col("DERMeasureAC[0].A")        # total current

    # Line voltages vs line-to-neutral voltage ratio (should be sqrt(3) ~1.732)
    lnv = col("DERMeasureAC[0].LNV")
    llv = col("DERMeasureAC[0].LLV")
    expected_ratio = 3.0 ** 0.5
    actual_ratio = (llv / lnv.replace(0, np.nan)).fillna(expected_ratio)
    df["fe_llv_lnv_ratio_dev"] = (actual_ratio - expected_ratio).abs().astype(np.float32)

    # ------------------------------------------------------------------ frequency deviation
    hz = col("DERMeasureAC[0].Hz")
    df["fe_hz_dev"] = (hz - 60.0).abs().astype(np.float32)

    # ------------------------------------------------------------------ mode flags
    mode = col("DERMeasureAC[0].DERMode")
    df["fe_mode_nonzero"] = (mode != 0).astype(np.float32)

    return df
