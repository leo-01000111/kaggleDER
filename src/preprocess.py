"""
preprocess.py
-------------
Loads raw train/test CSVs in chunks, cleans, encodes, and saves to parquet.
Uses float32 dtypes and streaming parquet writes to stay within RAM limits.

Usage:
    py src/preprocess.py
"""

import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data-source")
OUT_DIR = os.path.join(ROOT, "data-source")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_PARQUET = os.path.join(OUT_DIR, "train.parquet")
TEST_PARQUET = os.path.join(OUT_DIR, "test.parquet")

CHUNK_SIZE = 150_000  # rows per chunk — ~250MB per chunk at float32

CAT_COLS = [
    "common[0].Mn",
    "common[0].Md",
    "common[0].Opt",
    "common[0].Vr",
    "common[0].SN",
    "DERMeasureDC[0].Prt[0].IDStr",
    "DERMeasureDC[0].Prt[1].IDStr",
    # DeviceType removed from dataset (2026-03 leak fix)
]


def scan_header(path: str, sample_rows: int = 10_000):
    """
    Read a small sample to:
    - Get all column names
    - Identify always-null columns (null in >99.9% of sample)
    - Build dtype map (float32 for floats, keep ints and objects as-is)
    """
    sample = pd.read_csv(path, nrows=sample_rows)
    all_cols = sample.columns.tolist()

    # Columns to drop: >99.9% null in sample
    null_frac = sample.isnull().mean()
    drop_cols = set(null_frac[null_frac >= 0.999].index.tolist())
    drop_cols -= {"Id", "Label"}

    # Build dtype dict for memory-efficient reading
    dtype_map = {}
    for col in all_cols:
        if col in drop_cols or col in ("Id", "Label"):
            continue
        if col in CAT_COLS:
            dtype_map[col] = str  # keep as string, encode later
        elif sample[col].dtype == np.float64:
            dtype_map[col] = np.float32
        elif sample[col].dtype == np.int64:
            dtype_map[col] = np.int32
        # else: leave as default

    keep_cols = [c for c in all_cols if c not in drop_cols]
    print(f"  Total columns: {len(all_cols)}")
    print(f"  Dropping {len(drop_cols)} always-null columns")
    print(f"  Keeping {len(keep_cols)} columns")

    return keep_cols, drop_cols, dtype_map


def build_cat_mappings(path: str, cat_cols: list[str], keep_cols: list[str]) -> dict:
    """Build label-encoding maps from a full scan of cat columns (cheap)."""
    usecols = [c for c in cat_cols if c in keep_cols]
    if not usecols:
        return {}
    print(f"  Scanning categoricals: {usecols}")
    sample = pd.read_csv(path, usecols=usecols, dtype=str)
    mappings = {}
    for col in usecols:
        uniq = sorted(sample[col].dropna().unique())
        mappings[col] = {v: i for i, v in enumerate(uniq)}
        mappings[col]["__nan__"] = -1
    return mappings


def encode_chunk(chunk: pd.DataFrame, cat_mappings: dict) -> pd.DataFrame:
    for col, mapping in cat_mappings.items():
        if col not in chunk.columns:
            continue
        chunk[col] = chunk[col].map(lambda x: mapping.get(str(x), mapping.get("__nan__", -1))
                                    if pd.notna(x) else -1).astype(np.int8)
    return chunk


def process_csv(path: str, out_path: str, keep_cols: list[str],
                dtype_map: dict, cat_mappings: dict, has_label: bool):
    usecols = keep_cols  # already excludes drop_cols

    writer = None
    total_rows = 0

    reader = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype_map,
        chunksize=CHUNK_SIZE,
    )

    for i, chunk in enumerate(reader):
        chunk = encode_chunk(chunk, cat_mappings)

        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")

        writer.write_table(table)
        total_rows += len(chunk)

        if (i + 1) % 5 == 0:
            print(f"    ... processed {total_rows:,} rows")

    if writer:
        writer.close()

    print(f"  Done: {total_rows:,} rows written to {out_path}")


def main():
    # ------------------------------------------------------------------ scan
    print("Scanning train header and sample ...")
    keep_cols, drop_cols, dtype_map = scan_header(TRAIN_CSV)

    print("\nBuilding categorical encodings from train ...")
    cat_mappings = build_cat_mappings(TRAIN_CSV, CAT_COLS, keep_cols)
    for col, m in cat_mappings.items():
        print(f"  {col}: {len(m)-1} categories")

    # ------------------------------------------------------------------ train
    print(f"\nProcessing train ({TRAIN_CSV}) ...")
    process_csv(TRAIN_CSV, TRAIN_PARQUET, keep_cols, dtype_map, cat_mappings, has_label=True)

    # ------------------------------------------------------------------ test
    print(f"\nProcessing test ({TEST_CSV}) ...")
    # Test has no Label column — keep_cols for test excludes Label
    test_keep = [c for c in keep_cols if c != "Label"]
    process_csv(TEST_CSV, TEST_PARQUET, test_keep, dtype_map, cat_mappings, has_label=False)

    # ------------------------------------------------------------------ verify
    print("\nVerifying output ...")
    import pyarrow.parquet as pq
    train_meta = pq.read_metadata(TRAIN_PARQUET)
    test_meta = pq.read_metadata(TEST_PARQUET)
    print(f"  Train parquet: {train_meta.num_rows:,} rows, {train_meta.num_columns} cols")
    print(f"  Test parquet:  {test_meta.num_rows:,} rows, {test_meta.num_columns} cols")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
