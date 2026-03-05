"""
sample_train.py
---------------
Creates a stratified sample of train.parquet without loading the full file.
Reads one row-group at a time (~370MB each) and samples 35% per class.
Result: ~826k rows saved to train_sample.parquet.

Run this once before train.py if you have limited RAM.

Usage:
    py src/sample_train.py
"""

import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data-source")
TRAIN_PARQUET = os.path.join(DATA_DIR, "train.parquet")
SAMPLE_PARQUET = os.path.join(DATA_DIR, "train_sample.parquet")

SAMPLE_FRAC = 0.50
SEED = 42


def main():
    pf = pq.ParquetFile(TRAIN_PARQUET)
    n_groups = pf.num_row_groups
    print(f"Source: {TRAIN_PARQUET}")
    print(f"Row groups: {n_groups} | Sample fraction: {SAMPLE_FRAC}")
    print()

    writer = None
    total_out = 0

    for i in range(n_groups):
        chunk = pf.read_row_group(i).to_pandas()
        sampled = (
            chunk.groupby("Label", group_keys=False)
                 .apply(lambda g: g.sample(frac=SAMPLE_FRAC, random_state=SEED))
        )
        table = pa.Table.from_pandas(sampled, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(SAMPLE_PARQUET, table.schema, compression="snappy")
        writer.write_table(table)

        total_out += len(sampled)
        print(f"  Group {i+1:>2}/{n_groups}: {len(sampled):>6} rows sampled "
              f"(label balance: {sampled['Label'].mean():.3f})")

    if writer:
        writer.close()

    print(f"\nDone: {total_out:,} rows → {SAMPLE_PARQUET}")
    print(f"Estimated RAM to load: {total_out * 548 * 4 / 1e9:.2f} GB (float32)")


if __name__ == "__main__":
    main()
