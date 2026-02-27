#!/usr/bin/env python

import glob
import os
import pandas as pd

# Find all deconvolution coef files
files = sorted(glob.glob("SAMPLE_*/*_deconvolutionCoefs.csv"))

rows = []

for path in files:
    print(f"Reading {path}")
    df = pd.read_csv(path, index_col=0)

    # Sample ID from filename (before first "_deconvolutionCoefs")
    fname = os.path.basename(path)
    sample_id = fname.split("_deconvolutionCoefs")[0]

    # Keep only NNLS rows (e.g. "SAMPLE_100008_1-NNLS")
    nnls_df = df[df.index.str.contains("-NNLS")]
    if nnls_df.empty:
        # skip files that don't have NNLS rows
        continue

    for idx, row in nnls_df.iterrows():
        rec = {
            "sample_id": sample_id,
            "row_name": idx,
            "method": "NNLS",
        }
        # Add all coefficient columns (cell types + r + rmse)
        for col, val in row.items():
            rec[col] = val

        rows.append(rec)

# Combine into a single DataFrame
if not rows:
    raise SystemExit("No NNLS rows collected; check your file pattern and contents.")

big_df_nnls = pd.DataFrame(rows)

# Optional: drop exact duplicate rows (your file seems to repeat NNLS rows)
big_df_nnls = big_df_nnls.drop_duplicates()

# Optional: sort by sample_id
big_df_nnls = big_df_nnls.sort_values(["sample_id"])

# Save to CSV
out_path = "all_deconvolutionCoefs_with_nnls.csv"
big_df_nnls.to_csv(out_path, index=False)
print(f"Written {out_path} with shape {big_df_nnls.shape}")
