"""
Merge several S-7 run directories into one consolidated output directory.

Use case: the S-7 grid is sliced across multiple processes (each with its
own --out-dir) so that the multi-core host runs them in parallel without
contention on done.csv. When all processes finish, this script combines
their CSVs into a single canonical output directory whose layout is
identical to a single-process run.

Usage:
    python merge_runs.py --inputs runs/s7_p1 runs/s7_p2 runs/s7_p3 \\
                          --output runs/s7_merged

Behaviour:
    * Concatenates results.csv, results_aggregated.csv and done.csv.
    * De-duplicates by (generator, n_synth, ablation, model, optimiser).
      If the same key appears in two input runs, the row with the smaller
      total_elapsed_s wins (i.e. the faster successful repeat is kept).
    * Writes the consolidated CSVs to --output.
    * Prints a coverage summary.

ROBUSTNESS NOTE (v2):
    Some results_aggregated.csv / results.csv files are *ragged*: the
    header is written once from the first appended row, but later rows can
    carry a different number of fields (timeout rows with 0 completed folds
    have only the 11 base columns, whereas partial/ok rows additionally
    carry the per-metric *_mean/*_std columns). A plain pandas read then
    fails with "Expected N fields ... saw M". The reader below tolerates
    this: every data row is aligned to its file header (overflow fields are
    dropped, short rows are padded with NA), and frames from different files
    are unioned on column name by pandas.concat. The base columns
    (generator, n_synth, ablation, model, optimiser, model_family,
    optimiser_family, n_folds_used, n_folds_ok, status, total_elapsed_s)
    are common to every file and therefore always survive, which is all the
    merge and the coverage summary require.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

KEY_COLS = ["generator", "n_synth", "ablation", "model", "optimiser"]


def read_csv_robust(p: Path) -> pd.DataFrame:
    """Read a possibly-ragged CSV: align every row to the file's header.

    Overflow fields (more than the header width) are truncated; short rows
    are padded with None. Returns an empty frame for an empty/headerless file.
    """
    with open(p, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame()
        width = len(header)
        rows = []
        for r in reader:
            if not r:
                continue
            if len(r) > width:
                r = r[:width]
            elif len(r) < width:
                r = r + [None] * (width - len(r))
            rows.append(r)
    df = pd.DataFrame(rows, columns=header)
    # normalise empty strings to NA so numeric coercion below behaves
    return df.replace("", pd.NA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more S-7 run directories to merge.")
    ap.add_argument("--output", required=True,
                    help="Output directory for the merged CSVs.")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Merge results_aggregated.csv
    agg_frames = []
    for inp in args.inputs:
        p = Path(inp) / "results_aggregated.csv"
        if p.exists():
            df = read_csv_robust(p)
            if df.empty:
                print(f"  {inp}/results_aggregated.csv: empty — skipping")
                continue
            df["_source"] = inp
            agg_frames.append(df)
            print(f"  {inp}/results_aggregated.csv: {len(df)} rows, {df.shape[1]-1} cols")
        else:
            print(f"  {inp}/results_aggregated.csv: MISSING — skipping")
    if not agg_frames:
        print("No input files found. Exiting.")
        return
    agg = pd.concat(agg_frames, ignore_index=True)
    print(f"\nMerged aggregated total: {len(agg)} rows (with possible duplicates)")

    # De-dup by (key) — keep the row with the smallest total_elapsed_s as the
    # canonical one. Coerce elapsed to numeric (robust reader yields strings).
    if "total_elapsed_s" in agg.columns:
        agg["total_elapsed_s"] = pd.to_numeric(agg["total_elapsed_s"], errors="coerce")
        agg = agg.sort_values("total_elapsed_s", kind="stable", na_position="last")
    agg_dedup = (agg.drop_duplicates(subset=KEY_COLS, keep="first")
                    .drop(columns=["_source"], errors="ignore"))
    print(f"After de-dup on key: {len(agg_dedup)} rows")
    agg_dedup.to_csv(out_dir / "results_aggregated.csv", index=False)

    # 2. Merge results.csv (per-fold rows)
    fold_frames = []
    for inp in args.inputs:
        p = Path(inp) / "results.csv"
        if p.exists():
            df = read_csv_robust(p)
            if not df.empty:
                fold_frames.append(df)
    if fold_frames:
        folds = pd.concat(fold_frames, ignore_index=True)
        kept_keys = set(agg_dedup[KEY_COLS].astype(str).agg("|".join, axis=1))
        folds["_k"] = folds[KEY_COLS].astype(str).agg("|".join, axis=1)
        folds = folds[folds["_k"].isin(kept_keys)].drop(columns=["_k"])
        folds.to_csv(out_dir / "results.csv", index=False)
        print(f"Merged per-fold: {len(folds)} rows")

    # 3. Rebuild done.csv from the deduplicated aggregated keys
    done = agg_dedup[KEY_COLS].copy()
    done.to_csv(out_dir / "done.csv", index=False)
    print(f"Wrote {out_dir / 'done.csv'} with {len(done)} unique keys")

    # 4. Coverage summary
    print("\n=== Coverage summary ===")
    for axis in ["generator", "ablation", "model", "optimiser"]:
        if axis in agg_dedup.columns:
            c = agg_dedup[axis].value_counts()
            print(f"  {axis} ({len(c)}): {dict(c)}")
    if "status" in agg_dedup.columns:
        print(f"  status: {dict(agg_dedup['status'].value_counts(dropna=False))}")


if __name__ == "__main__":
    main()
