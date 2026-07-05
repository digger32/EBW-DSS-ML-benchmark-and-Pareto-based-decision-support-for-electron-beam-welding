"""
Resume-safe CSV storage for the S-7 experimental campaign.

The pipeline writes one row per (generator, n_synth, ablation, model,
optimiser, fold) configuration to ``results.csv`` and one row per
(generator, n_synth, ablation, model, optimiser) configuration to
``results_aggregated.csv``. A second file ``done.csv`` tracks which
(generator, n_synth, ablation, model, optimiser) combinations have been
completed so the pipeline can resume after interruption.

v2 (schema-stable aggregated writer)
------------------------------------
``append_aggregated_row`` previously wrote the header once from the first
row and then appended each subsequent row as-is. Aggregated rows do not all
share the same schema: a timeout config that completed 0 folds carries only
the 11 base columns, whereas an ok/partial config additionally carries the
per-metric ``*_mean``/``*_std`` columns. Appending heterogeneous rows under a
header fixed by the first row produced a *ragged* CSV that pandas refused to
read ("Expected N fields ... saw M").

The fix keeps an in-memory canonical field list (``self._agg_fields``) and:
  * pads narrow rows to the canonical fields (missing cells -> empty);
  * drops any stray keys beyond the canonical fields;
  * widens the file exactly once if a genuinely new column appears, by
    rewriting it with the unioned header (rare: at most once per new schema).
Every row therefore has identical width and the file is always rectangular.
The same logic is applied to the per-fold ``results.csv``.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


KEY_COLS = ("generator", "n_synth", "ablation", "model", "optimiser")


def _read_header(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    with open(path, newline="") as f:
        try:
            return next(csv.reader(f))
        except StopIteration:
            return None


def _read_rows_robust(path: Path) -> tuple[list[str], list[list]]:
    """Read a possibly-ragged CSV; align rows to header (truncate/pad)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        w = len(header)
        rows = []
        for r in reader:
            if not r:
                continue
            if len(r) > w:
                r = r[:w]
            elif len(r) < w:
                r = r + [""] * (w - len(r))
            rows.append(r)
    return header, rows


class ResultStore:
    """Append-only store with a manifest of completed configurations."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.out_dir / "results.csv"
        self.agg_path = self.out_dir / "results_aggregated.csv"
        self.done_path = self.out_dir / "done.csv"
        self._done: set[tuple] = self._load_done()
        # canonical field lists, seeded from any existing files (resume-safe)
        self._agg_fields: list[str] | None = _read_header(self.agg_path)
        self._fold_fields: list[str] | None = _read_header(self.results_path)

    def _load_done(self) -> set[tuple]:
        if not self.done_path.exists():
            return set()
        out = set()
        with open(self.done_path) as f:
            r = csv.DictReader(f)
            for row in r:
                out.add(tuple(row[c] for c in KEY_COLS))
        return out

    def is_done(self, key: dict) -> bool:
        return tuple(str(key[c]) for c in KEY_COLS) in self._done

    # -- schema-stable append helpers ---------------------------------------
    @staticmethod
    def _widen_file(path: Path, new_fields: list[str]) -> None:
        """Rewrite an existing CSV under a wider (superset) header."""
        old_header, rows = _read_rows_robust(path)
        idx = {c: i for i, c in enumerate(old_header)}
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(new_fields)
            for r in rows:
                w.writerow([r[idx[c]] if c in idx and idx[c] < len(r) else ""
                            for c in new_fields])

    def _append_row_stable(self, path: Path, fields_attr: str, row: dict) -> None:
        fields = getattr(self, fields_attr)
        if fields is None:
            fields = list(row.keys())
            setattr(self, fields_attr, fields)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerow(row)
            return
        new_keys = [k for k in row.keys() if k not in fields]
        if new_keys:                              # genuinely new columns -> widen once
            fields = fields + new_keys
            setattr(self, fields_attr, fields)
            self._widen_file(path, fields)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
            w.writerow(row)

    def append_fold_rows(self, rows: list[dict]) -> None:
        for row in rows:
            self._append_row_stable(self.results_path, "_fold_fields", row)

    def append_aggregated_row(self, row: dict) -> None:
        self._append_row_stable(self.agg_path, "_agg_fields", row)

    def mark_done(self, key: dict) -> None:
        write_header = not self.done_path.exists()
        with open(self.done_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(KEY_COLS)
            w.writerow([str(key[c]) for c in KEY_COLS])
        self._done.add(tuple(str(key[c]) for c in KEY_COLS))

    def aggregated_df(self) -> pd.DataFrame:
        if not self.agg_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.agg_path)

    def results_df(self) -> pd.DataFrame:
        if not self.results_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.results_path)
