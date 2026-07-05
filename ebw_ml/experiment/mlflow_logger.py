"""
Thin MLflow wrapper that becomes a no-op if MLflow is unavailable or
explicitly disabled.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any


class _NullRun:
    def log_params(self, params: dict) -> None: ...
    def log_metrics(self, metrics: dict, step: int | None = None) -> None: ...
    def log_text(self, text: str, name: str) -> None: ...
    def set_tags(self, tags: dict) -> None: ...


class MLflowLogger:
    """Wrap MLflow with a graceful fallback when MLflow is not installed."""

    def __init__(self, tracking_uri: str | None = None,
                 experiment_name: str = "ebw_s7", enabled: bool = True) -> None:
        self.enabled = enabled
        self._mlflow = None
        if not enabled:
            return
        try:
            import mlflow
            self._mlflow = mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
        except ImportError:
            self.enabled = False

    @contextmanager
    def run(self, name: str, tags: dict | None = None):
        if not self.enabled:
            yield _NullRun()
            return
        with self._mlflow.start_run(run_name=name) as r:
            if tags:
                self._mlflow.set_tags(tags)

            class _R:
                def log_params(_, params):
                    safe = {k: str(v) if not isinstance(v, (int, float, bool, str)) else v
                            for k, v in params.items()}
                    self._mlflow.log_params(safe)

                def log_metrics(_, metrics, step=None):
                    safe = {k: float(v) for k, v in metrics.items()
                            if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v))}
                    self._mlflow.log_metrics(safe, step=step)

                def log_text(_, text, name):
                    self._mlflow.log_text(text, name)

                def set_tags(_, tags):
                    self._mlflow.set_tags(tags)

            yield _R()
