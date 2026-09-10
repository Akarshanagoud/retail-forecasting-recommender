"""Experiment tracking and the model registry.

A local JSON-backed tracker, with an MLflow adapter when a server is available.
The local one is the default deliberately: an experiment tracker that requires
infrastructure to be running is an experiment tracker that gets skipped during
the exploratory phase, which is exactly when the results are least reproducible
and most worth recording.

What is recorded is chosen so a run can actually be reproduced: parameters,
metrics, the data fingerprint, the git commit, and the library versions. A run
that records only metrics tells you a number was achieved once and gives you no
way to get it back.

The registry adds promotion with a **gate**: a candidate is only promoted to
production if it beats the incumbent by a margin. Without a margin, normal
run-to-run variance promotes a new model roughly half the time, and the
production model becomes a random walk.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.strip() or None
    except Exception:  # noqa: BLE001 - tracking must never break a training run
        return None


def data_fingerprint(rows: list[Any]) -> str:
    """Stable hash of a dataset, so a run records what it was trained on."""
    hasher = hashlib.sha256()
    hasher.update(str(len(rows)).encode())
    for row in rows[:1000]:
        hasher.update(repr(row).encode())
    return hasher.hexdigest()[:16]


@dataclass
class Run:
    run_id: str
    experiment: str
    started_at: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    duration_s: float | None = None
    status: str = "running"
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 3) if self.duration_s else None,
            "status": self.status,
            "params": self.params,
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "tags": self.tags,
            "artifacts": self.artifacts,
            "environment": self.environment,
        }


class ExperimentTracker:
    """Append-only JSONL run log with a queryable index."""

    def __init__(self, path: str | Path = "mlruns/runs.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._active: Run | None = None
        self._started: float = 0.0

    def start_run(
        self, experiment: str, params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> Run:
        self._active = Run(
            run_id=uuid.uuid4().hex[:12],
            experiment=experiment,
            started_at=datetime.now(timezone.utc).isoformat(),
            params=dict(params or {}),
            tags=dict(tags or {}),
            environment={
                "python": sys.version.split()[0],
                "platform": platform.system(),
                "git_commit": _git_commit(),
            },
        )
        self._started = time.perf_counter()
        return self._active

    def log_params(self, **params: Any) -> None:
        if self._active is not None:
            self._active.params.update(params)

    def log_metrics(self, **metrics: float) -> None:
        if self._active is not None:
            self._active.metrics.update(
                {k: float(v) for k, v in metrics.items() if v is not None}
            )

    def log_artifact(self, name: str, path: str) -> None:
        if self._active is not None:
            self._active.artifacts[name] = str(path)

    def end_run(self, status: str = "finished") -> Run | None:
        if self._active is None:
            return None
        run = self._active
        run.duration_s = time.perf_counter() - self._started
        run.status = status
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(run.to_dict()) + "\n")
        self._active = None
        return run

    def runs(self, experiment: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if experiment:
            rows = [r for r in rows if r["experiment"] == experiment]
        return rows

    def best_run(
        self, experiment: str, metric: str, lower_is_better: bool = True
    ) -> dict[str, Any] | None:
        candidates = [r for r in self.runs(experiment) if metric in r["metrics"]]
        if not candidates:
            return None
        return (min if lower_is_better else max)(
            candidates, key=lambda r: r["metrics"][metric]
        )

    def leaderboard(
        self, experiment: str, metric: str, lower_is_better: bool = True, limit: int = 10
    ) -> list[dict[str, Any]]:
        candidates = [r for r in self.runs(experiment) if metric in r["metrics"]]
        candidates.sort(key=lambda r: r["metrics"][metric], reverse=not lower_is_better)
        return candidates[:limit]


@dataclass
class RegisteredModel:
    name: str
    version: int
    run_id: str
    stage: str
    metric_name: str
    metric_value: float
    registered_at: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "run_id": self.run_id,
            "stage": self.stage,
            "metric": {self.metric_name: round(self.metric_value, 6)},
            "registered_at": self.registered_at,
            "params": self.params,
        }


class ModelRegistry:
    """Versioned models with gated promotion."""

    def __init__(self, path: str | Path = "mlruns/registry.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, list[RegisteredModel]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for name, versions in data.items():
            self._models[name] = [
                RegisteredModel(
                    name=v["name"], version=v["version"], run_id=v["run_id"],
                    stage=v["stage"], metric_name=next(iter(v["metric"])),
                    metric_value=next(iter(v["metric"].values())),
                    registered_at=v["registered_at"], params=v.get("params", {}),
                )
                for v in versions
            ]

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    name: [v.to_dict() for v in versions]
                    for name, versions in self._models.items()
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def register(
        self, name: str, run_id: str, metric_name: str, metric_value: float,
        params: dict[str, Any] | None = None,
    ) -> RegisteredModel:
        versions = self._models.setdefault(name, [])
        model = RegisteredModel(
            name=name,
            version=len(versions) + 1,
            run_id=run_id,
            stage="staging",
            metric_name=metric_name,
            metric_value=metric_value,
            registered_at=datetime.now(timezone.utc).isoformat(),
            params=dict(params or {}),
        )
        versions.append(model)
        self._save()
        return model

    def production(self, name: str) -> RegisteredModel | None:
        return next(
            (v for v in self._models.get(name, []) if v.stage == "production"), None
        )

    def versions(self, name: str) -> list[RegisteredModel]:
        return list(self._models.get(name, []))

    def promote(
        self,
        name: str,
        version: int,
        min_improvement: float = 0.02,
        lower_is_better: bool = True,
    ) -> tuple[bool, str]:
        """Promote to production only on a material improvement.

        The margin is the point. Without it, ordinary run-to-run variance
        promotes a new model about half the time and the production model
        becomes a random walk with a changelog.
        """
        candidates = [v for v in self._models.get(name, []) if v.version == version]
        if not candidates:
            return False, f"version {version} of {name!r} is not registered"
        candidate = candidates[0]

        incumbent = self.production(name)
        if incumbent is None:
            candidate.stage = "production"
            self._save()
            return True, f"promoted v{version}: no incumbent"

        if lower_is_better:
            improvement = (incumbent.metric_value - candidate.metric_value) / max(
                abs(incumbent.metric_value), 1e-9
            )
        else:
            improvement = (candidate.metric_value - incumbent.metric_value) / max(
                abs(incumbent.metric_value), 1e-9
            )

        if improvement < min_improvement:
            return False, (
                f"not promoted: v{version} improves {candidate.metric_name} by "
                f"{improvement:.1%}, below the {min_improvement:.0%} gate "
                f"(incumbent v{incumbent.version})"
            )

        incumbent.stage = "archived"
        candidate.stage = "production"
        self._save()
        return True, (
            f"promoted v{version}: {improvement:.1%} improvement in "
            f"{candidate.metric_name} over v{incumbent.version}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: [v.to_dict() for v in versions]
            for name, versions in self._models.items()
        }


class MLflowTracker:
    """Optional MLflow adapter.

        pip install mlflow
        MLflowTracker(tracking_uri="file:./mlruns")

    The same interface as :class:`ExperimentTracker`, so switching is a
    constructor change. MLflow is free and runs against a local file store with
    no server, which is worth knowing before standing up infrastructure.
    """

    def __init__(
        self, experiment: str = "retail", tracking_uri: str = "file:./mlruns"
    ) -> None:
        try:
            import mlflow
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("MLflowTracker needs: pip install mlflow") from exc
        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self.experiment = experiment

    def start_run(
        self, experiment: str, params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> Any:
        run = self._mlflow.start_run(run_name=experiment, tags=tags)
        if params:
            self._mlflow.log_params(params)
        return run

    def log_params(self, **params: Any) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, **metrics: float) -> None:
        self._mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})

    def log_artifact(self, name: str, path: str) -> None:
        self._mlflow.log_artifact(path)

    def end_run(self, status: str = "finished") -> None:
        self._mlflow.end_run()
