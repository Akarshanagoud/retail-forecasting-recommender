"""Rolling-origin backtesting.

A single train/test split on a time series answers one question: how did this
model do on that particular fortnight? It cannot distinguish a good model from
a model that got a quiet December, and it is the reason so many forecasting
projects ship a number nobody can reproduce next quarter.

Rolling-origin evaluation refits at successive cutoffs and scores each
horizon, so the reported accuracy is an average over many origins and the
variance across them is itself visible. That variance is usually the more
useful number: a model with 12% WAPE and 2% spread is deployable; a model with
10% WAPE and 9% spread is a coin flip.

Two rules are enforced structurally rather than by care:

* **Training data ends strictly before the cutoff.** The split is by index into
  a chronologically sorted series; no shuffle is possible.
* **Only known-in-advance features are used for the horizon.** Price and
  promotion plans are legitimately known; target lags reaching into the horizon
  are not, and the feature builder refuses to produce them.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .data import SalesRecord
from .features import FeatureConfig, build_features
from .metrics import ForecastScore, aggregate_scores, score_forecast


@dataclass
class BacktestConfig:
    #: Days forecast at each origin.
    horizon: int = 14
    #: Days between successive origins.
    step: int = 14
    #: Minimum training days before the first origin.
    min_train_days: int = 180
    #: Cap on the number of origins, newest kept.
    max_origins: int = 8
    season_length: int = 7

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "step": self.step,
            "min_train_days": self.min_train_days,
            "max_origins": self.max_origins,
        }


@dataclass
class OriginResult:
    """One refit-and-forecast at a single cutoff."""

    cutoff: date
    train_size: int
    score: ForecastScore
    fit_ms: float
    predict_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cutoff": self.cutoff.isoformat(),
            "train_size": self.train_size,
            "fit_ms": round(self.fit_ms, 2),
            "predict_ms": round(self.predict_ms, 2),
            **self.score.to_dict(),
        }


@dataclass
class BacktestResult:
    model: str
    sku: str
    store: str
    origins: list[OriginResult] = field(default_factory=list)
    error: str | None = None

    @property
    def scores(self) -> list[ForecastScore]:
        return [o.score for o in self.origins]

    @property
    def mean_wape(self) -> float | None:
        values = [o.score.wape for o in self.origins if o.score.wape != float("inf")]
        return statistics.fmean(values) if values else None

    @property
    def wape_stdev(self) -> float | None:
        """Spread across origins. Often more informative than the mean."""
        values = [o.score.wape for o in self.origins if o.score.wape != float("inf")]
        return statistics.stdev(values) if len(values) > 1 else 0.0

    @property
    def mean_mase(self) -> float | None:
        values = [o.score.mase for o in self.origins if o.score.mase is not None]
        return statistics.fmean(values) if values else None

    def to_dict(self, include_origins: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "sku": self.sku,
            "store": self.store,
            "origins": len(self.origins),
            "mean_wape": round(self.mean_wape, 4) if self.mean_wape is not None else None,
            "wape_stdev": round(self.wape_stdev, 4)
            if self.wape_stdev is not None
            else None,
            "mean_mase": round(self.mean_mase, 4) if self.mean_mase is not None else None,
            "error": self.error,
            **aggregate_scores(self.scores),
        }
        if include_origins:
            payload["per_origin"] = [o.to_dict() for o in self.origins]
        return payload


def rolling_origins(
    n: int, config: BacktestConfig
) -> list[int]:
    """Indices at which to cut. Each leaves a full horizon of held-out data."""
    origins: list[int] = []
    cutoff = config.min_train_days
    while cutoff + config.horizon <= n:
        origins.append(cutoff)
        cutoff += config.step
    return origins[-config.max_origins :]


def backtest_series(
    records: list[SalesRecord],
    model_factory: Any,
    config: BacktestConfig | None = None,
    feature_config: FeatureConfig | None = None,
) -> BacktestResult:
    """Backtest one model on one SKU-store series.

    ``model_factory`` is a zero-argument callable returning a fresh
    :class:`Forecaster`. A factory rather than an instance because each origin
    must refit from scratch: reusing a fitted model across origins leaks the
    later data into the earlier evaluation.
    """
    settings = config or BacktestConfig()
    features = feature_config or FeatureConfig()
    ordered = sorted(records, key=lambda r: r.date)
    sku = ordered[0].sku if ordered else "?"
    store = ordered[0].store if ordered else "?"

    probe = model_factory()
    result = BacktestResult(model=probe.name, sku=sku, store=store)

    if len(ordered) < settings.min_train_days + settings.horizon:
        result.error = (
            f"series has {len(ordered)} days, needs at least "
            f"{settings.min_train_days + settings.horizon}"
        )
        return result

    targets = [r.units for r in ordered]
    matrix = build_features(ordered, features) if probe.uses_features else None

    for cutoff in rolling_origins(len(ordered), settings):
        model = model_factory()
        train_y = targets[:cutoff]
        actual = targets[cutoff : cutoff + settings.horizon]

        X_train: list[list[float]] | None = None
        X_future: list[list[float]] | None = None
        if probe.uses_features and matrix is not None:
            # Feature rows start after the warmup, so shift the cutoff to match.
            offset = matrix.warmup
            train_end = cutoff - offset
            if train_end <= 10:
                continue
            X_train = matrix.X[:train_end]
            train_y = matrix.y[:train_end]
            X_future = matrix.X[train_end : train_end + settings.horizon]
            actual = matrix.y[train_end : train_end + settings.horizon]
            if len(X_future) < settings.horizon:
                continue

        started = time.perf_counter()
        try:
            model.fit(train_y, X_train)
        except (ValueError, ImportError) as exc:
            result.error = f"fit failed at {ordered[cutoff].date}: {exc}"
            return result
        fit_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        predicted = model.predict(settings.horizon, X_future)
        predict_ms = (time.perf_counter() - started) * 1000

        result.origins.append(
            OriginResult(
                cutoff=ordered[cutoff].date,
                train_size=len(train_y),
                score=score_forecast(
                    actual, predicted, history=train_y,
                    season_length=settings.season_length,
                ),
                fit_ms=fit_ms,
                predict_ms=predict_ms,
            )
        )

    if not result.origins and result.error is None:
        result.error = "no usable origins"
    return result


def backtest_dataset(
    series_by_key: dict[tuple[str, str], list[SalesRecord]],
    model_factories: dict[str, Any],
    config: BacktestConfig | None = None,
    feature_config: FeatureConfig | None = None,
    limit: int | None = None,
) -> dict[str, list[BacktestResult]]:
    """Backtest several models across several series."""
    settings = config or BacktestConfig()
    keys = sorted(series_by_key)
    if limit:
        keys = keys[:limit]

    out: dict[str, list[BacktestResult]] = {}
    for name, factory in model_factories.items():
        out[name] = [
            backtest_series(series_by_key[key], factory, settings, feature_config)
            for key in keys
        ]
    return out


@dataclass
class ModelComparison:
    model: str
    series: int
    wape: float | None
    wape_stdev: float | None
    mase: float | None
    beats_baseline: int
    failures: int
    mean_fit_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "series": self.series,
            "wape": round(self.wape, 4) if self.wape is not None else None,
            "wape_stdev": round(self.wape_stdev, 4)
            if self.wape_stdev is not None
            else None,
            "mase": round(self.mase, 4) if self.mase is not None else None,
            "beats_seasonal_naive": self.beats_baseline,
            "failures": self.failures,
            "mean_fit_ms": round(self.mean_fit_ms, 2),
        }


def compare_models(
    results: dict[str, list[BacktestResult]]
) -> list[ModelComparison]:
    """Roll backtests up into one row per model, best WAPE first."""
    comparisons: list[ModelComparison] = []
    for model, runs in results.items():
        usable = [r for r in runs if r.origins]
        every_score = [score for r in usable for score in r.scores]
        aggregate = aggregate_scores(every_score)
        fit_times = [o.fit_ms for r in usable for o in r.origins]

        comparisons.append(
            ModelComparison(
                model=model,
                series=len(usable),
                wape=aggregate.get("wape"),
                wape_stdev=statistics.fmean([r.wape_stdev or 0.0 for r in usable])
                if usable
                else None,
                mase=aggregate.get("mase"),
                beats_baseline=sum(1 for s in every_score if s.beats_seasonal_naive),
                failures=len(runs) - len(usable),
                mean_fit_ms=statistics.fmean(fit_times) if fit_times else 0.0,
            )
        )

    comparisons.sort(key=lambda c: (c.wape if c.wape is not None else float("inf")))
    return comparisons


def render_comparison(comparisons: list[ModelComparison]) -> str:
    header = (
        f"{'MODEL':<28}{'SERIES':>7}{'WAPE':>9}{'STDEV':>9}{'MASE':>8}"
        f"{'BEATS SN':>10}{'FIT ms':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in comparisons:
        wape = f"{row.wape:.4f}" if row.wape is not None else "-"
        stdev = f"{row.wape_stdev:.4f}" if row.wape_stdev is not None else "-"
        mase = f"{row.mase:.3f}" if row.mase is not None else "-"
        lines.append(
            f"{row.model:<28}{row.series:>7}{wape:>9}{stdev:>9}{mase:>8}"
            f"{row.beats_baseline:>10}{row.mean_fit_ms:>9.2f}"
        )
    return "\n".join(lines)
