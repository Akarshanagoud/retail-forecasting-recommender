"""HTTP service and web UI: a demand-planning studio over the platform.

    pip install -e ".[server]"
    retailml ui                    # opens http://127.0.0.1:8769/ui/

Everything comes from the synthetic generator with a fixed seed. Rolling-origin
backtests for every model and series, both recommenders, and a model registry
with a production baseline are computed in the background at start-up, and the
UI shows their progress; promotion in the browser goes through the same gate as
the Airflow DAG.
"""
from __future__ import annotations

import math
import statistics
import tempfile
import threading
import time
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .backtest import BacktestConfig, backtest_series, compare_models, rolling_origins
from .data import SalesRecord, generate_dataset
from .drift import DriftReport, check_feature_drift, check_performance_drift, columns_from_matrix
from .features import build_features, future_features
from .forecasting import (
    EnsembleForecaster,
    HoltWinters,
    Naive,
    RidgeRegression,
    SeasonalMovingAverage,
    SeasonalNaive,
)
from .metrics import score_forecast
from .recommender import ItemItemRecommender, PopularityRecommender, temporal_split
from .recommender import evaluate as evaluate_reco
from .retraining import should_retrain
from .tracking import ExperimentTracker, ModelRegistry

WEB_DIR = Path(__file__).parent / "web"
REGISTRY_NAME = "demand_forecast"
MODELS: dict[str, Any] = {
    "seasonal_naive": SeasonalNaive,
    "naive": Naive,
    "seasonal_moving_average": SeasonalMovingAverage,
    "holt_winters": HoltWinters,
    "ridge": lambda: RidgeRegression(alpha=5.0),
    "ensemble": lambda: EnsembleForecaster([SeasonalNaive(), RidgeRegression(alpha=5.0)]),
}
LABELS = {
    "seasonal_naive": "Seasonal naive",
    "naive": "Naive",
    "seasonal_moving_average": "Seasonal moving average",
    "holt_winters": "Holt-Winters",
    "ridge": "Ridge regression",
    "ensemble": "Ensemble",
}


class TrainRequest(BaseModel):
    model: str


class PromoteRequest(BaseModel):
    version: int


def _check_model(name: str) -> None:
    if name not in MODELS:
        raise HTTPException(status_code=422, detail=f"unknown model {name!r}")


def _finite(value: float | None) -> float | None:
    """JSON has no infinity: a WAPE over an all-zero window is undefined, not huge."""
    return round(value, 4) if value is not None and math.isfinite(value) else None


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def forecast_at(records: list[SalesRecord], factory: Any, cutoff: int, horizon: int, matrix: Any = None) -> list[float]:
    """Refit on everything before ``cutoff`` and forecast ``horizon`` days, as the backtest does."""
    model = factory()
    targets = [r.units for r in records]
    if not model.uses_features:
        return model.fit(targets[:cutoff]).predict(horizon)
    matrix = matrix or build_features(records)
    train_end = cutoff - matrix.warmup
    model.fit(matrix.y[:train_end], matrix.X[:train_end])
    return model.predict(horizon, matrix.X[train_end : train_end + horizon])


def forecast_future(records: list[SalesRecord], factory: Any, horizon: int, matrix: Any = None) -> list[float]:
    model = factory()
    targets = [r.units for r in records]
    if not model.uses_features:
        return model.fit(targets).predict(horizon)
    matrix = matrix or build_features(records)
    last = records[-1]
    # Planned promotions are not known for the future, so the plan assumes none.
    future = [
        SalesRecord(date=last.date + timedelta(days=i + 1), sku=last.sku, store=last.store, category=last.category, units=0.0, price=last.price, on_promotion=False)
        for i in range(horizon)
    ]
    model.fit(matrix.y, matrix.X)
    return model.predict(horizon, future_features(records, future))


class Studio:
    """Everything the UI shows, computed once from a seeded dataset.

    The dataset is generated immediately; backtests and recommenders are
    computed by :meth:`warm`, in a background thread when ``background`` is set,
    and :attr:`progress` reports how far it has got.
    """

    def __init__(
        self,
        n_skus: int = 10,
        n_stores: int = 2,
        days: int = 730,
        seed: int = 7,
        reco_customers: int = 1200,
        workspace: Path | None = None,
        background: bool = False,
    ) -> None:
        self.started = time.perf_counter()
        self.dataset = generate_dataset(n_skus=n_skus, n_stores=n_stores, days=days, n_customers=800, seed=seed)
        self.specs = {(s.sku, s.store): s for s in self.dataset.specs}
        self.series = {key: sorted(self.dataset.series(*key), key=lambda r: r.date) for key in self.dataset.series_keys()}
        self.config = BacktestConfig(horizon=14, step=28, max_origins=6)
        self.reco_customers = reco_customers
        self.workspace = workspace
        self.progress = {"done": 0, "total": len(self.series) * len(MODELS) + 1, "stage": "starting"}
        self.ready = threading.Event()
        self.error: str | None = None
        self.startup_seconds: float | None = None
        self._views: dict[tuple[str, str, str], dict[str, Any]] = {}
        if background:
            threading.Thread(target=self.warm, name="retailml-warmup", daemon=True).start()
        else:
            self.warm()

    def warm(self) -> None:
        try:
            self._warm()
        except Exception as exc:  # surfaced through /health instead of a silently dead thread
            self.error = f"{type(exc).__name__}: {exc}"
            raise

    def _warm(self) -> None:
        results: dict[str, list[Any]] = {}
        for name, factory in MODELS.items():
            self.progress["stage"] = f"backtesting {LABELS[name]}"
            results[name] = []
            for key in sorted(self.series):
                results[name].append(backtest_series(self.series[key], factory, self.config))
                self.progress["done"] += 1
        self.results = results
        self.comparisons = compare_models(results)

        self.progress["stage"] = "fitting recommenders"
        reco = generate_dataset(n_skus=40, n_stores=2, days=400, n_customers=self.reco_customers, seed=11)
        self.reco_categories = {s.sku: s.category for s in reco.specs}
        train, test = temporal_split(reco.interactions, 0.2)
        catalogue = {i.sku for i in reco.interactions}
        self.cf = ItemItemRecommender().fit(train)
        self.popularity = PopularityRecommender().fit(train)
        self.reco_scores = {
            "item-item CF": evaluate_reco(self.cf, train, test, k=10, catalogue=catalogue).to_dict(),
            "popularity": evaluate_reco(self.popularity, train, test, k=10, catalogue=catalogue).to_dict(),
        }
        self.purchases: dict[str, Counter[str]] = defaultdict(Counter)
        for interaction in train:
            self.purchases[interaction.customer][interaction.sku] += interaction.quantity

        root = self.workspace or Path(tempfile.mkdtemp(prefix="retailml-"))
        self.tracker = ExperimentTracker(root / "runs.jsonl")
        self.registry = ModelRegistry(root / "registry.json")
        baseline = self.train("seasonal_naive")
        self.registry.promote(REGISTRY_NAME, baseline["version"])
        self.progress["done"] += 1
        self.progress["stage"] = "ready"
        self.startup_seconds = round(time.perf_counter() - self.started, 2)
        self.ready.set()

    def require_ready(self) -> None:
        if self.error:
            raise HTTPException(status_code=500, detail=f"start-up failed: {self.error}")
        if not self.ready.is_set():
            raise HTTPException(status_code=503, detail="backtests are still running", headers={"Retry-After": "2"})

    # -- forecasting -----------------------------------------------------------
    def _result(self, model: str, key: tuple[str, str]) -> Any:
        return next(r for r in self.results[model] if (r.sku, r.store) == key)

    def overview(self) -> dict[str, Any]:
        best_by_series = {}
        for key in self.series:
            scored = [(m, self._result(m, key).mean_wape) for m in MODELS]
            scored = [(m, w) for m, w in scored if w is not None]
            best_by_series[key] = min(scored, key=lambda mw: mw[1]) if scored else (None, None)
        summary = self.dataset.summary()
        first = min(r.date for r in self.dataset.sales)
        last = max(r.date for r in self.dataset.sales)
        return {
            "version": __version__,
            "summary": summary,
            "start": first.isoformat(),
            "end": last.isoformat(),
            "startup_seconds": self.startup_seconds,
            "models": [{"name": m, "label": LABELS[m]} for m in MODELS],
            "series": [
                {
                    "sku": sku,
                    "store": store,
                    "category": spec.category,
                    "intermittent": spec.intermittent,
                    "regime_change_day": spec.regime_change_day,
                    "mean_units": round(statistics.fmean(r.units for r in self.series[(sku, store)]), 1),
                    "best_model": best_by_series[(sku, store)][0],
                    "best_wape": round(best_by_series[(sku, store)][1], 4) if best_by_series[(sku, store)][1] is not None else None,
                }
                for (sku, store), spec in sorted(self.specs.items())
            ],
        }

    def series_view(self, sku: str, store: str, model: str, history: int = 168, horizon: int = 28) -> dict[str, Any]:
        _check_model(model)
        key = (sku, store)
        if key not in self.series:
            raise HTTPException(status_code=404, detail=f"no series {sku} at {store}")
        if (sku, store, model) not in self._views:
            self._views[(sku, store, model)] = self._series_view(key, model, history, horizon)
        return self._views[(sku, store, model)]

    def _series_view(self, key: tuple[str, str], model: str, history: int, horizon: int) -> dict[str, Any]:
        sku, store = key
        records = self.series[key]
        factory = MODELS[model]
        matrix = build_features(records) if factory().uses_features else None
        n = len(records)
        visible = records[-history:]
        offset = n - len(visible)

        # Backtest windows inside the visible range, plus wider ones for the interval.
        cutoffs = rolling_origins(n, BacktestConfig(horizon=14, step=14, max_origins=10))
        residuals: dict[int, list[float]] = defaultdict(list)
        windows = []
        for cutoff in cutoffs:
            predicted = forecast_at(records, factory, cutoff, 14, matrix)
            actual = [r.units for r in records[cutoff : cutoff + 14]]
            for step, (a, p) in enumerate(zip(actual, predicted, strict=False)):
                residuals[step].append(a - p)
            if cutoff >= offset:
                windows.append(
                    {
                        "cutoff": records[cutoff].date.isoformat(),
                        "start_index": cutoff - offset,
                        "predicted": [round(p, 2) for p in predicted],
                        "wape": _finite(score_forecast(actual, predicted, history=[r.units for r in records[:cutoff]]).wape),
                    }
                )
        forecast = forecast_future(records, factory, horizon, matrix)
        lower, upper = [], []
        for step, value in enumerate(forecast):
            # Empirical 80% interval from this model's own backtest errors at the same
            # horizon step. A biased model can have both quantiles on one side, so the
            # band is widened to always contain the point forecast it is drawn around.
            errors = residuals.get(min(step, 13), [0.0])
            lower.append(round(max(0.0, min(value, value + _quantile(errors, 0.1))), 2))
            upper.append(round(max(value, value + _quantile(errors, 0.9)), 2))
        result = self._result(model, key)
        spec = self.specs[key]
        units = [r.units for r in records]
        return {
            "sku": sku,
            "store": store,
            "model": model,
            "category": spec.category,
            "spec": {"intermittent": spec.intermittent, "price": round(spec.price, 2), "promo_probability": spec.promo_probability, "regime_change_day": spec.regime_change_day},
            "dates": [r.date.isoformat() for r in visible] + [(records[-1].date + timedelta(days=i + 1)).isoformat() for i in range(horizon)],
            "actual": [r.units for r in visible],
            "promotion": [r.on_promotion for r in visible],
            "holiday": [r.holiday for r in visible],
            "windows": windows,
            "forecast": [round(v, 2) for v in forecast],
            "lower": lower,
            "upper": upper,
            "metrics": result.to_dict(),
            "stats": {
                "mean_units": round(statistics.fmean(units), 1),
                "zero_share": round(sum(1 for u in units if u == 0) / n, 4),
                "promotion_days": sum(1 for r in records if r.on_promotion),
                "days": n,
            },
            "by_model": [
                {"model": m, "label": LABELS[m], **{k: v for k, v in self._result(m, key).to_dict().items() if k in ("mean_wape", "mean_mase", "wape_stdev")}}
                for m in MODELS
            ],
        }

    def leaderboard(self) -> dict[str, Any]:
        keys = sorted(self.series)
        return {
            "config": self.config.to_dict(),
            "comparisons": [{**c.to_dict(), "label": LABELS.get(c.model, c.model)} for c in self.comparisons],
            "matrix": {
                "series": [f"{sku} @ {store}" for sku, store in keys],
                "models": list(MODELS),
                "wape": [[self._result(m, key).mean_wape for m in MODELS] for key in keys],
            },
        }

    # -- recommendations ---------------------------------------------------------
    def customers(self, query: str = "", limit: int = 40) -> list[dict[str, Any]]:
        ranked = sorted(self.purchases.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
        rows = [{"customer": c, "purchases": sum(items.values()), "distinct": len(items)} for c, items in ranked if query.lower() in c.lower()]
        return rows[:limit]

    def recommend(self, customer: str, k: int = 8) -> dict[str, Any]:
        if customer not in self.purchases:
            raise HTTPException(status_code=404, detail=f"no purchase history for {customer!r}")
        history = self.purchases[customer]
        cf = self.cf.recommend(customer, k=k)
        popular = self.popularity.recommend(customer, k=k)
        return {
            "customer": customer,
            "history": [{"sku": sku, "quantity": q, "category": self.reco_categories.get(sku)} for sku, q in history.most_common()],
            "cf": [{**r.to_dict(), "category": self.reco_categories.get(r.sku)} for r in cf],
            "popular": [{**r.to_dict(), "category": self.reco_categories.get(r.sku)} for r in popular],
            "scores": self.reco_scores,
        }

    # -- drift and retraining ----------------------------------------------------
    def drift(self, sku: str, store: str, model_age_days: float = 12.0) -> dict[str, Any]:
        key = (sku, store)
        if key not in self.series:
            raise HTTPException(status_code=404, detail=f"no series {sku} at {store}")
        matrix = build_features(self.series[key])
        split = len(matrix.X) // 2
        origins = self._result("ridge", key).origins
        half = max(1, len(origins) // 2)
        baseline = statistics.fmean(o.score.wape for o in origins[:half]) if origins else 0.0
        current = statistics.fmean(o.score.wape for o in origins[half:]) if origins[half:] else baseline
        report = DriftReport(
            feature_results=check_feature_drift(
                columns_from_matrix(matrix.names, matrix.X[:split]),
                columns_from_matrix(matrix.names, matrix.X[split:]),
            ),
            performance=check_performance_drift(baseline, current),
        )
        decision = should_retrain(report, model_age_days=model_age_days, has_production_model=self.registry.production(REGISTRY_NAME) is not None)
        spec = self.specs[key]
        return {
            "sku": sku,
            "store": store,
            "reference_period": [matrix.dates[0].isoformat(), matrix.dates[split - 1].isoformat()],
            "current_period": [matrix.dates[split].isoformat(), matrix.dates[-1].isoformat()],
            "regime_change_day": spec.regime_change_day,
            "report": report.to_dict(),
            "baseline_wape": round(baseline, 4),
            "current_wape": round(current, 4),
            "decision": decision.to_dict(),
        }

    # -- tracking and registry ---------------------------------------------------
    def train(self, model: str) -> dict[str, Any]:
        _check_model(model)
        row = next(c for c in self.comparisons if c.model == model)
        self.tracker.start_run("forecasting", params={"model": model, **self.config.to_dict()})
        self.tracker.log_metrics(wape=row.wape or 0.0, mase=row.mase or 0.0)
        run = self.tracker.end_run()
        version = self.registry.register(REGISTRY_NAME, run.run_id if run else "unknown", "wape", row.wape or 1.0, params={"model": model})
        return version.to_dict()

    def promote(self, version: int) -> dict[str, Any]:
        promoted, reason = self.registry.promote(REGISTRY_NAME, version)
        return {"promoted": promoted, "reason": reason, "registry": self.registry_view()}

    def registry_view(self) -> dict[str, Any]:
        production = self.registry.production(REGISTRY_NAME)
        return {
            "name": REGISTRY_NAME,
            "production": production.to_dict() if production else None,
            "versions": [v.to_dict() for v in reversed(self.registry.versions(REGISTRY_NAME))],
            "runs": list(reversed(self.tracker.runs("forecasting"))),
            "labels": LABELS,
        }


def create_app(studio: Studio | None = None) -> FastAPI:
    state = studio or Studio(background=True)
    app = FastAPI(title="retailml studio", version=__version__, description="Demand forecasting and recommendations, measured.")
    app.state.studio = state

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok" if state.ready.is_set() else ("error" if state.error else "warming"),
            "version": __version__,
            "series": len(state.series),
            "progress": state.progress,
            "error": state.error,
            "startup_seconds": state.startup_seconds,
        }

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        state.require_ready()
        return state.overview()

    @app.get("/api/series")
    def series(sku: str, store: str, model: str = "ridge") -> dict[str, Any]:
        state.require_ready()
        return state.series_view(sku, store, model)

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        state.require_ready()
        return state.leaderboard()

    @app.get("/api/customers")
    def customers(q: str = "") -> list[dict[str, Any]]:
        state.require_ready()
        return state.customers(q)

    @app.get("/api/recommend")
    def recommend(customer: str, k: int = 8) -> dict[str, Any]:
        state.require_ready()
        return state.recommend(customer, max(1, min(k, 20)))

    @app.get("/api/drift")
    def drift(sku: str, store: str, model_age_days: float = 12.0) -> dict[str, Any]:
        state.require_ready()
        return state.drift(sku, store, model_age_days)

    @app.get("/api/registry")
    def registry() -> dict[str, Any]:
        state.require_ready()
        return state.registry_view()

    @app.post("/api/registry/train")
    def train(body: TrainRequest) -> dict[str, Any]:
        state.require_ready()
        return {"registered": state.train(body.model), "registry": state.registry_view()}

    @app.post("/api/registry/promote")
    def promote(body: PromoteRequest) -> dict[str, Any]:
        state.require_ready()
        return state.promote(body.version)

    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    return app


app = create_app()
