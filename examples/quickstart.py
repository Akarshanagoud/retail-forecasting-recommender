"""Runnable tour of the platform. No keys, no cluster, no downloads.

    python examples/quickstart.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retailml import (  # noqa: E402
    BacktestConfig,
    EnsembleForecaster,
    ExperimentTracker,
    HoltWinters,
    ItemItemRecommender,
    ModelRegistry,
    Naive,
    PopularityRecommender,
    RidgeRegression,
    SeasonalMovingAverage,
    SeasonalNaive,
    assert_no_leakage,
    build_features,
    check_feature_drift,
    check_performance_drift,
    evaluate_candidate,
    generate_dataset,
    should_retrain,
    temporal_split,
)
from retailml.backtest import backtest_dataset, compare_models, render_comparison  # noqa: E402
from retailml.drift import DriftReport, columns_from_matrix  # noqa: E402
from retailml.recommender import RecoComparison  # noqa: E402
from retailml.recommender import evaluate as evaluate_reco  # noqa: E402

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    rule("1. Synthetic data with the structure that makes retail hard")
    dataset = generate_dataset(n_skus=10, n_stores=2, days=730, n_customers=800)
    summary = dataset.summary()
    print(f"  {summary['rows']:,} rows across {summary['series']} series")
    print(f"  {summary['zero_days']:,} zero-demand days (intermittent SKUs)")
    print(f"  {summary['promotions']:,} promotion days with price elasticity")
    print(f"  {summary['interactions']:,} customer interactions")

    series = dataset.series(*dataset.series_keys()[0])
    by_weekday: dict[int, list[float]] = {}
    for record in series:
        by_weekday.setdefault(record.date.weekday(), []).append(record.units)
    print("\n  weekly seasonality of the first series:")
    for day in range(7):
        mean = sum(by_weekday[day]) / len(by_weekday[day])
        print(f"    {WEEKDAYS[day]}  {mean:6.1f}  {'#' * int(mean / 4)}")

    rule("2. Features that cannot leak, proved rather than promised")
    assert_no_leakage(series)
    matrix = build_features(series)
    print(f"  {len(matrix.X)} rows x {len(matrix.names)} features "
          f"({matrix.warmup} warmup rows dropped)")
    print("  perturbing a future target changed no earlier feature row")

    rule("3. Rolling-origin backtest against the baseline that matters")
    series_by_key = {key: dataset.series(*key) for key in dataset.series_keys()}
    results = backtest_dataset(
        series_by_key,
        {
            "naive": Naive,
            "seasonal_naive": SeasonalNaive,
            "seasonal_moving_average": SeasonalMovingAverage,
            "holt_winters": HoltWinters,
            "ridge": lambda: RidgeRegression(alpha=5.0),
            "ensemble": lambda: EnsembleForecaster(
                [SeasonalNaive(), RidgeRegression(alpha=5.0)]
            ),
        },
        BacktestConfig(horizon=14, step=28, max_origins=6),
    )
    comparisons = compare_models(results)
    print(render_comparison(comparisons))
    best = comparisons[0]
    baseline = next(c for c in comparisons if c.model == "seasonal_naive")
    print("\n  MASE below 1.0 means better than 'same day last week'.")
    print(f"  {best.model} improves WAPE by "
          f"{(baseline.wape - best.wape) / baseline.wape:.1%} over the baseline.")
    print("  STDEV is the spread across origins: a low mean with a high spread")
    print("  is a coin flip, not a deployable model.")

    rule("4. Recommender, scored against showing everyone the bestsellers")
    # A larger catalogue than the forecasting dataset, deliberately: with fewer
    # items than k, every recommender returns the whole catalogue and
    # precision, recall and coverage all saturate at 1.0 regardless of quality.
    reco_dataset = generate_dataset(
        n_skus=40, n_stores=2, days=400, n_customers=1200, seed=11
    )
    train, test = temporal_split(reco_dataset.interactions, 0.2)
    catalogue = {i.sku for i in reco_dataset.interactions}
    print(f"  catalogue of {len(catalogue)} items, k=10\n")
    comparison = RecoComparison()
    comparison.results["popularity"] = evaluate_reco(
        PopularityRecommender().fit(train), train, test, k=10, catalogue=catalogue
    )
    comparison.results["item-item CF"] = evaluate_reco(
        ItemItemRecommender().fit(train), train, test, k=10, catalogue=catalogue
    )
    print(comparison.render())

    model = ItemItemRecommender().fit(train)
    customer = sorted(model.user_items)[0]
    print(f"\n  explainable recommendations for {customer}:")
    for recommendation in model.recommend(customer, k=3):
        print(f"    {recommendation.sku:<10} because you bought "
              f"{', '.join(recommendation.because_of[:2])}")

    rule("5. Drift detection: what moved, and does it matter")
    split = len(matrix.X) // 2
    report = DriftReport(
        feature_results=check_feature_drift(
            columns_from_matrix(matrix.names, matrix.X[:split]),
            columns_from_matrix(matrix.names, matrix.X[split:]),
        ),
        performance=check_performance_drift(0.20, 0.31),
    )
    print(report.render())

    rule("6. Retraining requires evidence, and promotion requires improvement")
    trigger = should_retrain(report, model_age_days=12, has_production_model=True)
    print(f"  trigger : {[r.value for r in trigger.reasons] or 'none'}")

    marginal = evaluate_candidate(0.299, 0.300)
    material = evaluate_candidate(0.240, 0.300)
    print(f"  marginal candidate: promoted={marginal.promoted} - {marginal.reason}")
    print(f"  material candidate: promoted={material.promoted} - {material.reason}")

    rule("7. Every run is tracked, every promotion is gated")
    workspace = Path(tempfile.mkdtemp())
    tracker = ExperimentTracker(workspace / "runs.jsonl")
    registry = ModelRegistry(workspace / "registry.json")

    for row in comparisons[:3]:
        tracker.start_run("forecasting", params={"model": row.model})
        tracker.log_metrics(wape=row.wape or 0.0)
        run = tracker.end_run()
        registry.register(
            "demand_forecast", run.run_id, "wape", row.wape or 1.0,
            params={"model": row.model},
        )

    print("  leaderboard:")
    for entry in tracker.leaderboard("forecasting", "wape", limit=3):
        print(f"    {entry['params']['model']:<28} WAPE {entry['metrics']['wape']:.4f}")

    promoted, reason = registry.promote("demand_forecast", 1)
    print(f"\n  promote v1: {promoted} - {reason}")
    promoted, reason = registry.promote("demand_forecast", 2)
    print(f"  promote v2: {promoted} - {reason}")

    print("\nDone. Every number above was computed on this machine, in seconds.")


if __name__ == "__main__":
    main()
