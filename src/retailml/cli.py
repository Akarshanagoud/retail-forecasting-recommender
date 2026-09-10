"""Command line interface.

    retailml generate --days 730 --out data/
    retailml backtest --models seasonal_naive,holt_winters,ridge
    retailml recommend --customer CUST-0001
    retailml evaluate-reco
    retailml drift --reference-days 180
    retailml retrain
    retailml leakage-check

Exit codes: 0 fine, 1 a quality gate failed or drift was detected, 2 usage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backtest import BacktestConfig, backtest_dataset, compare_models, render_comparison
from .data import Dataset, generate_dataset
from .drift import DriftReport, check_feature_drift, check_performance_drift, columns_from_matrix
from .features import FeatureConfig, assert_no_leakage, build_features
from .forecasting import (
    Drift,
    EnsembleForecaster,
    HoltWinters,
    Naive,
    RidgeRegression,
    SeasonalMovingAverage,
    SeasonalNaive,
)
from .recommender import (
    ItemItemRecommender,
    PopularityRecommender,
    RecoComparison,
    RecommenderConfig,
    temporal_split,
)
from .recommender import evaluate as evaluate_reco
from .retraining import (
    RetrainingPolicy,
    evaluate_candidate,
    make_record,
    should_retrain,
)
from .tracking import ExperimentTracker, ModelRegistry, data_fingerprint

EXIT_OK, EXIT_GATE, EXIT_USAGE = 0, 1, 2

FACTORIES = {
    "naive": Naive,
    "seasonal_naive": SeasonalNaive,
    "seasonal_moving_average": SeasonalMovingAverage,
    "drift": Drift,
    "holt_winters": HoltWinters,
    "ridge": lambda: RidgeRegression(alpha=5.0),
    "ensemble": lambda: EnsembleForecaster(
        [SeasonalNaive(), RidgeRegression(alpha=5.0)]
    ),
}


def _dataset(args: argparse.Namespace) -> Dataset:
    return generate_dataset(
        n_skus=args.skus,
        n_stores=args.stores,
        days=args.days,
        n_customers=args.customers,
        seed=args.seed,
    )


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skus", type=int, default=12)
    parser.add_argument("--stores", type=int, default=3)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--customers", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true")


def cmd_generate(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    summary = dataset.summary()

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "sales.jsonl").write_text(
            "\n".join(json.dumps(r.to_dict()) for r in dataset.sales) + "\n",
            encoding="utf-8",
        )
        (out / "interactions.jsonl").write_text(
            "\n".join(json.dumps(i.to_dict()) for i in dataset.interactions) + "\n",
            encoding="utf-8",
        )
        (out / "specs.json").write_text(
            json.dumps([s.to_dict() for s in dataset.specs], indent=2), encoding="utf-8"
        )
        summary["written_to"] = str(out)

    print(json.dumps(summary, indent=2))
    return EXIT_OK


def cmd_leakage(args: argparse.Namespace) -> int:
    """Prove empirically that no feature depends on a future target."""
    dataset = _dataset(args)
    keys = dataset.series_keys()[: args.series]
    failures: list[str] = []

    for sku, store in keys:
        try:
            assert_no_leakage(dataset.series(sku, store))
        except AssertionError as exc:
            failures.append(f"{sku}/{store}: {exc}")

    if args.json:
        print(json.dumps({"series_checked": len(keys), "failures": failures}, indent=2))
    else:
        print(f"checked {len(keys)} series for target leakage")
        if failures:
            for failure in failures:
                print(f"  FAIL {failure}")
        else:
            print("  no feature row changed when a future target was perturbed")

    return EXIT_GATE if failures else EXIT_OK


def cmd_backtest(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in requested if m not in FACTORIES]
    if unknown:
        print(f"error: unknown model(s) {unknown}; choose from {sorted(FACTORIES)}",
              file=sys.stderr)
        return EXIT_USAGE

    series_by_key = {key: dataset.series(*key) for key in dataset.series_keys()}
    config = BacktestConfig(
        horizon=args.horizon, step=args.step, max_origins=args.origins,
        min_train_days=args.min_train_days,
    )

    tracker = ExperimentTracker(args.tracking_path)
    registry = ModelRegistry(args.registry_path)
    fingerprint = data_fingerprint(dataset.sales)

    results = backtest_dataset(
        series_by_key,
        {name: FACTORIES[name] for name in requested},
        config,
        FeatureConfig(),
        limit=args.series,
    )
    comparisons = compare_models(results)

    for comparison in comparisons:
        tracker.start_run(
            experiment=args.experiment,
            params={
                "model": comparison.model,
                "horizon": config.horizon,
                "origins": config.max_origins,
                "data_fingerprint": fingerprint,
            },
            tags={"stage": "backtest"},
        )
        tracker.log_metrics(
            wape=comparison.wape or 0.0,
            wape_stdev=comparison.wape_stdev or 0.0,
            mase=comparison.mase or 0.0,
            mean_fit_ms=comparison.mean_fit_ms,
        )
        run = tracker.end_run()
        if run is not None and comparison.wape is not None and args.register:
            registry.register(
                name=args.model_name, run_id=run.run_id,
                metric_name="wape", metric_value=comparison.wape,
                params={"model": comparison.model},
            )

    if args.json:
        print(json.dumps([c.to_dict() for c in comparisons], indent=2))
    else:
        print(render_comparison(comparisons))
        print(f"\n{len(series_by_key)} series, horizon {config.horizon}d, "
              f"{config.max_origins} rolling origins")
        best = comparisons[0]
        baseline = next(
            (c for c in comparisons if c.model == "seasonal_naive"), None
        )
        if baseline and baseline.wape and best.wape and best.model != "seasonal_naive":
            lift = (baseline.wape - best.wape) / baseline.wape
            print(f"best: {best.model} ({lift:.1%} better WAPE than seasonal naive)")

    if args.max_wape is not None:
        best_wape = comparisons[0].wape
        if best_wape is None or best_wape > args.max_wape:
            print(f"\nFAIL: best WAPE {best_wape} exceeds gate {args.max_wape}",
                  file=sys.stderr)
            return EXIT_GATE
    return EXIT_OK


def cmd_recommend(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    train, _test = temporal_split(dataset.interactions, args.test_fraction)
    model = ItemItemRecommender(
        RecommenderConfig(popularity_damping=args.damping)
    ).fit(train)

    customer = args.customer or sorted(model.user_items)[0]
    recommendations = model.recommend(customer, k=args.k)

    if args.json:
        print(json.dumps(
            {
                "customer": customer,
                "model": model.describe(),
                "recommendations": [r.to_dict() for r in recommendations],
            },
            indent=2,
        ))
        return EXIT_OK

    history = sorted(model.user_items.get(customer, set()))
    print(f"customer : {customer}")
    print(f"history  : {', '.join(history[:8]) or '(none - cold start)'}")
    print(f"\ntop {args.k} recommendations:")
    for rank, recommendation in enumerate(recommendations, start=1):
        because = ", ".join(recommendation.because_of[:3])
        print(f"  {rank:>2}. {recommendation.sku:<10} score {recommendation.score:>8.4f}"
              f"   because you bought: {because}")
    return EXIT_OK


def cmd_evaluate_reco(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    train, test = temporal_split(dataset.interactions, args.test_fraction)
    catalogue = {i.sku for i in dataset.interactions}

    comparison = RecoComparison()
    comparison.results["popularity"] = evaluate_reco(
        PopularityRecommender().fit(train), train, test, k=args.k, catalogue=catalogue
    )
    for damping in (0.0, 0.5, 1.0):
        model = ItemItemRecommender(
            RecommenderConfig(popularity_damping=damping)
        ).fit(train)
        comparison.results[f"item-item (damping {damping})"] = evaluate_reco(
            model, train, test, k=args.k, catalogue=catalogue
        )

    if args.json:
        print(json.dumps(comparison.to_dict(), indent=2))
    else:
        print(comparison.render())
        print(f"\n{len(train)} train / {len(test)} test interactions, "
              f"temporal split, catalogue of {len(catalogue)} items")

    best = max(comparison.results.items(), key=lambda kv: kv[1].ndcg)
    baseline = comparison.results["popularity"]
    if args.min_ndcg_lift is not None:
        lift = (best[1].ndcg - baseline.ndcg) / max(baseline.ndcg, 1e-9)
        if lift < args.min_ndcg_lift:
            print(f"\nFAIL: best model beats popularity by {lift:.1%}, "
                  f"gate is {args.min_ndcg_lift:.0%}", file=sys.stderr)
            return EXIT_GATE
    return EXIT_OK


def cmd_drift(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    key = dataset.series_keys()[args.series_index]
    series = dataset.series(*key)
    matrix = build_features(series)

    split = args.reference_days
    if split >= len(matrix.X) - 10:
        print("error: reference window leaves too little recent data", file=sys.stderr)
        return EXIT_USAGE

    reference = columns_from_matrix(matrix.names, matrix.X[:split])
    current = columns_from_matrix(matrix.names, matrix.X[split:])
    feature_results = check_feature_drift(reference, current)

    performance = None
    if args.baseline_error is not None and args.recent_error is not None:
        performance = check_performance_drift(args.baseline_error, args.recent_error)

    report = DriftReport(feature_results=feature_results, performance=performance)

    if args.json:
        print(json.dumps({"series": f"{key[0]}/{key[1]}", **report.to_dict()}, indent=2))
    else:
        print(f"series: {key[0]} / {key[1]}")
        print(report.render())

    return EXIT_GATE if report.verdict.value == "drifted" and args.fail_on_drift else EXIT_OK


def cmd_retrain(args: argparse.Namespace) -> int:
    """One pass of the drift-triggered retraining loop."""
    dataset = _dataset(args)
    registry = ModelRegistry(args.registry_path)
    tracker = ExperimentTracker(args.tracking_path)
    policy = RetrainingPolicy(min_improvement=args.min_improvement)

    key = dataset.series_keys()[args.series_index]
    series = dataset.series(*key)
    matrix = build_features(series)
    split = len(matrix.X) // 2
    reference = columns_from_matrix(matrix.names, matrix.X[:split])
    current = columns_from_matrix(matrix.names, matrix.X[split:])

    incumbent = registry.production(args.model_name)
    drift_report = DriftReport(
        feature_results=check_feature_drift(reference, current),
        performance=check_performance_drift(
            args.baseline_error, args.recent_error
        )
        if args.baseline_error is not None and args.recent_error is not None
        else None,
    )

    trigger = should_retrain(
        drift=drift_report,
        model_age_days=args.model_age_days,
        has_production_model=incumbent is not None,
        policy=policy,
    )

    if not trigger.should_retrain:
        record = make_record(trigger, model_name=args.model_name)
        print(json.dumps(record.to_dict(), indent=2) if args.json else record.render())
        return EXIT_OK

    # Train and backtest a candidate on the same rolling origins.
    series_by_key = {key: series}
    results = backtest_dataset(
        series_by_key,
        {"candidate": FACTORIES[args.candidate_model]},
        BacktestConfig(horizon=args.horizon, max_origins=args.origins),
        FeatureConfig(),
    )
    comparisons = compare_models(results)
    candidate_error = comparisons[0].wape if comparisons and comparisons[0].wape else 1.0

    tracker.start_run(
        experiment=args.experiment,
        params={"model": args.candidate_model, "trigger": [r.value for r in trigger.reasons]},
        tags={"stage": "retrain"},
    )
    tracker.log_metrics(wape=candidate_error)
    run = tracker.end_run()

    registered = registry.register(
        name=args.model_name,
        run_id=run.run_id if run else "unknown",
        metric_name="wape",
        metric_value=candidate_error,
        params={"model": args.candidate_model},
    )

    promotion = evaluate_candidate(
        candidate_error, incumbent.metric_value if incumbent else None, policy
    )
    if promotion.promoted:
        registry.promote(
            args.model_name, registered.version, min_improvement=policy.min_improvement
        )

    record = make_record(
        trigger, promotion, model_name=args.model_name, candidate_version=registered.version
    )
    print(json.dumps(record.to_dict(), indent=2) if args.json else record.render())
    return EXIT_OK


def cmd_registry(args: argparse.Namespace) -> int:
    registry = ModelRegistry(args.registry_path)
    print(json.dumps(registry.to_dict(), indent=2))
    return EXIT_OK


def cmd_runs(args: argparse.Namespace) -> int:
    tracker = ExperimentTracker(args.tracking_path)
    rows = tracker.leaderboard(args.experiment, args.metric, limit=args.limit)
    if args.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    if not rows:
        print("no runs recorded")
        return EXIT_OK
    print(f"{'RUN':<14}{'MODEL':<28}{args.metric.upper():>10}  STARTED")
    for row in rows:
        print(f"{row['run_id']:<14}{str(row['params'].get('model', '')):<28}"
              f"{row['metrics'][args.metric]:>10.4f}  {row['started_at'][:19]}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retailml",
        description="Retail demand forecasting and recommendation, end to end.",
    )
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")

    generate = sub.add_parser("generate", help="generate the synthetic dataset")
    _add_data_args(generate)
    generate.add_argument("--out", help="write JSONL files to this directory")
    generate.set_defaults(func=cmd_generate)

    leakage = sub.add_parser("leakage-check", help="prove features contain no leakage")
    _add_data_args(leakage)
    leakage.add_argument("--series", type=int, default=6)
    leakage.set_defaults(func=cmd_leakage)

    backtest = sub.add_parser("backtest", help="rolling-origin model comparison")
    _add_data_args(backtest)
    backtest.add_argument("--models", default="seasonal_naive,seasonal_moving_average,holt_winters,ridge,ensemble")
    backtest.add_argument("--horizon", type=int, default=14)
    backtest.add_argument("--step", type=int, default=28)
    backtest.add_argument("--origins", type=int, default=6)
    backtest.add_argument("--min-train-days", type=int, default=180)
    backtest.add_argument("--series", type=int, help="limit the number of series")
    backtest.add_argument("--experiment", default="forecasting")
    backtest.add_argument("--model-name", default="demand_forecast")
    backtest.add_argument("--tracking-path", default="mlruns/runs.jsonl")
    backtest.add_argument("--registry-path", default="mlruns/registry.json")
    backtest.add_argument("--register", action="store_true")
    backtest.add_argument("--max-wape", type=float, help="fail if the best WAPE exceeds this")
    backtest.set_defaults(func=cmd_backtest)

    recommend = sub.add_parser("recommend", help="recommendations for one customer")
    _add_data_args(recommend)
    recommend.add_argument("--customer")
    recommend.add_argument("-k", type=int, default=10)
    recommend.add_argument("--damping", type=float, default=0.5)
    recommend.add_argument("--test-fraction", type=float, default=0.2)
    recommend.set_defaults(func=cmd_recommend)

    evaluate = sub.add_parser("evaluate-reco", help="score recommenders against baselines")
    _add_data_args(evaluate)
    evaluate.add_argument("-k", type=int, default=10)
    evaluate.add_argument("--test-fraction", type=float, default=0.2)
    evaluate.add_argument("--min-ndcg-lift", type=float,
                          help="fail unless the best model beats popularity by this")
    evaluate.set_defaults(func=cmd_evaluate_reco)

    drift = sub.add_parser("drift", help="feature and performance drift")
    _add_data_args(drift)
    drift.add_argument("--series-index", type=int, default=0)
    drift.add_argument("--reference-days", type=int, default=300)
    drift.add_argument("--baseline-error", type=float)
    drift.add_argument("--recent-error", type=float)
    drift.add_argument("--fail-on-drift", action="store_true")
    drift.set_defaults(func=cmd_drift)

    retrain = sub.add_parser("retrain", help="one pass of the retraining loop")
    _add_data_args(retrain)
    retrain.add_argument("--series-index", type=int, default=0)
    retrain.add_argument("--candidate-model", default="ridge", choices=sorted(FACTORIES))
    retrain.add_argument("--model-name", default="demand_forecast")
    retrain.add_argument("--experiment", default="forecasting")
    retrain.add_argument("--horizon", type=int, default=14)
    retrain.add_argument("--origins", type=int, default=4)
    retrain.add_argument("--model-age-days", type=float)
    retrain.add_argument("--baseline-error", type=float)
    retrain.add_argument("--recent-error", type=float)
    retrain.add_argument("--min-improvement", type=float, default=0.02)
    retrain.add_argument("--tracking-path", default="mlruns/runs.jsonl")
    retrain.add_argument("--registry-path", default="mlruns/registry.json")
    retrain.set_defaults(func=cmd_retrain)

    registry = sub.add_parser("registry", help="show the model registry")
    registry.add_argument("--registry-path", default="mlruns/registry.json")
    registry.set_defaults(func=cmd_registry)

    runs = sub.add_parser("runs", help="experiment leaderboard")
    runs.add_argument("--experiment", default="forecasting")
    runs.add_argument("--metric", default="wape")
    runs.add_argument("--limit", type=int, default=10)
    runs.add_argument("--tracking-path", default="mlruns/runs.jsonl")
    runs.add_argument("--json", action="store_true")
    runs.set_defaults(func=cmd_runs)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(f"retailml {__version__}")
        return EXIT_OK

    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE

    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
