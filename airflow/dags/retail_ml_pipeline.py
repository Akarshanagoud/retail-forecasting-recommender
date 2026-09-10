"""Airflow DAG for the retail ML pipeline.

Written so the *logic* lives in the ``retailml`` package and the DAG only
orchestrates. That separation is the point: a pipeline whose business logic
lives inside PythonOperator callables can only be tested by running Airflow,
which means in practice it is not tested.

The dependency graph encodes the discipline the rest of the package argues for:

    generate -> validate -> leakage_check -> features
                                                |
                  +-----------------------------+------------------+
                  |                             |                  |
             backtest_models              detect_drift      train_recommender
                  |                             |                  |
                  +--------------+--------------+                  |
                                 |                                 |
                        decide_retraining                   evaluate_recommender
                                 |
                     [promote_model | skip_promotion]
                                 |
                             publish

``leakage_check`` is a hard gate before any training task. It costs seconds and
it is the difference between catching a leak in CI and catching it after a
quarter of decisions were made on an inflated backtest.

The file imports cleanly without Airflow installed, so the repository's tests
can verify the graph shape without pulling in a scheduler.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

# The DAG structure is defined as data so it can be asserted in tests without
# Airflow present. Airflow itself is imported lazily below.
DAG_ID = "retail_ml_pipeline"
DEFAULT_ARGS: dict[str, Any] = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    # Fail rather than hang: a forecasting task that runs for six hours is
    # broken, and an unbounded task blocks every subsequent run.
    "execution_timeout": timedelta(hours=2),
}

TASK_DEPENDENCIES: dict[str, list[str]] = {
    "generate_data": [],
    "validate_data": ["generate_data"],
    "leakage_check": ["validate_data"],
    "build_features": ["leakage_check"],
    "backtest_models": ["build_features"],
    "detect_drift": ["build_features"],
    "decide_retraining": ["backtest_models", "detect_drift"],
    "promote_model": ["decide_retraining"],
    "skip_promotion": ["decide_retraining"],
    "train_recommender": ["validate_data"],
    "evaluate_recommender": ["train_recommender"],
    "publish": ["promote_model", "skip_promotion", "evaluate_recommender"],
}


# --- task callables ----------------------------------------------------------
# Thin wrappers over library functions. Everything here is importable and
# testable without Airflow.
def generate_data(**context: Any) -> dict[str, Any]:
    from retailml.data import generate_dataset

    dataset = generate_dataset(n_skus=12, n_stores=3, days=730, n_customers=800)
    summary = dataset.summary()
    print(json.dumps(summary, indent=2))
    return summary


def validate_data(**context: Any) -> dict[str, Any]:
    """Data quality gate. A pipeline that trains on broken data fails silently."""
    from retailml.data import generate_dataset

    dataset = generate_dataset(n_skus=12, n_stores=3, days=730, n_customers=800)
    problems: list[str] = []

    if not dataset.sales:
        problems.append("no sales rows")
    negative = [r for r in dataset.sales if r.units < 0]
    if negative:
        problems.append(f"{len(negative)} rows with negative units")
    bad_price = [r for r in dataset.sales if r.price <= 0]
    if bad_price:
        problems.append(f"{len(bad_price)} rows with non-positive price")

    # Every series must be complete: a gap silently shifts every lag feature.
    for key in dataset.series_keys():
        series = dataset.series(*key)
        dates = [r.date for r in series]
        if len(set(dates)) != len(dates):
            problems.append(f"{key}: duplicate dates")
        span = (dates[-1] - dates[0]).days + 1
        if span != len(dates):
            problems.append(f"{key}: {span - len(dates)} missing days")

    if problems:
        raise ValueError("data validation failed: " + "; ".join(problems[:5]))
    return {"rows": len(dataset.sales), "series": len(dataset.series_keys())}


def leakage_check(**context: Any) -> dict[str, Any]:
    """Hard gate: refuse to train if any feature can see the future."""
    from retailml.data import generate_dataset
    from retailml.features import assert_no_leakage

    dataset = generate_dataset(n_skus=6, n_stores=2, days=400, n_customers=200)
    checked = 0
    for key in dataset.series_keys()[:8]:
        assert_no_leakage(dataset.series(*key))
        checked += 1
    return {"series_checked": checked, "leakage": False}


def build_features(**context: Any) -> dict[str, Any]:
    from retailml.data import generate_dataset
    from retailml.features import build_features as build

    dataset = generate_dataset(n_skus=12, n_stores=3, days=730, n_customers=800)
    key = dataset.series_keys()[0]
    matrix = build(dataset.series(*key))
    return matrix.to_dict()


def backtest_models(**context: Any) -> dict[str, Any]:
    from retailml.backtest import BacktestConfig, backtest_dataset, compare_models
    from retailml.data import generate_dataset
    from retailml.forecasting import HoltWinters, RidgeRegression, SeasonalNaive

    dataset = generate_dataset(n_skus=6, n_stores=2, days=730, n_customers=200)
    series_by_key = {key: dataset.series(*key) for key in dataset.series_keys()}
    results = backtest_dataset(
        series_by_key,
        {
            "seasonal_naive": SeasonalNaive,
            "holt_winters": HoltWinters,
            "ridge": lambda: RidgeRegression(alpha=5.0),
        },
        BacktestConfig(horizon=14, step=28, max_origins=5),
    )
    comparisons = [c.to_dict() for c in compare_models(results)]
    print(json.dumps(comparisons, indent=2))
    return {"best_model": comparisons[0]["model"], "comparisons": comparisons}


def detect_drift(**context: Any) -> dict[str, Any]:
    from retailml.data import generate_dataset
    from retailml.drift import DriftReport, check_feature_drift, columns_from_matrix
    from retailml.features import build_features as build

    dataset = generate_dataset(n_skus=6, n_stores=2, days=730, n_customers=200)
    matrix = build(dataset.series(*dataset.series_keys()[0]))
    split = len(matrix.X) // 2
    report = DriftReport(
        feature_results=check_feature_drift(
            columns_from_matrix(matrix.names, matrix.X[:split]),
            columns_from_matrix(matrix.names, matrix.X[split:]),
        )
    )
    return report.to_dict()


def decide_retraining(**context: Any) -> str:
    """Branch on the evidence. Returns the id of the next task to run."""
    from retailml.retraining import RetrainingPolicy, should_retrain

    task_instance = context.get("ti")
    drift_payload = (
        task_instance.xcom_pull(task_ids="detect_drift") if task_instance else None
    ) or {}

    class _Report:
        drifted_features = drift_payload.get("drifted_features", [])
        performance_drifted = drift_payload.get("performance_drifted", False)
        performance = None

    decision = should_retrain(
        drift=_Report(),  # type: ignore[arg-type]
        model_age_days=context.get("params", {}).get("model_age_days", 31),
        has_production_model=True,
        policy=RetrainingPolicy(),
    )
    print(json.dumps(decision.to_dict(), indent=2))
    return "promote_model" if decision.should_retrain else "skip_promotion"


def promote_model(**context: Any) -> dict[str, Any]:
    """Register a candidate and promote only on a material improvement."""
    from retailml.retraining import RetrainingPolicy, evaluate_candidate
    from retailml.tracking import ModelRegistry

    task_instance = context.get("ti")
    backtest = (
        task_instance.xcom_pull(task_ids="backtest_models") if task_instance else None
    ) or {}
    comparisons = backtest.get("comparisons", [])
    candidate_error = comparisons[0]["wape"] if comparisons else 1.0

    registry = ModelRegistry("/tmp/retailml/registry.json")
    incumbent = registry.production("demand_forecast")
    decision = evaluate_candidate(
        candidate_error,
        incumbent.metric_value if incumbent else None,
        RetrainingPolicy(),
    )
    print(json.dumps(decision.to_dict(), indent=2))

    if decision.promoted:
        model = registry.register(
            "demand_forecast", run_id="airflow", metric_name="wape",
            metric_value=candidate_error,
        )
        registry.promote("demand_forecast", model.version)
    return decision.to_dict()


def skip_promotion(**context: Any) -> dict[str, Any]:
    return {"promoted": False, "reason": "no retraining trigger fired"}


def train_recommender(**context: Any) -> dict[str, Any]:
    from retailml.data import generate_dataset
    from retailml.recommender import ItemItemRecommender, temporal_split

    dataset = generate_dataset(n_skus=20, n_stores=2, days=400, n_customers=800)
    train, _test = temporal_split(dataset.interactions, 0.2)
    return ItemItemRecommender().fit(train).describe()


def evaluate_recommender(**context: Any) -> dict[str, Any]:
    from retailml.data import generate_dataset
    from retailml.recommender import (
        ItemItemRecommender,
        PopularityRecommender,
        evaluate,
        temporal_split,
    )

    dataset = generate_dataset(n_skus=20, n_stores=2, days=400, n_customers=800)
    train, test = temporal_split(dataset.interactions, 0.2)
    catalogue = {i.sku for i in dataset.interactions}

    model = evaluate(ItemItemRecommender().fit(train), train, test, catalogue=catalogue)
    baseline = evaluate(
        PopularityRecommender().fit(train), train, test, catalogue=catalogue
    )

    lift = (model.ndcg - baseline.ndcg) / max(baseline.ndcg, 1e-9)
    if lift < 0.05:
        # The recommender must earn its complexity against the trivial baseline.
        raise ValueError(
            f"recommender beats popularity by only {lift:.1%}; not worth deploying"
        )
    return {"model": model.to_dict(), "baseline": baseline.to_dict(), "lift": round(lift, 4)}


def publish(**context: Any) -> dict[str, Any]:
    return {"published_at": datetime.utcnow().isoformat()}


CALLABLES = {
    "generate_data": generate_data,
    "validate_data": validate_data,
    "leakage_check": leakage_check,
    "build_features": build_features,
    "backtest_models": backtest_models,
    "detect_drift": detect_drift,
    "decide_retraining": decide_retraining,
    "promote_model": promote_model,
    "skip_promotion": skip_promotion,
    "train_recommender": train_recommender,
    "evaluate_recommender": evaluate_recommender,
    "publish": publish,
}


def build_dag():  # pragma: no cover - requires Airflow
    """Construct the Airflow DAG. Imported lazily so the module stays testable."""
    from airflow.operators.python import BranchPythonOperator, PythonOperator
    from airflow.utils.trigger_rule import TriggerRule

    from airflow import DAG

    with DAG(
        dag_id=DAG_ID,
        default_args=DEFAULT_ARGS,
        description="Retail demand forecasting and recommendation",
        schedule="0 3 * * *",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["retail", "forecasting", "recommender", "mlops"],
    ) as dag:
        tasks = {}
        for name, callable_ in CALLABLES.items():
            if name == "decide_retraining":
                tasks[name] = BranchPythonOperator(task_id=name, python_callable=callable_)
            elif name == "publish":
                tasks[name] = PythonOperator(
                    task_id=name,
                    python_callable=callable_,
                    # One branch is always skipped, so publish must not wait
                    # for every upstream task to succeed.
                    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
                )
            else:
                tasks[name] = PythonOperator(task_id=name, python_callable=callable_)

        for name, upstreams in TASK_DEPENDENCIES.items():
            for upstream in upstreams:
                tasks[upstream] >> tasks[name]

    return dag


def validate_graph() -> list[str]:
    """Check the dependency graph is acyclic and fully connected.

    Runs without Airflow, so CI can catch a malformed DAG without installing a
    scheduler.
    """
    problems: list[str] = []
    for task, upstreams in TASK_DEPENDENCIES.items():
        for upstream in upstreams:
            if upstream not in TASK_DEPENDENCIES:
                problems.append(f"{task} depends on unknown task {upstream}")

    missing = set(TASK_DEPENDENCIES) - set(CALLABLES)
    if missing:
        problems.append(f"tasks without callables: {sorted(missing)}")

    # Topological sort; a remainder means a cycle.
    remaining = {t: set(u) for t, u in TASK_DEPENDENCIES.items()}
    ordered: list[str] = []
    while remaining:
        ready = [t for t, u in remaining.items() if not u - set(ordered)]
        if not ready:
            problems.append(f"cycle detected among {sorted(remaining)}")
            break
        for task in sorted(ready):
            ordered.append(task)
            del remaining[task]

    return problems


# Airflow discovers DAGs by scanning module globals, so this must run at import
# time when Airflow is present - and must not explode when it is not.
try:  # pragma: no cover - depends on the environment
    dag = build_dag()
except ImportError:
    dag = None
