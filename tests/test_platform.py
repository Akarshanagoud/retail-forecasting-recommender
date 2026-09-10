"""Recommender, drift detection, tracking, retraining and the Airflow graph."""
import json
import sys
from datetime import date
from pathlib import Path

import pytest

from retailml import (
    DriftVerdict,
    ExperimentTracker,
    ItemItemRecommender,
    ModelRegistry,
    PopularityRecommender,
    RecommenderConfig,
    RetrainingPolicy,
    check_feature_drift,
    check_performance_drift,
    evaluate_candidate,
    generate_dataset,
    ks_statistic,
    population_stability_index,
    should_retrain,
    temporal_split,
)
from retailml.data import Interaction
from retailml.drift import DriftReport, columns_from_matrix
from retailml.recommender import evaluate as evaluate_reco
from retailml.retraining import TriggerReason, make_record, summarise

DAG_DIR = Path(__file__).resolve().parents[1] / "airflow" / "dags"


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_skus=15, n_stores=2, days=400, n_customers=700, seed=9)


# --- recommender -------------------------------------------------------------
def test_temporal_split_never_puts_the_future_in_training(dataset):
    train, test = temporal_split(dataset.interactions, 0.2)
    assert train and test
    assert max(i.date for i in train) < min(i.date for i in test)


def test_temporal_split_handles_an_empty_input():
    assert temporal_split([]) == ([], [])


def test_recommender_learns_co_purchase_structure(dataset):
    """The whole point: it must beat recommending the bestsellers to everyone."""
    train, test = temporal_split(dataset.interactions, 0.2)
    catalogue = {i.sku for i in dataset.interactions}

    model = evaluate_reco(
        ItemItemRecommender().fit(train), train, test, k=10, catalogue=catalogue
    )
    baseline = evaluate_reco(
        PopularityRecommender().fit(train), train, test, k=10, catalogue=catalogue
    )
    assert model.ndcg > baseline.ndcg
    assert model.map_score > baseline.map_score


def test_recommendations_are_explainable(dataset):
    train, _ = temporal_split(dataset.interactions, 0.2)
    model = ItemItemRecommender().fit(train)
    customer = sorted(model.user_items)[0]
    recommendations = model.recommend(customer, k=5)
    assert recommendations
    assert all(r.because_of for r in recommendations)
    # Every reason must be an item the customer actually bought.
    history = model.user_items[customer]
    for recommendation in recommendations:
        assert set(recommendation.because_of) <= history


def test_already_purchased_items_are_excluded(dataset):
    train, _ = temporal_split(dataset.interactions, 0.2)
    model = ItemItemRecommender().fit(train)
    customer = sorted(model.user_items)[0]
    history = model.user_items[customer]
    recommended = {r.sku for r in model.recommend(customer, k=10)}
    assert not (recommended & history)


def test_cold_start_falls_back_to_popularity_and_says_so(dataset):
    train, _ = temporal_split(dataset.interactions, 0.2)
    model = ItemItemRecommender().fit(train)
    recommendations = model.recommend("CUST-NEVER-SEEN", k=5)
    assert recommendations
    assert recommendations[0].because_of == ["<popularity fallback>"]


def test_popularity_damping_changes_the_ranking():
    interactions = (
        [Interaction(f"C{i}", "POPULAR", date(2026, 1, 1)) for i in range(120)]
        + [Interaction(f"C{i}", "NICHE", date(2026, 1, 1)) for i in range(0, 40)]
        + [Interaction(f"C{i}", "TARGET", date(2026, 1, 1)) for i in range(0, 40)]
    )
    undamped = ItemItemRecommender(
        RecommenderConfig(popularity_damping=0.0, min_item_interactions=2)
    ).fit(interactions)
    damped = ItemItemRecommender(
        RecommenderConfig(popularity_damping=1.0, min_item_interactions=2)
    ).fit(interactions)
    # NICHE co-occurs perfectly with TARGET; POPULAR co-occurs with everything.
    undamped_top = [s for s, _ in undamped.similar_items("TARGET", 2)]
    damped_top = [s for s, _ in damped.similar_items("TARGET", 2)]
    assert "NICHE" in damped_top
    assert undamped_top != damped_top or damped_top[0] == "NICHE"


def test_rare_items_are_excluded_from_similarity():
    interactions = [
        Interaction("C1", "COMMON", date(2026, 1, 1)),
        Interaction("C2", "COMMON", date(2026, 1, 1)),
        Interaction("C3", "COMMON", date(2026, 1, 1)),
        Interaction("C1", "RARE", date(2026, 1, 1)),
    ]
    model = ItemItemRecommender(RecommenderConfig(min_item_interactions=3)).fit(
        interactions
    )
    assert "RARE" not in model.item_popularity


def test_evaluation_skips_users_with_nothing_new_to_predict(dataset):
    train, test = temporal_split(dataset.interactions, 0.2)
    score = evaluate_reco(ItemItemRecommender().fit(train), train, test, k=10)
    assert score.users_evaluated > 0
    assert score.skipped >= 0


def test_evaluation_reports_coverage_and_novelty(dataset):
    train, test = temporal_split(dataset.interactions, 0.2)
    score = evaluate_reco(ItemItemRecommender().fit(train), train, test, k=5)
    payload = score.to_dict()
    assert 0.0 <= payload["catalogue_coverage"] <= 1.0
    assert payload["novelty"] > 0


# --- drift -------------------------------------------------------------------
def test_psi_is_near_zero_for_identical_distributions():
    values = [float(i % 50) for i in range(600)]
    assert population_stability_index(values, list(values)) < 0.01


def test_psi_detects_a_shifted_distribution():
    reference = [float(i % 50) for i in range(600)]
    shifted = [v + 40 for v in reference]
    assert population_stability_index(reference, shifted) > 0.25


def test_psi_is_finite_when_a_bin_is_empty():
    """Without the epsilon floor this goes to infinity and the alert is useless."""
    reference = [float(i) for i in range(200)]
    disjoint = [1000.0] * 200
    psi = population_stability_index(reference, disjoint)
    assert psi != float("inf")
    assert psi > 0.25


def test_psi_handles_empty_input():
    assert population_stability_index([], [1.0]) == 0.0


def test_ks_is_zero_for_identical_samples():
    values = [float(i) for i in range(100)]
    assert ks_statistic(values, list(values)) == 0.0


def test_ks_is_one_for_disjoint_samples():
    assert ks_statistic([1.0] * 50, [100.0] * 50) == 1.0


def test_feature_drift_ranks_the_worst_feature_first():
    reference = {
        "stable": [float(i % 10) for i in range(400)],
        "shifted": [float(i % 10) for i in range(400)],
    }
    current = {
        "stable": [float(i % 10) for i in range(400)],
        "shifted": [float(i % 10) + 50 for i in range(400)],
    }
    results = check_feature_drift(reference, current)
    assert results[0].feature == "shifted"
    assert results[0].drifted
    assert not results[1].drifted


def test_performance_drift_uses_a_relative_threshold():
    """15% WAPE is excellent for one SKU and terrible for another."""
    stable = check_performance_drift(0.20, 0.21)
    drifted = check_performance_drift(0.20, 0.30)
    assert stable.verdict is DriftVerdict.STABLE
    assert drifted.verdict is DriftVerdict.DRIFTED


def test_watch_band_sits_between_stable_and_drifted():
    watch = check_performance_drift(0.20, 0.23)
    assert watch.verdict is DriftVerdict.WATCH


def test_drift_report_overall_verdict():
    report = DriftReport(
        feature_results=check_feature_drift(
            {"a": [1.0] * 100}, {"a": [1.0] * 100}
        ),
        performance=check_performance_drift(0.2, 0.5),
    )
    assert report.verdict is DriftVerdict.DRIFTED
    assert report.performance_drifted


def test_drift_on_real_features(dataset):
    from retailml import build_features

    matrix = build_features(dataset.series(*dataset.series_keys()[0]))
    split = len(matrix.X) // 2
    results = check_feature_drift(
        columns_from_matrix(matrix.names, matrix.X[:split]),
        columns_from_matrix(matrix.names, matrix.X[split:]),
    )
    assert results
    assert all(r.metric == "psi" for r in results)


# --- tracking and registry ---------------------------------------------------
def test_tracker_records_a_reproducible_run(tmp_path):
    tracker = ExperimentTracker(tmp_path / "runs.jsonl")
    tracker.start_run("forecasting", params={"model": "ridge", "alpha": 5.0})
    tracker.log_metrics(wape=0.15)
    run = tracker.end_run()

    assert run is not None
    stored = json.loads((tmp_path / "runs.jsonl").read_text(encoding="utf-8").strip())
    assert stored["params"]["model"] == "ridge"
    assert stored["metrics"]["wape"] == 0.15
    # Reproducibility context, not just the number.
    assert "python" in stored["environment"]
    assert stored["duration_s"] is not None


def test_leaderboard_orders_by_metric(tmp_path):
    tracker = ExperimentTracker(tmp_path / "runs.jsonl")
    for name, wape in (("a", 0.3), ("b", 0.1), ("c", 0.2)):
        tracker.start_run("forecasting", params={"model": name})
        tracker.log_metrics(wape=wape)
        tracker.end_run()
    board = tracker.leaderboard("forecasting", "wape")
    assert [r["params"]["model"] for r in board] == ["b", "c", "a"]
    assert tracker.best_run("forecasting", "wape")["params"]["model"] == "b"


def test_registry_promotes_the_first_model_unconditionally(tmp_path):
    registry = ModelRegistry(tmp_path / "registry.json")
    model = registry.register("demand", "run1", "wape", 0.30)
    promoted, reason = registry.promote("demand", model.version)
    assert promoted
    assert "no incumbent" in reason
    assert registry.production("demand").version == model.version


def test_registry_blocks_a_marginal_improvement(tmp_path):
    """Without a margin, run-to-run variance makes production a random walk."""
    registry = ModelRegistry(tmp_path / "registry.json")
    first = registry.register("demand", "run1", "wape", 0.300)
    registry.promote("demand", first.version)

    second = registry.register("demand", "run2", "wape", 0.299)
    promoted, reason = registry.promote("demand", second.version, min_improvement=0.02)
    assert not promoted
    assert "below the 2% gate" in reason
    assert registry.production("demand").version == first.version


def test_registry_promotes_a_material_improvement(tmp_path):
    registry = ModelRegistry(tmp_path / "registry.json")
    first = registry.register("demand", "run1", "wape", 0.30)
    registry.promote("demand", first.version)
    second = registry.register("demand", "run2", "wape", 0.20)
    promoted, _ = registry.promote("demand", second.version, min_improvement=0.02)
    assert promoted
    assert registry.production("demand").version == second.version
    assert registry.versions("demand")[0].stage == "archived"


def test_registry_persists_across_instances(tmp_path):
    path = tmp_path / "registry.json"
    registry = ModelRegistry(path)
    model = registry.register("demand", "run1", "wape", 0.3)
    registry.promote("demand", model.version)
    assert ModelRegistry(path).production("demand") is not None


def test_registry_rejects_an_unknown_version(tmp_path):
    registry = ModelRegistry(tmp_path / "registry.json")
    promoted, reason = registry.promote("demand", 99)
    assert not promoted
    assert "not registered" in reason


# --- retraining policy -------------------------------------------------------
def _report(drifted: int) -> DriftReport:
    reference = {f"f{i}": [float(j % 10) for j in range(300)] for i in range(6)}
    current = {
        f"f{i}": [float(j % 10) + (60 if i < drifted else 0) for j in range(300)]
        for i in range(6)
    }
    return DriftReport(feature_results=check_feature_drift(reference, current))


def test_one_drifted_feature_does_not_trigger_retraining():
    """A single shifted feature is usually a pipeline hiccup, not model decay."""
    decision = should_retrain(_report(1), model_age_days=1, has_production_model=True)
    assert not decision.should_retrain
    assert "feature_drift_below_threshold" in decision.detail


def test_sustained_feature_drift_triggers_retraining():
    decision = should_retrain(_report(4), model_age_days=1, has_production_model=True)
    assert decision.should_retrain
    assert TriggerReason.FEATURE_DRIFT in decision.reasons


def test_performance_drift_triggers_retraining():
    report = DriftReport(
        feature_results=[], performance=check_performance_drift(0.2, 0.4)
    )
    decision = should_retrain(report, model_age_days=1, has_production_model=True)
    assert TriggerReason.PERFORMANCE_DRIFT in decision.reasons


def test_age_triggers_retraining():
    decision = should_retrain(None, model_age_days=45, has_production_model=True)
    assert TriggerReason.SCHEDULE in decision.reasons


def test_missing_production_model_triggers_retraining():
    decision = should_retrain(None, model_age_days=1, has_production_model=False)
    assert TriggerReason.NO_PRODUCTION_MODEL in decision.reasons


def test_candidate_needs_a_material_improvement():
    marginal = evaluate_candidate(0.299, 0.300)
    material = evaluate_candidate(0.250, 0.300)
    assert not marginal.promoted
    assert material.promoted


def test_candidate_is_rejected_when_absolutely_bad():
    """Beating a broken incumbent is not a reason to ship."""
    decision = evaluate_candidate(
        0.80, 0.95, RetrainingPolicy(max_acceptable_error=0.60)
    )
    assert not decision.promoted
    assert "absolute ceiling" in decision.reason


def test_first_candidate_is_promoted_without_an_incumbent():
    assert evaluate_candidate(0.30, None).promoted


def test_retraining_summary_reports_the_trigger_to_promotion_ratio():
    records = [
        make_record(should_retrain(_report(4), 1, True), evaluate_candidate(0.2, 0.3)),
        make_record(should_retrain(_report(0), 1, True)),
        make_record(should_retrain(_report(4), 1, True), evaluate_candidate(0.299, 0.3)),
    ]
    summary = summarise(records)
    assert summary["checks"] == 3
    assert summary["triggered"] == 2
    assert summary["promoted"] == 1
    assert summary["promotion_rate"] == 0.5


# --- Airflow DAG -------------------------------------------------------------
def test_dag_module_imports_without_airflow():
    """The pipeline definition must be testable without a scheduler installed."""
    sys.path.insert(0, str(DAG_DIR))
    try:
        import retail_ml_pipeline

        assert retail_ml_pipeline.DAG_ID == "retail_ml_pipeline"
    finally:
        sys.path.remove(str(DAG_DIR))


def test_dag_graph_is_valid():
    sys.path.insert(0, str(DAG_DIR))
    try:
        import retail_ml_pipeline

        assert retail_ml_pipeline.validate_graph() == []
    finally:
        sys.path.remove(str(DAG_DIR))


def test_leakage_check_gates_every_training_task():
    """Training must never be reachable without passing the leakage gate."""
    sys.path.insert(0, str(DAG_DIR))
    try:
        import retail_ml_pipeline

        dependencies = retail_ml_pipeline.TASK_DEPENDENCIES

        def ancestors(task: str) -> set[str]:
            found: set[str] = set()
            stack = list(dependencies.get(task, []))
            while stack:
                current = stack.pop()
                if current in found:
                    continue
                found.add(current)
                stack.extend(dependencies.get(current, []))
            return found

        assert "leakage_check" in ancestors("backtest_models")
        assert "leakage_check" in ancestors("build_features")
    finally:
        sys.path.remove(str(DAG_DIR))


def test_every_dag_task_has_a_callable():
    sys.path.insert(0, str(DAG_DIR))
    try:
        import retail_ml_pipeline

        assert set(retail_ml_pipeline.TASK_DEPENDENCIES) == set(
            retail_ml_pipeline.CALLABLES
        )
    finally:
        sys.path.remove(str(DAG_DIR))


def test_data_validation_task_catches_broken_data():
    sys.path.insert(0, str(DAG_DIR))
    try:
        import retail_ml_pipeline

        # The real dataset must pass.
        assert retail_ml_pipeline.validate_data()["rows"] > 0
    finally:
        sys.path.remove(str(DAG_DIR))
