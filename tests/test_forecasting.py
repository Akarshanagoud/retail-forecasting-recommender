"""Data generation, metrics, models, features and backtesting."""
import math

import pytest

from retailml import (
    BacktestConfig,
    Drift,
    EnsembleForecaster,
    FeatureConfig,
    HoltWinters,
    Naive,
    RidgeRegression,
    SeasonalMovingAverage,
    SeasonalNaive,
    assert_no_leakage,
    backtest_series,
    build_features,
    build_model,
    generate_dataset,
    score_forecast,
)
from retailml.backtest import backtest_dataset, compare_models, rolling_origins
from retailml.metrics import (
    aggregate_scores,
    bias,
    coverage,
    mae,
    mape,
    mase,
    pinball_loss,
    rmse,
    smape,
    wape,
)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_skus=4, n_stores=2, days=560, n_customers=200, seed=3)


# --- data generation ---------------------------------------------------------
def test_generation_is_deterministic():
    a = generate_dataset(n_skus=2, n_stores=1, days=120, n_customers=40, seed=5)
    b = generate_dataset(n_skus=2, n_stores=1, days=120, n_customers=40, seed=5)
    assert [r.units for r in a.sales] == [r.units for r in b.sales]


def test_different_seeds_give_different_data():
    a = generate_dataset(n_skus=2, n_stores=1, days=120, n_customers=40, seed=1)
    b = generate_dataset(n_skus=2, n_stores=1, days=120, n_customers=40, seed=2)
    assert [r.units for r in a.sales] != [r.units for r in b.sales]


def test_series_are_complete_and_ordered(dataset):
    for key in dataset.series_keys():
        series = dataset.series(*key)
        dates = [r.date for r in series]
        assert dates == sorted(dates)
        span = (dates[-1] - dates[0]).days + 1
        assert span == len(dates), "a gap would silently shift every lag feature"


def test_demand_is_never_negative(dataset):
    assert all(r.units >= 0 for r in dataset.sales)


def test_weekly_seasonality_is_present(dataset):
    """Saturday should out-sell Tuesday. Without this the data teaches nothing."""
    series = dataset.series(*dataset.series_keys()[0])
    by_weekday: dict[int, list[float]] = {}
    for record in series:
        by_weekday.setdefault(record.date.weekday(), []).append(record.units)
    means = {d: sum(v) / len(v) for d, v in by_weekday.items()}
    assert max(means.values()) > 1.2 * min(means.values())


def test_promotions_move_demand(dataset):
    """Price elasticity must be visible, or feature models cannot beat baselines."""
    series = dataset.series(*dataset.series_keys()[0])
    promo = [r.units for r in series if r.on_promotion]
    normal = [r.units for r in series if not r.on_promotion]
    assert promo and normal
    assert sum(promo) / len(promo) > sum(normal) / len(normal)


def test_intermittent_series_have_zero_days(dataset):
    assert any(r.units == 0 for r in dataset.sales)


def test_interactions_have_category_structure(dataset):
    """Co-purchase structure is what a collaborative filter learns."""
    assert dataset.interactions
    per_customer: dict[str, set[str]] = {}
    for interaction in dataset.interactions:
        per_customer.setdefault(interaction.customer, set()).add(interaction.sku)
    assert max(len(v) for v in per_customer.values()) > 1


# --- metrics -----------------------------------------------------------------
def test_perfect_forecast_scores_zero():
    actual = [10.0, 12.0, 8.0]
    score = score_forecast(actual, list(actual))
    assert score.mae == 0.0
    assert score.rmse == 0.0
    assert score.wape == 0.0


def test_rmse_penalises_large_errors_more_than_mae():
    actual = [10.0, 10.0, 10.0, 10.0]
    spread = [11.0, 9.0, 11.0, 9.0]
    concentrated = [10.0, 10.0, 10.0, 14.0]
    assert mae(actual, spread) == mae(actual, concentrated)
    assert rmse(actual, concentrated) > rmse(actual, spread)


def test_bias_detects_a_systematically_high_forecast():
    assert bias([10.0, 10.0], [12.0, 12.0]) == 2.0
    assert bias([10.0, 10.0], [8.0, 12.0]) == 0.0


def test_mape_is_none_when_every_actual_is_zero():
    assert mape([0.0, 0.0], [1.0, 2.0]) is None


def test_mape_skips_zero_actuals_rather_than_dividing_by_them():
    # This is exactly why WAPE is the headline metric instead.
    assert mape([0.0, 10.0], [5.0, 11.0]) == pytest.approx(0.1)


def test_wape_is_defined_with_zero_actuals():
    assert wape([0.0, 10.0], [1.0, 11.0]) == pytest.approx(0.2)


def test_wape_weights_by_volume():
    """A 5-unit miss on a 500-unit SKU is not the same as on a 5-unit SKU."""
    big = wape([500.0], [505.0])
    small = wape([5.0], [10.0])
    assert small > big


def test_smape_is_bounded():
    assert 0 <= smape([10.0, 0.0], [0.0, 10.0]) <= 2


def test_mase_below_one_means_better_than_seasonal_naive():
    # Deliberately not perfectly periodic: an exactly repeating history has a
    # seasonal-difference scale of zero, which makes MASE undefined.
    history = [
        10.0 + (i % 7) + (i * 0.3 % 4) for i in range(42)
    ]
    good = mase([10.0], [10.0], history)
    bad = mase([10.0], [40.0], history)
    assert good is not None and good < 1.0
    assert bad is not None and bad > 1.0


def test_mase_is_none_for_a_flat_history():
    assert mase([1.0], [1.0], [5.0] * 30) is None


def test_pinball_penalises_under_forecast_at_high_quantiles():
    under = pinball_loss([100.0], [90.0], quantile=0.9)
    over = pinball_loss([100.0], [110.0], quantile=0.9)
    assert under > over


def test_pinball_rejects_an_invalid_quantile():
    with pytest.raises(ValueError, match="between 0 and 1"):
        pinball_loss([1.0], [1.0], quantile=1.5)


def test_coverage_counts_actuals_inside_the_interval():
    assert coverage([5.0, 15.0], [0.0, 0.0], [10.0, 10.0]) == 0.5


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="length mismatch"):
        mae([1.0, 2.0], [1.0])


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError, match="empty"):
        mae([], [])


def test_aggregate_recomputes_wape_from_totals():
    """An average of ratios is not the ratio of totals."""
    high_volume = score_forecast([1000.0], [900.0])   # wape 0.10
    low_volume = score_forecast([1.0], [2.0])         # wape 1.00
    aggregated = aggregate_scores([high_volume, low_volume])
    naive_mean = (high_volume.wape + low_volume.wape) / 2
    assert aggregated["wape"] == pytest.approx(101 / 1001, abs=1e-4)
    assert aggregated["wape"] < naive_mean


# --- models ------------------------------------------------------------------
def test_naive_repeats_the_last_value():
    assert Naive().fit([1.0, 2.0, 5.0]).predict(3) == [5.0, 5.0, 5.0]


def test_seasonal_naive_repeats_the_last_season():
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert SeasonalNaive(7).fit(y).predict(9) == [1, 2, 3, 4, 5, 6, 7, 1, 2]


def test_seasonal_naive_falls_back_to_the_mean_on_short_history():
    predictions = SeasonalNaive(7).fit([2.0, 4.0]).predict(3)
    assert all(p == pytest.approx(3.0) for p in predictions)


def test_seasonal_moving_average_smooths_across_seasons():
    y = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 4
    prediction = SeasonalMovingAverage(7, seasons=4).fit(y).predict(1)[0]
    assert prediction == pytest.approx(10.0)


def test_drift_extrapolates_the_slope():
    assert Drift().fit([10.0, 11.0, 12.0]).predict(2) == [13.0, 14.0]


def test_no_model_predicts_negative_demand():
    """A negative forecast propagates into a nonsensical replenishment order."""
    y = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.0]
    for model in (Naive(), SeasonalNaive(7), Drift(), SeasonalMovingAverage(7)):
        assert all(p >= 0 for p in model.fit(y).predict(20))


def test_holt_winters_captures_multiplicative_seasonality():
    profile = [0.8, 0.9, 1.0, 1.0, 1.2, 1.4, 0.7]
    y = [100 * profile[i % 7] * (1 + 0.001 * i) for i in range(140)]
    predictions = HoltWinters(7).fit(y).predict(7)
    # Saturday (index 5) must remain the peak.
    assert predictions.index(max(predictions)) == 5


def test_holt_winters_damps_a_long_horizon():
    """An undamped trend extrapolated far out produces absurd forecasts."""
    y = [100 + 2 * i for i in range(120)]
    damped = HoltWinters(7, damped=0.9).fit(y).predict(120)
    undamped = HoltWinters(7, damped=1.0).fit(y).predict(120)
    assert damped[-1] < undamped[-1]


def test_holt_winters_degrades_gracefully_on_short_history():
    predictions = HoltWinters(7).fit([5.0] * 8).predict(5)
    assert len(predictions) == 5
    assert all(p >= 0 for p in predictions)


def test_holt_winters_rejects_invalid_smoothing():
    with pytest.raises(ValueError, match="between 0 and 1"):
        HoltWinters(alpha=1.5)


def test_ridge_recovers_a_linear_relationship():
    X = [[float(i), float(i * 2)] for i in range(60)]
    y = [3.0 * row[0] + 10.0 for row in X]
    model = RidgeRegression(alpha=0.01).fit(y, X)
    predictions = model.predict(3, [[60.0, 120.0], [61.0, 122.0], [62.0, 124.0]])
    assert predictions[0] == pytest.approx(190.0, rel=0.05)


def test_ridge_shrinks_coefficients_as_alpha_grows():
    X = [[float(i), float(i) + 0.01] for i in range(50)]
    y = [2.0 * i for i in range(50)]
    weak = RidgeRegression(alpha=0.01).fit(y, X)
    strong = RidgeRegression(alpha=1000.0).fit(y, X)
    assert sum(abs(c) for c in strong.coefficients) < sum(abs(c) for c in weak.coefficients)


def test_ridge_requires_features():
    with pytest.raises(ValueError, match="requires a feature matrix"):
        RidgeRegression().fit([1.0, 2.0])


def test_ridge_rejects_a_negative_alpha():
    with pytest.raises(ValueError, match="non-negative"):
        RidgeRegression(alpha=-1.0)


def test_ensemble_averages_its_members():
    y = [10.0] * 30
    ensemble = EnsembleForecaster([Naive(), Naive()])
    assert ensemble.fit(y).predict(2) == [10.0, 10.0]


def test_ensemble_rejects_an_empty_model_list():
    with pytest.raises(ValueError, match="at least one model"):
        EnsembleForecaster([])


def test_build_model_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("prophet")


# --- features ----------------------------------------------------------------
def test_features_contain_no_leakage(dataset):
    for key in dataset.series_keys()[:4]:
        assert_no_leakage(dataset.series(*key))


def test_lag_of_zero_is_rejected():
    with pytest.raises(ValueError, match="leakage"):
        FeatureConfig(lags=(0, 1))


def test_rolling_mean_excludes_the_current_target(dataset):
    series = dataset.series(*dataset.series_keys()[0])
    config = FeatureConfig(lags=(1,), rolling_windows=(3,), rolling_std_windows=())
    matrix = build_features(series, config)
    index = matrix.names.index("rolling_mean_3")
    targets = [r.units for r in sorted(series, key=lambda r: r.date)]
    warmup = matrix.warmup
    expected = sum(targets[warmup - 3 : warmup]) / 3
    assert matrix.X[0][index] == pytest.approx(expected)


def test_feature_matrix_rows_align_with_targets(dataset):
    series = dataset.series(*dataset.series_keys()[0])
    matrix = build_features(series)
    assert len(matrix.X) == len(matrix.y) == len(matrix.dates)
    targets = [r.units for r in sorted(series, key=lambda r: r.date)]
    assert matrix.y == targets[matrix.warmup :]


def test_split_is_chronological(dataset):
    matrix = build_features(dataset.series(*dataset.series_keys()[0]))
    left, right = matrix.split_at(100)
    assert len(left) == 100
    assert max(left.dates) < min(right.dates)


def test_calendar_features_are_known_in_advance(dataset):
    matrix = build_features(dataset.series(*dataset.series_keys()[0]))
    assert "dow_0" in matrix.names
    assert "annual_sin_1" in matrix.names
    assert "is_holiday" in matrix.names


def test_price_and_promotion_features_present(dataset):
    matrix = build_features(dataset.series(*dataset.series_keys()[0]))
    for name in ("price", "discount_depth", "on_promotion", "promo_last_week"):
        assert name in matrix.names


def test_leakage_check_catches_a_deliberately_leaky_feature(dataset):
    """The guard must actually fail when leakage is introduced."""
    import retailml.features as features_module

    series = dataset.series(*dataset.series_keys()[0])
    original = features_module.build_features

    def leaky(records, config=None):
        matrix = original(records, config)
        ordered = sorted(records, key=lambda r: r.date)
        targets = [r.units for r in ordered]
        # Append tomorrow's target: textbook leakage.
        matrix.names = matrix.names + ["tomorrow"]
        for i, row in enumerate(matrix.X):
            future_index = min(matrix.warmup + i + 1, len(targets) - 1)
            row.append(targets[future_index])
        return matrix

    features_module.build_features = leaky
    try:
        with pytest.raises(AssertionError, match="leakage"):
            features_module.assert_no_leakage(series)
    finally:
        features_module.build_features = original


# --- backtesting -------------------------------------------------------------
def test_rolling_origins_leave_a_full_horizon():
    config = BacktestConfig(horizon=14, step=14, min_train_days=100, max_origins=10)
    origins = rolling_origins(200, config)
    assert origins
    assert all(o + config.horizon <= 200 for o in origins)
    assert all(o >= config.min_train_days for o in origins)


def test_backtest_produces_multiple_origins(dataset):
    series = dataset.series(*dataset.series_keys()[0])
    result = backtest_series(series, SeasonalNaive, BacktestConfig(max_origins=4))
    assert len(result.origins) > 1
    assert result.error is None


def test_backtest_reports_variance_across_origins(dataset):
    """A model with a good mean and a huge spread is a coin flip."""
    series = dataset.series(*dataset.series_keys()[0])
    result = backtest_series(series, SeasonalNaive, BacktestConfig(max_origins=5))
    assert result.wape_stdev is not None
    assert result.mean_wape is not None


def test_backtest_refuses_a_series_that_is_too_short():
    dataset = generate_dataset(n_skus=1, n_stores=1, days=60, n_customers=10)
    result = backtest_series(
        dataset.series(*dataset.series_keys()[0]), SeasonalNaive, BacktestConfig()
    )
    assert result.origins == []
    assert "needs at least" in (result.error or "")


def test_training_never_includes_the_evaluation_window(dataset):
    """The structural guarantee that makes the backtest meaningful."""
    series = sorted(dataset.series(*dataset.series_keys()[0]), key=lambda r: r.date)
    config = BacktestConfig(horizon=14, step=28, max_origins=3)
    result = backtest_series(series, SeasonalNaive, config)
    for origin in result.origins:
        cutoff_index = [r.date for r in series].index(origin.cutoff)
        assert origin.train_size <= cutoff_index


def test_feature_model_beats_seasonal_naive(dataset):
    """The synthetic data has price and promotion structure only features see.

    If this fails, either the generator stopped producing learnable structure or
    the feature pipeline stopped exposing it - both worth failing the build for.
    """
    series_by_key = {key: dataset.series(*key) for key in dataset.series_keys()}
    results = backtest_dataset(
        series_by_key,
        {"seasonal_naive": SeasonalNaive, "ridge": lambda: RidgeRegression(alpha=5.0)},
        BacktestConfig(horizon=14, step=28, max_origins=4),
    )
    comparisons = {c.model: c for c in compare_models(results)}
    assert comparisons["ridge"].wape < comparisons["seasonal_naive"].wape


def test_every_model_runs_in_the_backtest(dataset):
    series_by_key = {dataset.series_keys()[0]: dataset.series(*dataset.series_keys()[0])}
    results = backtest_dataset(
        series_by_key,
        {
            "naive": Naive,
            "seasonal_naive": SeasonalNaive,
            "seasonal_ma": SeasonalMovingAverage,
            "drift": Drift,
            "holt_winters": HoltWinters,
            "ridge": lambda: RidgeRegression(alpha=5.0),
        },
        BacktestConfig(horizon=14, step=28, max_origins=3),
    )
    for name, runs in results.items():
        assert runs[0].origins, f"{name} produced no origins: {runs[0].error}"


def test_comparison_is_sorted_by_accuracy(dataset):
    series_by_key = {dataset.series_keys()[0]: dataset.series(*dataset.series_keys()[0])}
    results = backtest_dataset(
        series_by_key,
        {"naive": Naive, "seasonal_naive": SeasonalNaive},
        BacktestConfig(max_origins=3),
    )
    comparisons = compare_models(results)
    wapes = [c.wape for c in comparisons if c.wape is not None]
    assert wapes == sorted(wapes)


def test_scores_are_finite(dataset):
    series = dataset.series(*dataset.series_keys()[0])
    result = backtest_series(series, HoltWinters, BacktestConfig(max_origins=3))
    for origin in result.origins:
        assert math.isfinite(origin.score.mae)
        assert math.isfinite(origin.score.rmse)


def test_future_features_never_read_placeholder_targets():
    from datetime import date, timedelta

    from retailml.data import SalesRecord
    from retailml.features import FeatureConfig, future_features

    start = date(2025, 1, 6)
    history = [
        SalesRecord(date=start + timedelta(days=i), sku="S", store="T", category="c", units=100.0 + (i % 7), price=10.0, on_promotion=False)
        for i in range(120)
    ]
    future = [
        SalesRecord(date=start + timedelta(days=120 + i), sku="S", store="T", category="c", units=0.0, price=10.0, on_promotion=False)
        for i in range(28)
    ]
    config = FeatureConfig(calendar=False, price=False, promotion=False, fourier_order=0)
    rows = future_features(history, future, config)
    assert len(rows) == 28
    # Every lag value is an observed weekday level (100-106), never the 0 placeholder.
    lag_columns = len(config.lags)
    assert all(100.0 <= value <= 106.0 for row in rows for value in row[:lag_columns])
