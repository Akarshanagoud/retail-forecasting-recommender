"""retailml - retail demand forecasting and recommendation, end to end."""
from .backtest import (
    BacktestConfig,
    BacktestResult,
    backtest_dataset,
    backtest_series,
    compare_models,
    render_comparison,
)
from .data import Dataset, Interaction, SalesRecord, SeriesSpec, generate_dataset
from .drift import (
    DriftReport,
    DriftVerdict,
    check_feature_drift,
    check_performance_drift,
    ks_statistic,
    population_stability_index,
)
from .features import FeatureConfig, FeatureMatrix, assert_no_leakage, build_features
from .forecasting import (
    Drift,
    EnsembleForecaster,
    Forecaster,
    HoltWinters,
    Naive,
    RidgeRegression,
    SeasonalMovingAverage,
    SeasonalNaive,
    SklearnForecaster,
    build_model,
)
from .metrics import ForecastScore, aggregate_scores, mase, score_forecast, wape
from .recommender import (
    ItemItemRecommender,
    PopularityRecommender,
    RecommenderConfig,
    RecoScore,
    temporal_split,
)
from .retraining import (
    RetrainingPolicy,
    TriggerReason,
    evaluate_candidate,
    should_retrain,
)
from .tracking import ExperimentTracker, ModelRegistry

__version__ = "0.1.0"

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Dataset",
    "Drift",
    "DriftReport",
    "DriftVerdict",
    "EnsembleForecaster",
    "ExperimentTracker",
    "FeatureConfig",
    "FeatureMatrix",
    "ForecastScore",
    "Forecaster",
    "HoltWinters",
    "Interaction",
    "ItemItemRecommender",
    "ModelRegistry",
    "Naive",
    "PopularityRecommender",
    "RecoScore",
    "RecommenderConfig",
    "RetrainingPolicy",
    "RidgeRegression",
    "SalesRecord",
    "SeasonalMovingAverage",
    "SeasonalNaive",
    "SeriesSpec",
    "SklearnForecaster",
    "TriggerReason",
    "aggregate_scores",
    "assert_no_leakage",
    "backtest_dataset",
    "backtest_series",
    "build_features",
    "build_model",
    "check_feature_drift",
    "check_performance_drift",
    "compare_models",
    "evaluate_candidate",
    "generate_dataset",
    "ks_statistic",
    "mase",
    "population_stability_index",
    "render_comparison",
    "score_forecast",
    "should_retrain",
    "temporal_split",
    "wape",
    "__version__",
]
