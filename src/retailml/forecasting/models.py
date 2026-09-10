"""Forecasting models.

The baselines are here because they are the only honest yardstick. A gradient
boosting model with 4% WAPE sounds impressive until seasonal naive scores 4.2%
on the same series, at which point the model is 200 lines of code and a
retraining pipeline bought for nothing. Every model in this package is scored
against `SeasonalNaive` by default, and MASE reports the ratio directly.

Holt-Winters and ridge regression are implemented directly rather than
imported. Both are short enough to be readable, and having them in the repo
means the seasonal decomposition and the regularisation path are inspectable
rather than being a call into a library. A scikit-learn adapter is available
for the models where reimplementation would be pointless.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any


class Forecaster(abc.ABC):
    """Fits on a history and predicts the next ``horizon`` periods."""

    name: str = "forecaster"
    #: Whether the model consumes exogenous features (price, promotion, calendar).
    uses_features: bool = False

    @abc.abstractmethod
    def fit(self, y: list[float], X: list[list[float]] | None = None) -> Forecaster:
        ...

    @abc.abstractmethod
    def predict(self, horizon: int, X: list[list[float]] | None = None) -> list[float]:
        ...

    def fit_predict(
        self,
        y: list[float],
        horizon: int,
        X: list[list[float]] | None = None,
        X_future: list[list[float]] | None = None,
    ) -> list[float]:
        return self.fit(y, X).predict(horizon, X_future)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "uses_features": self.uses_features}


def _clip(values: list[float]) -> list[float]:
    """Demand cannot be negative. Clipping is not cosmetic: a negative forecast
    propagates into a nonsensical replenishment order."""
    return [max(0.0, v) for v in values]


class Naive(Forecaster):
    """Last observed value, repeated. The floor for any forecast."""

    name = "naive"

    def __init__(self) -> None:
        self._last = 0.0

    def fit(self, y: list[float], X: list[list[float]] | None = None) -> Naive:
        if not y:
            raise ValueError("cannot fit on an empty series")
        self._last = y[-1]
        return self

    def predict(self, horizon: int, X: list[list[float]] | None = None) -> list[float]:
        return _clip([self._last] * horizon)


class SeasonalNaive(Forecaster):
    """The same day last week. The baseline that actually has to be beaten.

    In daily retail this is a strong forecast: day-of-week is usually the
    single largest source of variation, and this captures it exactly with no
    parameters to fit or get wrong.
    """

    name = "seasonal_naive"

    def __init__(self, season_length: int = 7) -> None:
        self.season_length = season_length
        self._season: list[float] = []

    def fit(self, y: list[float], X: list[list[float]] | None = None) -> SeasonalNaive:
        if not y:
            raise ValueError("cannot fit on an empty series")
        if len(y) < self.season_length:
            # Too short for a full season: fall back to the mean rather than
            # inventing a seasonal pattern from partial data.
            mean = sum(y) / len(y)
            self._season = [mean] * self.season_length
        else:
            self._season = y[-self.season_length :]
        return self

    def predict(self, horizon: int, X: list[list[float]] | None = None) -> list[float]:
        return _clip(
            [self._season[i % self.season_length] for i in range(horizon)]
        )


class SeasonalMovingAverage(Forecaster):
    """Average of the last ``k`` occurrences of each position in the season.

    Less reactive than seasonal naive and correspondingly less noisy, which
    usually wins on volatile or intermittent series.
    """

    name = "seasonal_moving_average"

    def __init__(self, season_length: int = 7, seasons: int = 4) -> None:
        self.season_length = season_length
        self.seasons = seasons
        self._profile: list[float] = []

    def fit(
        self, y: list[float], X: list[list[float]] | None = None
    ) -> SeasonalMovingAverage:
        if not y:
            raise ValueError("cannot fit on an empty series")
        profile: list[float] = []
        for position in range(self.season_length):
            # Walk backwards through matching positions in the season.
            samples: list[float] = []
            index = len(y) - self.season_length + position
            while index >= 0 and len(samples) < self.seasons:
                samples.append(y[index])
                index -= self.season_length
            profile.append(sum(samples) / len(samples) if samples else 0.0)
        self._profile = profile
        return self

    def predict(self, horizon: int, X: list[list[float]] | None = None) -> list[float]:
        return _clip([self._profile[i % self.season_length] for i in range(horizon)])


class Drift(Forecaster):
    """Extrapolates the average slope between the first and last observation."""

    name = "drift"

    def __init__(self) -> None:
        self._last = 0.0
        self._slope = 0.0

    def fit(self, y: list[float], X: list[list[float]] | None = None) -> Drift:
        if not y:
            raise ValueError("cannot fit on an empty series")
        self._last = y[-1]
        self._slope = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0.0
        return self

    def predict(self, horizon: int, X: list[list[float]] | None = None) -> list[float]:
        return _clip([self._last + self._slope * (i + 1) for i in range(horizon)])


class HoltWinters(Forecaster):
    """Triple exponential smoothing with multiplicative seasonality.

    Multiplicative rather than additive because retail seasonality scales with
    volume: a Saturday is not "a Tuesday plus 40 units", it is "a Tuesday times
    1.4", and the difference compounds as soon as the trend is non-zero.

    Implemented directly so the initialisation and the update equations are
    visible. Seasonal initialisation uses per-period averages across whole
    seasons, which is materially more stable than seeding from the first season
    alone on noisy retail data.
    """

    name = "holt_winters"

    def __init__(
        self,
        season_length: int = 7,
        alpha: float = 0.15,
        beta: float = 0.02,
        gamma: float = 0.15,
        damped: float = 0.92,
    ) -> None:
        for parameter, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if not 0 <= value <= 1:
                raise ValueError(f"{parameter} must be between 0 and 1")
        self.season_length = season_length
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        #: Damping on the trend. An undamped trend extrapolated 28 days out is
        #: how a forecast ends up predicting negative or absurd demand.
        self.damped = damped
        self._level = 0.0
        self._trend = 0.0
        self._season: list[float] = []
        self.fitted: list[float] = []

    def fit(self, y: list[float], X: list[list[float]] | None = None) -> HoltWinters:
        m = self.season_length
        if len(y) < 2 * m:
            # Not enough data for a seasonal fit; degrade to a level model
            # rather than fabricating a seasonal profile.
            self._level = sum(y) / len(y) if y else 0.0
            self._trend = 0.0
            self._season = [1.0] * m
            self.fitted = list(y)
            return self

        seasons = len(y) // m
        season_averages = [
            sum(y[i * m : (i + 1) * m]) / m for i in range(seasons)
        ]
        overall = sum(season_averages) / seasons or 1.0

        # Per-position seasonal index, averaged across whole seasons.
        seasonal: list[float] = []
        for position in range(m):
            ratios = [
                y[i * m + position] / season_averages[i]
                for i in range(seasons)
                if season_averages[i] > 0
            ]
            seasonal.append(sum(ratios) / len(ratios) if ratios else 1.0)
        # Normalise so the season multiplies to a neutral effect.
        scale = sum(seasonal) / m or 1.0
        seasonal = [s / scale for s in seasonal]

        level = season_averages[0] or overall
        trend = (season_averages[-1] - season_averages[0]) / (max(seasons - 1, 1) * m)
        fitted: list[float] = []

        for t, value in enumerate(y):
            index = t % m
            previous_seasonal = seasonal[index] or 1.0
            fitted.append((level + trend) * previous_seasonal)

            previous_level = level
            level = self.alpha * (value / previous_seasonal) + (1 - self.alpha) * (
                level + trend
            )
            trend = self.beta * (level - previous_level) + (1 - self.beta) * trend
            if level > 0:
                seasonal[index] = (
                    self.gamma * (value / level) + (1 - self.gamma) * previous_seasonal
                )

        self._level, self._trend, self._season = level, trend, seasonal
        self.fitted = fitted
        return self

    def predict(self, horizon: int, X: list[list[float]] | None = None) -> list[float]:
        m = self.season_length
        out: list[float] = []
        for h in range(1, horizon + 1):
            # Geometric damping: the trend contribution converges rather than
            # growing without bound over a long horizon.
            damping = sum(self.damped**i for i in range(1, h + 1))
            out.append((self._level + damping * self._trend) * self._season[(h - 1) % m])
        return _clip(out)


class RidgeRegression(Forecaster):
    """Ridge regression on exogenous features, solved by normal equations.

    Regularised rather than plain least squares because the feature matrix here
    is deliberately collinear: day-of-week dummies, lags at 1/7/14 and rolling
    means all encode overlapping information, and unregularised OLS on that
    produces unstable coefficients that swing wildly between retrainings.

    Gaussian elimination with partial pivoting is used directly so the solve
    has no dependency, and the ridge penalty guarantees the system is
    non-singular for any lambda > 0.
    """

    name = "ridge"
    uses_features = True

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coefficients: list[float] = []
        self.intercept: float = 0.0
        self._means: list[float] = []
        self._scales: list[float] = []

    def _standardise(self, X: list[list[float]], fit: bool) -> list[list[float]]:
        """Scale features to comparable ranges.

        Without this the ridge penalty is applied unevenly: a price in the tens
        and a binary promotion flag would be shrunk by wildly different amounts
        for the same lambda.
        """
        if fit:
            columns = list(zip(*X, strict=True)) if X else []
            self._means = [sum(c) / len(c) for c in columns]
            self._scales = []
            for column, mean in zip(columns, self._means, strict=True):
                variance = sum((v - mean) ** 2 for v in column) / len(column)
                self._scales.append(math.sqrt(variance) or 1.0)
        return [
            [
                (value - mean) / scale
                for value, mean, scale in zip(row, self._means, self._scales, strict=True)
            ]
            for row in X
        ]

    def fit(self, y: list[float], X: list[list[float]] | None = None) -> RidgeRegression:
        if X is None:
            raise ValueError("RidgeRegression requires a feature matrix")
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if not X:
            raise ValueError("cannot fit on an empty series")

        Z = self._standardise(X, fit=True)
        n_features = len(Z[0])
        target_mean = sum(y) / len(y) if self.fit_intercept else 0.0
        centred = [value - target_mean for value in y]

        # Normal equations: (Z'Z + alpha*I) beta = Z'y
        gram = [[0.0] * n_features for _ in range(n_features)]
        rhs = [0.0] * n_features
        for row, target in zip(Z, centred, strict=True):
            for i in range(n_features):
                rhs[i] += row[i] * target
                for j in range(n_features):
                    gram[i][j] += row[i] * row[j]
        for i in range(n_features):
            gram[i][i] += self.alpha

        self.coefficients = _solve(gram, rhs)
        self.intercept = target_mean
        return self

    def predict(self, horizon: int, X: list[list[float]] | None = None) -> list[float]:
        if X is None:
            raise ValueError("RidgeRegression requires future features to predict")
        if len(X) < horizon:
            raise ValueError(f"need {horizon} future feature rows, got {len(X)}")
        Z = self._standardise(X[:horizon], fit=False)
        return _clip(
            [
                self.intercept
                + sum(c * v for c, v in zip(self.coefficients, row, strict=True))
                for row in Z
            ]
        )


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(rhs)
    augmented = [row[:] + [value] for row, value in zip(matrix, rhs, strict=True)]

    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            # Ridge guarantees this cannot happen for alpha > 0; guard anyway.
            continue
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(column + 1, n):
            factor = augmented[row][column] / augmented[column][column]
            for k in range(column, n + 1):
                augmented[row][k] -= factor * augmented[column][k]

    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        if abs(augmented[row][row]) < 1e-12:
            continue
        total = augmented[row][n] - sum(
            augmented[row][c] * solution[c] for c in range(row + 1, n)
        )
        solution[row] = total / augmented[row][row]
    return solution


class SklearnForecaster(Forecaster):
    """Adapter for any scikit-learn regressor.

        from sklearn.ensemble import HistGradientBoostingRegressor
        model = SklearnForecaster(HistGradientBoostingRegressor(max_iter=200))

    Reimplementing gradient boosting would prove nothing; the interesting part
    is that it enters the same backtest and is scored against the same
    baselines as everything else.
    """

    name = "sklearn"
    uses_features = True

    def __init__(self, estimator: Any = None) -> None:
        if estimator is None:
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "SklearnForecaster needs scikit-learn: pip install 'retailml[ml]'"
                ) from exc
            estimator = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.08, random_state=0
            )
        self.estimator = estimator
        self.name = f"sklearn:{type(estimator).__name__}"

    def fit(self, y: list[float], X: list[list[float]] | None = None) -> SklearnForecaster:
        if X is None:
            raise ValueError("SklearnForecaster requires a feature matrix")
        self.estimator.fit(X, y)
        return self

    def predict(self, horizon: int, X: list[list[float]] | None = None) -> list[float]:
        if X is None:
            raise ValueError("SklearnForecaster requires future features")
        return _clip([float(v) for v in self.estimator.predict(X[:horizon])])


@dataclass
class EnsembleForecaster(Forecaster):
    """Weighted average of several forecasters.

    Averaging a seasonal baseline with a feature model is the cheapest reliable
    accuracy gain in retail: the baseline anchors the day-of-week pattern while
    the feature model supplies the promotion response, and the average is more
    robust than either when one of them is having a bad week.
    """

    models: list[Forecaster] = field(default_factory=list)
    weights: list[float] | None = None
    name: str = "ensemble"

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("an ensemble needs at least one model")
        if self.weights is not None and len(self.weights) != len(self.models):
            raise ValueError("weights and models must be the same length")
        self.uses_features = any(m.uses_features for m in self.models)
        self.name = "ensemble(" + "+".join(m.name for m in self.models) + ")"

    def fit(self, y: list[float], X: list[list[float]] | None = None) -> EnsembleForecaster:
        for model in self.models:
            model.fit(y, X if model.uses_features else None)
        return self

    def predict(self, horizon: int, X: list[list[float]] | None = None) -> list[float]:
        weights = self.weights or [1.0] * len(self.models)
        total_weight = sum(weights)
        predictions = [
            model.predict(horizon, X if model.uses_features else None)
            for model in self.models
        ]
        return _clip(
            [
                sum(w * p[i] for w, p in zip(weights, predictions, strict=True))
                / total_weight
                for i in range(horizon)
            ]
        )


MODELS: dict[str, Any] = {
    "naive": Naive,
    "seasonal_naive": SeasonalNaive,
    "seasonal_moving_average": SeasonalMovingAverage,
    "drift": Drift,
    "holt_winters": HoltWinters,
    "ridge": RidgeRegression,
    "sklearn": SklearnForecaster,
}


def build_model(name: str, **kwargs: Any) -> Forecaster:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)
