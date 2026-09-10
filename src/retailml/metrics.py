"""Forecast accuracy metrics.

Metric choice is a modelling decision disguised as a reporting decision, and
retail is where the disguise fails loudest:

* **MAPE is undefined on zero-demand days**, and retail is full of them. Slow
  movers are exactly the SKUs where forecasting is hard, so a metric that
  quietly drops them reports on the easy subset and calls it accuracy.
* **MAPE is asymmetric.** Over-forecasting is bounded at 100% error;
  under-forecasting is unbounded. Optimising it biases forecasts low, which in
  retail means stockouts.
* **MAE across SKUs is dominated by the high-volume ones.** Being 5 units out
  on a 500-unit SKU and on a 5-unit SKU are not the same error, and averaging
  raw MAE says they are.

So the default headline metric here is **WAPE** (volume-weighted, defined at
zero) and **MASE** (scaled against a seasonal naive baseline, so a number below
1.0 means "better than assuming next Tuesday equals last Tuesday" — which is
the only comparison that means anything across series of different scales).

Pinball loss is included because a retail forecast is usually consumed as a
stocking decision, and the cost of over- and under-stocking is not symmetric.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _validate(actual: list[float], predicted: list[float]) -> None:
    if len(actual) != len(predicted):
        raise ValueError(
            f"length mismatch: {len(actual)} actuals vs {len(predicted)} predictions"
        )
    if not actual:
        raise ValueError("cannot score an empty series")


def mae(actual: list[float], predicted: list[float]) -> float:
    """Mean absolute error."""
    _validate(actual, predicted)
    return sum(abs(a - p) for a, p in zip(actual, predicted, strict=True)) / len(actual)


def rmse(actual: list[float], predicted: list[float]) -> float:
    """Root mean squared error. Penalises large misses more than MAE."""
    _validate(actual, predicted)
    return math.sqrt(
        sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=True)) / len(actual)
    )


def bias(actual: list[float], predicted: list[float]) -> float:
    """Mean signed error. Positive means the forecast runs high.

    Reported separately from MAE because a forecast can have excellent MAE and
    a persistent bias, and only the bias shows up as systematic overstock.
    """
    _validate(actual, predicted)
    return sum(p - a for a, p in zip(actual, predicted, strict=True)) / len(actual)


def mape(actual: list[float], predicted: list[float]) -> float | None:
    """Mean absolute percentage error over non-zero actuals.

    Returns ``None`` when every actual is zero, rather than a number that looks
    like an answer. The count of skipped points is available via
    :func:`zero_fraction`.
    """
    _validate(actual, predicted)
    pairs = [(a, p) for a, p in zip(actual, predicted, strict=True) if a != 0]
    if not pairs:
        return None
    return sum(abs(a - p) / abs(a) for a, p in pairs) / len(pairs)


def smape(actual: list[float], predicted: list[float]) -> float:
    """Symmetric MAPE, bounded in [0, 2]. Defined when actual and forecast differ."""
    _validate(actual, predicted)
    total = 0.0
    counted = 0
    for a, p in zip(actual, predicted, strict=True):
        denominator = (abs(a) + abs(p)) / 2
        if denominator == 0:
            continue  # both zero: a perfect forecast, contributing zero error
        total += abs(a - p) / denominator
        counted += 1
    return total / counted if counted else 0.0


def wape(actual: list[float], predicted: list[float]) -> float:
    """Weighted absolute percentage error: sum|error| / sum|actual|.

    The retail default. Defined when individual actuals are zero, and weighted
    by volume so a 5-unit miss on a 500-unit SKU does not count the same as a
    5-unit miss on a 5-unit SKU.
    """
    _validate(actual, predicted)
    denominator = sum(abs(a) for a in actual)
    if denominator == 0:
        return 0.0 if all(p == 0 for p in predicted) else float("inf")
    return sum(abs(a - p) for a, p in zip(actual, predicted, strict=True)) / denominator


def mase(
    actual: list[float],
    predicted: list[float],
    history: list[float],
    season_length: int = 7,
) -> float | None:
    """Mean absolute scaled error against an in-sample seasonal naive baseline.

    Below 1.0 means the model beats "same day last week" on the training data's
    own difficulty scale. This is the metric that survives comparison across
    series of wildly different volumes.

    Returns ``None`` when the history is too short or perfectly flat, since the
    scale factor would be zero.
    """
    _validate(actual, predicted)
    if len(history) <= season_length:
        return None
    differences = [
        abs(history[i] - history[i - season_length])
        for i in range(season_length, len(history))
    ]
    scale = sum(differences) / len(differences) if differences else 0.0
    if scale == 0:
        return None
    return mae(actual, predicted) / scale


def pinball_loss(
    actual: list[float], predicted: list[float], quantile: float = 0.9
) -> float:
    """Quantile loss. The metric for a forecast consumed as a stocking level.

    At ``quantile=0.9`` under-forecasting is penalised nine times as heavily as
    over-forecasting, which is the right shape when a stockout costs more than
    a day of holding.
    """
    _validate(actual, predicted)
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between 0 and 1")
    total = 0.0
    for a, p in zip(actual, predicted, strict=True):
        total += (a - p) * quantile if a >= p else (p - a) * (1 - quantile)
    return total / len(actual)


def coverage(actual: list[float], lower: list[float], upper: list[float]) -> float:
    """Fraction of actuals inside the prediction interval.

    A nominal 80% interval that covers 40% of actuals is not a forecast with
    wide error bars, it is a broken uncertainty estimate.
    """
    _validate(actual, lower)
    _validate(actual, upper)
    inside = sum(
        1
        for a, low, high in zip(actual, lower, upper, strict=True)
        if low <= a <= high
    )
    return inside / len(actual)


def zero_fraction(actual: list[float]) -> float:
    """Fraction of zero-demand periods. Context for whether MAPE is usable."""
    return sum(1 for a in actual if a == 0) / len(actual) if actual else 0.0


@dataclass
class ForecastScore:
    """Every metric for one forecast, plus the context to interpret them."""

    n: int
    mae: float
    rmse: float
    bias: float
    wape: float
    smape: float
    mape: float | None
    mase: float | None
    pinball_90: float
    zero_fraction: float
    coverage_80: float | None = None
    #: Raw totals, carried so cross-series aggregation can recompute WAPE from
    #: sums. An average of ratios is not the ratio of totals, and only the
    #: latter is meaningful when series have different volumes.
    abs_error_sum: float = 0.0
    actual_sum: float = 0.0

    @property
    def beats_seasonal_naive(self) -> bool | None:
        """MASE below 1.0: better than assuming last week repeats."""
        return None if self.mase is None else self.mase < 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "bias": round(self.bias, 4),
            "wape": round(self.wape, 4),
            "smape": round(self.smape, 4),
            "mape": round(self.mape, 4) if self.mape is not None else None,
            "mase": round(self.mase, 4) if self.mase is not None else None,
            "pinball_90": round(self.pinball_90, 4),
            "zero_fraction": round(self.zero_fraction, 4),
            "coverage_80": round(self.coverage_80, 4)
            if self.coverage_80 is not None
            else None,
            "beats_seasonal_naive": self.beats_seasonal_naive,
        }


def score_forecast(
    actual: list[float],
    predicted: list[float],
    history: list[float] | None = None,
    season_length: int = 7,
    lower: list[float] | None = None,
    upper: list[float] | None = None,
) -> ForecastScore:
    """Compute the full metric set for one forecast."""
    return ForecastScore(
        n=len(actual),
        abs_error_sum=sum(abs(a - p) for a, p in zip(actual, predicted, strict=True)),
        actual_sum=sum(abs(a) for a in actual),
        mae=mae(actual, predicted),
        rmse=rmse(actual, predicted),
        bias=bias(actual, predicted),
        wape=wape(actual, predicted),
        smape=smape(actual, predicted),
        mape=mape(actual, predicted),
        mase=mase(actual, predicted, history, season_length) if history else None,
        pinball_90=pinball_loss(actual, predicted, 0.9),
        zero_fraction=zero_fraction(actual),
        coverage_80=coverage(actual, lower, upper) if lower and upper else None,
    )


def aggregate_scores(scores: list[ForecastScore]) -> dict[str, Any]:
    """Aggregate across series.

    WAPE is recomputed from totals rather than averaged, because an average of
    ratios is not the ratio of totals and only the latter is meaningful when
    series have different volumes.
    """
    if not scores:
        return {"series": 0}

    total_n = sum(s.n for s in scores)
    absolute_error = sum(s.abs_error_sum for s in scores)
    volume = sum(s.actual_sum for s in scores)

    valid_mase = [s.mase for s in scores if s.mase is not None]
    return {
        "series": len(scores),
        "points": total_n,
        "mae": round(sum(s.mae * s.n for s in scores) / total_n, 4),
        "rmse": round(
            math.sqrt(sum((s.rmse**2) * s.n for s in scores) / total_n), 4
        ),
        "bias": round(sum(s.bias * s.n for s in scores) / total_n, 4),
        "wape": round(absolute_error / volume, 4) if volume else None,
        "mase": round(sum(valid_mase) / len(valid_mase), 4) if valid_mase else None,
        "beat_seasonal_naive": sum(
            1 for s in scores if s.beats_seasonal_naive
        ),
        "mase_available": len(valid_mase),
        "mean_zero_fraction": round(
            sum(s.zero_fraction for s in scores) / len(scores), 4
        ),
    }
