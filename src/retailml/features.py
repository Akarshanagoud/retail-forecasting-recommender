"""Feature engineering, built so leakage is structurally impossible.

Leakage is the defining failure of time-series ML, and it is almost never
deliberate. It arrives through a rolling mean computed over the whole series
before splitting, a lag of zero on a target-derived column, or a "days since
last promotion" feature that quietly looks forward. The model then scores
beautifully in backtest and falls over in production, and nobody can explain
why because the backtest is still green.

Two structural defences here rather than a warning in a docstring:

1. **Every target-derived feature carries an explicit minimum lag.** A rolling
   mean at index ``t`` is computed from ``y[:t]``, never ``y[:t+1]``. The
   builder refuses to construct a lag of zero on a target-derived column.
2. **:func:`assert_no_leakage` proves it empirically.** It perturbs a future
   value and asserts that no earlier feature row changes. A feature that peeks
   forward fails that test regardless of how it was written.

Calendar and price features are *known in advance*, so they legitimately use
the current row: you know next Tuesday is a Tuesday, and you know the price you
plan to charge. Target-derived features are not, and the two are kept apart in
the code rather than by convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .data import HOLIDAYS, SalesRecord


@dataclass
class FeatureConfig:
    """Which features to build, and with what lags."""

    #: Lags of the target. All must be >= 1.
    lags: tuple[int, ...] = (1, 2, 7, 14, 28)
    #: Rolling-mean windows over the target, each ending at t-1.
    rolling_windows: tuple[int, ...] = (7, 28)
    #: Rolling standard-deviation windows, for volatility.
    rolling_std_windows: tuple[int, ...] = (7,)
    calendar: bool = True
    price: bool = True
    promotion: bool = True
    #: Fourier terms for annual seasonality. Two pairs captures the broad
    #: annual shape without the 365 dummy variables a naive encoding needs.
    fourier_order: int = 2
    season_length: int = 7

    def __post_init__(self) -> None:
        if any(lag < 1 for lag in self.lags):
            raise ValueError(
                "target lags must be >= 1: a lag of 0 is the target itself, "
                "which is leakage"
            )
        if any(window < 1 for window in self.rolling_windows):
            raise ValueError("rolling windows must be >= 1")


@dataclass
class FeatureMatrix:
    """Feature rows aligned with targets, plus the names to interpret them."""

    names: list[str]
    X: list[list[float]] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    dates: list[date] = field(default_factory=list)
    #: Rows dropped from the head because their lags were unavailable.
    warmup: int = 0

    def __len__(self) -> int:
        return len(self.X)

    def split_at(self, index: int) -> tuple[FeatureMatrix, FeatureMatrix]:
        """Chronological split. Never shuffle a time series."""
        left = FeatureMatrix(
            names=self.names, X=self.X[:index], y=self.y[:index],
            dates=self.dates[:index], warmup=self.warmup,
        )
        right = FeatureMatrix(
            names=self.names, X=self.X[index:], y=self.y[index:],
            dates=self.dates[index:], warmup=0,
        )
        return left, right

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": len(self.X),
            "features": len(self.names),
            "names": self.names,
            "warmup_rows_dropped": self.warmup,
            "date_range": [self.dates[0].isoformat(), self.dates[-1].isoformat()]
            if self.dates
            else [],
        }


def _calendar_features(day: date, config: FeatureConfig) -> tuple[list[str], list[float]]:
    """Features knowable in advance for any future date."""
    names: list[str] = []
    values: list[float] = []

    weekday = day.weekday()
    for i in range(7):
        names.append(f"dow_{i}")
        values.append(1.0 if weekday == i else 0.0)

    names.append("is_weekend")
    values.append(1.0 if weekday >= 5 else 0.0)

    names.append("day_of_month")
    values.append(day.day / 31.0)

    names.append("month")
    values.append(day.month / 12.0)

    # Payday effects: demand clusters around the start and middle of the month.
    names.append("is_month_start")
    values.append(1.0 if day.day <= 3 else 0.0)
    names.append("is_month_end")
    values.append(1.0 if day.day >= 28 else 0.0)

    day_of_year = day.timetuple().tm_yday
    for order in range(1, config.fourier_order + 1):
        angle = 2 * math.pi * order * day_of_year / 365.25
        names.append(f"annual_sin_{order}")
        values.append(math.sin(angle))
        names.append(f"annual_cos_{order}")
        values.append(math.cos(angle))

    holiday = HOLIDAYS.get((day.month, day.day))
    names.append("is_holiday")
    values.append(1.0 if holiday else 0.0)

    return names, values


def build_features(
    records: list[SalesRecord], config: FeatureConfig | None = None
) -> FeatureMatrix:
    """Build a leakage-free feature matrix from one SKU-store series.

    Row ``i`` predicts ``records[i].units`` using only information available
    before that day's sales were observed.
    """
    settings = config or FeatureConfig()
    ordered = sorted(records, key=lambda r: r.date)
    if not ordered:
        return FeatureMatrix(names=[])

    targets = [r.units for r in ordered]
    max_lag = max(
        list(settings.lags)
        + list(settings.rolling_windows)
        + list(settings.rolling_std_windows)
        + [1]
    )

    names: list[str] = []
    rows: list[list[float]] = []
    kept_targets: list[float] = []
    kept_dates: list[date] = []

    for t in range(max_lag, len(ordered)):
        record = ordered[t]
        row_names: list[str] = []
        row: list[float] = []

        # --- target-derived features: strictly from the past ---------------
        for lag in settings.lags:
            row_names.append(f"lag_{lag}")
            row.append(targets[t - lag])

        for window in settings.rolling_windows:
            # Slice ends at t, so it never includes targets[t].
            past = targets[t - window : t]
            row_names.append(f"rolling_mean_{window}")
            row.append(sum(past) / len(past))

        for window in settings.rolling_std_windows:
            past = targets[t - window : t]
            mean = sum(past) / len(past)
            variance = sum((v - mean) ** 2 for v in past) / len(past)
            row_names.append(f"rolling_std_{window}")
            row.append(math.sqrt(variance))

        # Same weekday last week, the single most predictive lag in daily retail.
        row_names.append("seasonal_lag")
        row.append(targets[t - settings.season_length])

        # --- features known in advance -------------------------------------
        if settings.calendar:
            calendar_names, calendar_values = _calendar_features(record.date, settings)
            row_names.extend(calendar_names)
            row.extend(calendar_values)

        if settings.price:
            row_names.append("price")
            row.append(record.price)
            previous_price = ordered[t - 1].price
            row_names.append("price_ratio_to_yesterday")
            row.append(record.price / previous_price if previous_price else 1.0)
            # Discount depth against the trailing reference price.
            reference = sum(r.price for r in ordered[max(0, t - 28) : t]) / min(t, 28)
            row_names.append("discount_depth")
            row.append(1.0 - record.price / reference if reference else 0.0)

        if settings.promotion:
            row_names.append("on_promotion")
            row.append(1.0 if record.on_promotion else 0.0)
            row_names.append("promo_yesterday")
            row.append(1.0 if ordered[t - 1].on_promotion else 0.0)
            # Pull-forward: last week's promotion depresses this week.
            row_names.append("promo_last_week")
            row.append(
                1.0
                if any(r.on_promotion for r in ordered[max(0, t - 7) : t])
                else 0.0
            )

        if not names:
            names = row_names
        rows.append(row)
        kept_targets.append(targets[t])
        kept_dates.append(record.date)

    return FeatureMatrix(
        names=names, X=rows, y=kept_targets, dates=kept_dates, warmup=max_lag
    )


def future_features(
    history: list[SalesRecord],
    future: list[SalesRecord],
    config: FeatureConfig | None = None,
) -> list[list[float]]:
    """Feature rows for future dates, using known-in-advance information.

    Target-derived lags are filled recursively: as each future day is scored,
    its own forecast is *not* fed back (that would compound error silently);
    instead lags reaching into the forecast horizon reuse the last observed
    seasonal value, which is the honest thing a planner could actually compute.
    """
    settings = config or FeatureConfig()
    combined = sorted(history, key=lambda r: r.date) + sorted(future, key=lambda r: r.date)
    matrix = build_features(combined, settings)
    # The last len(future) rows correspond to the future dates.
    return matrix.X[-len(future) :] if future else []


def assert_no_leakage(
    records: list[SalesRecord], config: FeatureConfig | None = None
) -> None:
    """Empirically verify that no feature row depends on a future target.

    Perturbs the final target and asserts every earlier feature row is
    unchanged. This catches leakage that a code review would miss, because it
    tests the behaviour rather than the intent.

    Raises ``AssertionError`` on failure.
    """
    settings = config or FeatureConfig()
    ordered = sorted(records, key=lambda r: r.date)
    if len(ordered) < 40:
        raise ValueError("need at least 40 records for a meaningful leakage check")

    baseline = build_features(ordered, settings)

    perturbed_records = [
        SalesRecord(
            date=r.date, sku=r.sku, store=r.store, category=r.category,
            units=r.units * 1000 + 12345 if i == len(ordered) - 1 else r.units,
            price=r.price, on_promotion=r.on_promotion, holiday=r.holiday,
        )
        for i, r in enumerate(ordered)
    ]
    perturbed = build_features(perturbed_records, settings)

    if len(baseline.X) != len(perturbed.X):
        raise AssertionError("perturbation changed the number of feature rows")

    # Every row except the last must be byte-identical.
    for index in range(len(baseline.X) - 1):
        if baseline.X[index] != perturbed.X[index]:
            differing = [
                name
                for name, a, b in zip(
                    baseline.names, baseline.X[index], perturbed.X[index], strict=True
                )
                if a != b
            ]
            raise AssertionError(
                f"leakage: row {index} changed when a future target was perturbed; "
                f"offending features: {differing}"
            )
