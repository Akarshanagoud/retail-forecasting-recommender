"""Drift detection.

Models decay, and retail models decay on a schedule: a new competitor opens, a
supplier changes pack size, a category gets reset. The question is never
whether drift happened but whether it happened *enough to matter*, and there
are two very different answers:

* **Data drift** - the inputs moved. Detectable immediately, from feature
  values alone, with no labels required.
* **Performance drift** - the model got worse. The thing you actually care
  about, but only observable once actuals arrive, which in retail means days
  or weeks late.

Both are implemented because they are used differently. Data drift is an early
warning that justifies looking; performance drift is the evidence that
justifies retraining. Retraining on data drift alone is how teams end up
retraining weekly for no measured benefit — the inputs shift constantly and the
model is often fine.

Implemented from the definitions rather than imported, so the binning and the
distributional assumptions are visible: PSI in particular is sensitive to bin
count and to how zero-population bins are handled, and a library call hides
exactly the choice that determines whether it fires.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DriftVerdict(str, Enum):
    STABLE = "stable"
    #: Moved, but within the range where monitoring rather than action is right.
    WATCH = "watch"
    DRIFTED = "drifted"


@dataclass
class DriftResult:
    feature: str
    metric: str
    statistic: float
    verdict: DriftVerdict
    threshold: float
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def drifted(self) -> bool:
        return self.verdict is DriftVerdict.DRIFTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "metric": self.metric,
            "statistic": round(self.statistic, 5),
            "verdict": self.verdict.value,
            "threshold": self.threshold,
            "detail": self.detail,
        }


def _quantile_bins(reference: list[float], bins: int) -> list[float]:
    """Bin edges at reference quantiles.

    Quantile bins rather than equal-width, because retail features are heavily
    skewed: equal-width bins on a long-tailed price distribution put 95% of the
    mass in one bin and PSI then measures almost nothing.
    """
    ordered = sorted(reference)
    if not ordered:
        return []
    edges = [
        ordered[min(int(len(ordered) * i / bins), len(ordered) - 1)]
        for i in range(1, bins)
    ]
    # Collapse duplicate edges, which appear when a feature is mostly constant.
    unique: list[float] = []
    for edge in edges:
        if not unique or edge > unique[-1]:
            unique.append(edge)
    return unique


def _bin_index(value: float, edges: list[float]) -> int:
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


def population_stability_index(
    reference: list[float],
    current: list[float],
    bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """PSI between two samples.

    Convention: < 0.1 stable, 0.1-0.25 watch, > 0.25 drifted. ``epsilon``
    floors empty bins — without it a single unseen bin sends PSI to infinity
    and the alert is meaningless.
    """
    if not reference or not current:
        return 0.0
    edges = _quantile_bins(reference, bins)
    n_bins = len(edges) + 1

    reference_counts = [0] * n_bins
    current_counts = [0] * n_bins
    for value in reference:
        reference_counts[_bin_index(value, edges)] += 1
    for value in current:
        current_counts[_bin_index(value, edges)] += 1

    total_reference = len(reference)
    total_current = len(current)
    psi = 0.0
    for expected, actual in zip(reference_counts, current_counts, strict=True):
        expected_share = max(expected / total_reference, epsilon)
        actual_share = max(actual / total_current, epsilon)
        psi += (actual_share - expected_share) * math.log(actual_share / expected_share)
    return psi


def ks_statistic(reference: list[float], current: list[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: max CDF gap.

    Complements PSI: KS is binning-free and sensitive to a shift anywhere in
    the distribution, while PSI is more interpretable and more stable on small
    samples. Disagreement between them is informative in itself.
    """
    if not reference or not current:
        return 0.0
    combined = sorted(set(reference) | set(current))
    reference_sorted = sorted(reference)
    current_sorted = sorted(current)

    def cdf(values: list[float], point: float) -> float:
        low, high = 0, len(values)
        while low < high:
            mid = (low + high) // 2
            if values[mid] <= point:
                low = mid + 1
            else:
                high = mid
        return low / len(values)

    return max(abs(cdf(reference_sorted, x) - cdf(current_sorted, x)) for x in combined)


def ks_critical_value(n1: int, n2: int, alpha: float = 0.05) -> float:
    """Approximate KS critical value, for turning the statistic into a verdict."""
    if n1 == 0 or n2 == 0:
        return 1.0
    coefficients = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}
    c = coefficients.get(alpha, 1.36)
    return c * math.sqrt((n1 + n2) / (n1 * n2))


def check_feature_drift(
    reference: dict[str, list[float]],
    current: dict[str, list[float]],
    psi_watch: float = 0.10,
    psi_drift: float = 0.25,
) -> list[DriftResult]:
    """PSI per feature, with a KS statistic recorded alongside."""
    results: list[DriftResult] = []
    for feature in sorted(set(reference) & set(current)):
        reference_values = reference[feature]
        current_values = current[feature]
        psi = population_stability_index(reference_values, current_values)
        ks = ks_statistic(reference_values, current_values)

        if psi >= psi_drift:
            verdict = DriftVerdict.DRIFTED
        elif psi >= psi_watch:
            verdict = DriftVerdict.WATCH
        else:
            verdict = DriftVerdict.STABLE

        results.append(
            DriftResult(
                feature=feature,
                metric="psi",
                statistic=psi,
                verdict=verdict,
                threshold=psi_drift,
                detail={
                    "ks": round(ks, 5),
                    "ks_critical_05": round(
                        ks_critical_value(len(reference_values), len(current_values)), 5
                    ),
                    "reference_mean": round(statistics.fmean(reference_values), 4)
                    if reference_values
                    else None,
                    "current_mean": round(statistics.fmean(current_values), 4)
                    if current_values
                    else None,
                    "reference_n": len(reference_values),
                    "current_n": len(current_values),
                },
            )
        )
    results.sort(key=lambda r: -r.statistic)
    return results


def check_performance_drift(
    baseline_error: float,
    recent_error: float,
    relative_tolerance: float = 0.20,
    watch_tolerance: float = 0.10,
) -> DriftResult:
    """Compare recent error against the error the model was accepted at.

    Relative rather than absolute, because the tolerable error depends entirely
    on the series: 15% WAPE is excellent for an intermittent SKU and terrible
    for a high-volume staple.
    """
    ratio = (
        0.0 if baseline_error <= 0 else (recent_error - baseline_error) / baseline_error
    )

    if ratio >= relative_tolerance:
        verdict = DriftVerdict.DRIFTED
    elif ratio >= watch_tolerance:
        verdict = DriftVerdict.WATCH
    else:
        verdict = DriftVerdict.STABLE

    return DriftResult(
        feature="__performance__",
        metric="relative_error_increase",
        statistic=ratio,
        verdict=verdict,
        threshold=relative_tolerance,
        detail={
            "baseline_error": round(baseline_error, 5),
            "recent_error": round(recent_error, 5),
        },
    )


@dataclass
class DriftReport:
    feature_results: list[DriftResult] = field(default_factory=list)
    performance: DriftResult | None = None

    @property
    def drifted_features(self) -> list[str]:
        return [r.feature for r in self.feature_results if r.drifted]

    @property
    def performance_drifted(self) -> bool:
        return self.performance is not None and self.performance.drifted

    @property
    def verdict(self) -> DriftVerdict:
        if self.performance_drifted or self.drifted_features:
            return DriftVerdict.DRIFTED
        if any(r.verdict is DriftVerdict.WATCH for r in self.feature_results):
            return DriftVerdict.WATCH
        return DriftVerdict.STABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "drifted_features": self.drifted_features,
            "performance_drifted": self.performance_drifted,
            "features": [r.to_dict() for r in self.feature_results],
            "performance": self.performance.to_dict() if self.performance else None,
        }

    def render(self) -> str:
        lines = [f"overall verdict: {self.verdict.value}"]
        if self.performance is not None:
            lines.append(
                f"performance    : {self.performance.verdict.value} "
                f"({self.performance.statistic:+.1%} vs baseline error)"
            )
        lines.append(f"{'FEATURE':<28}{'PSI':>9}{'KS':>9}  VERDICT")
        for result in self.feature_results[:15]:
            lines.append(
                f"{result.feature:<28}{result.statistic:>9.4f}"
                f"{result.detail.get('ks', 0):>9.4f}  {result.verdict.value}"
            )
        return "\n".join(lines)


def columns_from_matrix(names: list[str], rows: list[list[float]]) -> dict[str, list[float]]:
    """Transpose a feature matrix into per-feature columns for drift checks."""
    if not rows:
        return {name: [] for name in names}
    return {
        name: [row[i] for row in rows] for i, name in enumerate(names)
    }
