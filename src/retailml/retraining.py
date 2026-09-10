"""Automated retraining policy.

The gap this closes is the one between "we detected drift" and "we did
something about it, and it was the right thing".

Two failure modes bracket the problem. Retrain on every drift signal and you
retrain constantly for no measured benefit, because inputs shift all the time
and the model is usually fine. Retrain never, and accuracy decays until someone
notices in a business review months later.

So the policy here requires **evidence and improvement**:

1. A trigger fires (performance drift, sustained feature drift, or a schedule).
2. A candidate is trained and backtested on the *same* rolling origins as the
   incumbent, so the comparison is like for like.
3. The candidate is promoted only if it beats the incumbent by a margin.

Step 3 is what stops the production model becoming a random walk: without a
margin, run-to-run variance promotes a new model roughly half the time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .drift import DriftReport, DriftVerdict


class TriggerReason(str, Enum):
    PERFORMANCE_DRIFT = "performance_drift"
    FEATURE_DRIFT = "feature_drift"
    SCHEDULE = "schedule"
    NO_PRODUCTION_MODEL = "no_production_model"
    MANUAL = "manual"


@dataclass
class RetrainingPolicy:
    #: Retrain when recent error exceeds the accepted baseline by this much.
    performance_tolerance: float = 0.20
    #: Number of drifted features that justifies retraining on inputs alone.
    #: Deliberately more than one: a single shifted feature is usually noise
    #: or a data-pipeline hiccup, not a reason to rebuild the model.
    min_drifted_features: int = 3
    #: Retrain at least this often regardless of signals.
    max_age_days: int = 30
    #: Improvement a candidate must show before it replaces the incumbent.
    min_improvement: float = 0.02
    #: Refuse to promote a candidate whose accuracy is worse than this, even
    #: if it beats a bad incumbent. A relative win over a broken model is not
    #: a reason to ship.
    max_acceptable_error: float = 0.60

    def to_dict(self) -> dict[str, Any]:
        return {
            "performance_tolerance": self.performance_tolerance,
            "min_drifted_features": self.min_drifted_features,
            "max_age_days": self.max_age_days,
            "min_improvement": self.min_improvement,
            "max_acceptable_error": self.max_acceptable_error,
        }


@dataclass
class TriggerDecision:
    should_retrain: bool
    reasons: list[TriggerReason] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_retrain": self.should_retrain,
            "reasons": [r.value for r in self.reasons],
            "detail": self.detail,
        }


def should_retrain(
    drift: DriftReport | None,
    model_age_days: float | None,
    has_production_model: bool,
    policy: RetrainingPolicy | None = None,
) -> TriggerDecision:
    """Decide whether the evidence justifies retraining."""
    settings = policy or RetrainingPolicy()
    reasons: list[TriggerReason] = []
    detail: dict[str, Any] = {}

    if not has_production_model:
        reasons.append(TriggerReason.NO_PRODUCTION_MODEL)

    if drift is not None:
        if drift.performance_drifted:
            reasons.append(TriggerReason.PERFORMANCE_DRIFT)
            detail["performance"] = (
                drift.performance.to_dict() if drift.performance else None
            )
        drifted = drift.drifted_features
        detail["drifted_features"] = drifted
        if len(drifted) >= settings.min_drifted_features:
            reasons.append(TriggerReason.FEATURE_DRIFT)
        elif drifted:
            # Recorded but not acted on: the distinction is the whole point.
            detail["feature_drift_below_threshold"] = (
                f"{len(drifted)} drifted feature(s), threshold is "
                f"{settings.min_drifted_features}"
            )

    if model_age_days is not None and model_age_days >= settings.max_age_days:
        reasons.append(TriggerReason.SCHEDULE)
        detail["model_age_days"] = round(model_age_days, 1)

    return TriggerDecision(should_retrain=bool(reasons), reasons=reasons, detail=detail)


@dataclass
class PromotionDecision:
    promoted: bool
    reason: str
    candidate_error: float
    incumbent_error: float | None
    improvement: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "reason": self.reason,
            "candidate_error": round(self.candidate_error, 5),
            "incumbent_error": round(self.incumbent_error, 5)
            if self.incumbent_error is not None
            else None,
            "improvement": round(self.improvement, 5)
            if self.improvement is not None
            else None,
        }


def evaluate_candidate(
    candidate_error: float,
    incumbent_error: float | None,
    policy: RetrainingPolicy | None = None,
) -> PromotionDecision:
    """Decide whether a freshly trained candidate should go to production."""
    settings = policy or RetrainingPolicy()

    if candidate_error > settings.max_acceptable_error:
        return PromotionDecision(
            promoted=False,
            reason=(
                f"candidate error {candidate_error:.3f} exceeds the absolute "
                f"ceiling of {settings.max_acceptable_error:.2f}"
            ),
            candidate_error=candidate_error,
            incumbent_error=incumbent_error,
            improvement=None,
        )

    if incumbent_error is None:
        return PromotionDecision(
            promoted=True,
            reason="no incumbent model in production",
            candidate_error=candidate_error,
            incumbent_error=None,
            improvement=None,
        )

    improvement = (incumbent_error - candidate_error) / max(abs(incumbent_error), 1e-9)
    if improvement < settings.min_improvement:
        return PromotionDecision(
            promoted=False,
            reason=(
                f"improvement {improvement:.1%} is below the "
                f"{settings.min_improvement:.0%} promotion gate"
            ),
            candidate_error=candidate_error,
            incumbent_error=incumbent_error,
            improvement=improvement,
        )

    return PromotionDecision(
        promoted=True,
        reason=f"improves error by {improvement:.1%}",
        candidate_error=candidate_error,
        incumbent_error=incumbent_error,
        improvement=improvement,
    )


@dataclass
class RetrainingRecord:
    """One pass of the retraining loop, for the audit trail."""

    timestamp: str
    trigger: TriggerDecision
    promotion: PromotionDecision | None = None
    model_name: str = ""
    candidate_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model": self.model_name,
            "candidate_version": self.candidate_version,
            "trigger": self.trigger.to_dict(),
            "promotion": self.promotion.to_dict() if self.promotion else None,
        }

    def render(self) -> str:
        lines = [f"retraining check at {self.timestamp}"]
        if not self.trigger.should_retrain:
            lines.append("  trigger   : no retraining needed")
            drifted = self.trigger.detail.get("drifted_features")
            if drifted:
                lines.append(f"  note      : {len(drifted)} feature(s) drifted, below threshold")
            return "\n".join(lines)

        lines.append(
            f"  trigger   : {', '.join(r.value for r in self.trigger.reasons)}"
        )
        if self.promotion is not None:
            verb = "promoted" if self.promotion.promoted else "held back"
            lines.append(f"  candidate : v{self.candidate_version} {verb}")
            lines.append(f"  reason    : {self.promotion.reason}")
        return "\n".join(lines)


def make_record(
    trigger: TriggerDecision,
    promotion: PromotionDecision | None = None,
    model_name: str = "",
    candidate_version: int | None = None,
) -> RetrainingRecord:
    return RetrainingRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        trigger=trigger,
        promotion=promotion,
        model_name=model_name,
        candidate_version=candidate_version,
    )


def summarise(records: list[RetrainingRecord]) -> dict[str, Any]:
    """Aggregate the retraining history.

    The ratio of triggers to promotions is the number worth watching: many
    triggers and few promotions means the drift thresholds are too sensitive
    and the pipeline is burning compute to confirm the model is fine.
    """
    triggered = [r for r in records if r.trigger.should_retrain]
    promoted = [r for r in records if r.promotion and r.promotion.promoted]
    reasons: dict[str, int] = {}
    for record in triggered:
        for reason in record.trigger.reasons:
            reasons[reason.value] = reasons.get(reason.value, 0) + 1

    return {
        "checks": len(records),
        "triggered": len(triggered),
        "promoted": len(promoted),
        "promotion_rate": round(len(promoted) / len(triggered), 4) if triggered else None,
        "trigger_reasons": reasons,
    }


def drift_verdict_to_action(verdict: DriftVerdict) -> str:
    """Plain-language mapping, for dashboards and runbooks."""
    return {
        DriftVerdict.STABLE: "no action",
        DriftVerdict.WATCH: "monitor; do not retrain on this alone",
        DriftVerdict.DRIFTED: "evaluate a retrained candidate",
    }[verdict]
