"""Recommender evaluation.

Accuracy alone is a trap here. A recommender that shows the top ten sellers to
everyone scores respectably on recall — most purchases are of popular items —
while being commercially useless, because it recommends what customers would
have bought anyway and never surfaces the long tail.

So this reports accuracy *and* the two things accuracy hides:

* **Catalogue coverage** - what fraction of the catalogue ever gets recommended
  to anyone. A recommender covering 4% of the catalogue is a bestseller list
  with extra steps.
* **Novelty** - the mean self-information of recommended items. High novelty
  means the system is surfacing things a customer would not have found alone,
  which is the entire commercial case for having one.

The split is **temporal, not random**. A random split lets the model learn from
a customer's future purchases to predict their past, which inflates every
metric and cannot happen in production.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..data import Interaction


@dataclass
class RecoScore:
    users_evaluated: int
    k: int
    precision: float
    recall: float
    map_score: float
    ndcg: float
    hit_rate: float
    coverage: float
    novelty: float
    #: Users with no test items, skipped rather than counted as failures.
    skipped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "users_evaluated": self.users_evaluated,
            "skipped_users": self.skipped,
            "k": self.k,
            f"precision@{self.k}": round(self.precision, 4),
            f"recall@{self.k}": round(self.recall, 4),
            f"map@{self.k}": round(self.map_score, 4),
            f"ndcg@{self.k}": round(self.ndcg, 4),
            "hit_rate": round(self.hit_rate, 4),
            "catalogue_coverage": round(self.coverage, 4),
            "novelty": round(self.novelty, 4),
        }


def temporal_split(
    interactions: list[Interaction], test_fraction: float = 0.2
) -> tuple[list[Interaction], list[Interaction]]:
    """Split on time, not at random.

    A random split lets a model learn from a customer's future to predict their
    past. Every metric improves and none of it survives contact with
    production.
    """
    if not interactions:
        return [], []
    ordered = sorted(interactions, key=lambda i: i.date)
    cutoff_index = int(len(ordered) * (1 - test_fraction))
    cutoff_date: date = ordered[cutoff_index].date
    # Split on the date boundary so a single day never straddles the split.
    train = [i for i in ordered if i.date < cutoff_date]
    test = [i for i in ordered if i.date >= cutoff_date]
    return train, test


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))


def evaluate(
    model: Any,
    train: list[Interaction],
    test: list[Interaction],
    k: int = 10,
    catalogue: set[str] | None = None,
) -> RecoScore:
    """Score a fitted recommender against held-out interactions."""
    train_by_user: dict[str, set[str]] = defaultdict(set)
    for interaction in train:
        train_by_user[interaction.customer].add(interaction.sku)

    test_by_user: dict[str, set[str]] = defaultdict(set)
    for interaction in test:
        test_by_user[interaction.customer].add(interaction.sku)

    items = catalogue or {i.sku for i in train} | {i.sku for i in test}
    # Self-information for novelty: rarer item, higher novelty.
    counts: dict[str, int] = defaultdict(int)
    for interaction in train:
        counts[interaction.sku] += 1
    total = sum(counts.values()) or 1

    precisions: list[float] = []
    recalls: list[float] = []
    average_precisions: list[float] = []
    ndcgs: list[float] = []
    hits = 0
    evaluated = 0
    skipped = 0
    recommended_items: set[str] = set()
    novelties: list[float] = []

    for customer, relevant in test_by_user.items():
        # Only score against items the customer had not already bought: a
        # recommender is not being tested on re-purchases it was told to
        # exclude.
        history = train_by_user.get(customer, set())
        unseen_relevant = relevant - history
        if not unseen_relevant:
            skipped += 1
            continue

        recommendations = model.recommend(customer, k=k, history=history)
        recommended = [r.sku for r in recommendations]
        recommended_items.update(recommended)
        evaluated += 1

        flags = [1 if sku in unseen_relevant else 0 for sku in recommended]
        hit_count = sum(flags)

        precisions.append(hit_count / k if k else 0.0)
        recalls.append(hit_count / len(unseen_relevant))
        hits += 1 if hit_count else 0

        # Average precision over the ranked list.
        running = 0
        precision_sum = 0.0
        for rank, flag in enumerate(flags, start=1):
            if flag:
                running += 1
                precision_sum += running / rank
        average_precisions.append(
            precision_sum / min(len(unseen_relevant), k) if unseen_relevant else 0.0
        )

        ideal = _dcg([1] * min(len(unseen_relevant), k))
        ndcgs.append(_dcg(flags) / ideal if ideal else 0.0)

        for sku in recommended:
            probability = counts.get(sku, 0) / total
            novelties.append(-math.log2(probability) if probability > 0 else 0.0)

    return RecoScore(
        users_evaluated=evaluated,
        skipped=skipped,
        k=k,
        precision=statistics.fmean(precisions) if precisions else 0.0,
        recall=statistics.fmean(recalls) if recalls else 0.0,
        map_score=statistics.fmean(average_precisions) if average_precisions else 0.0,
        ndcg=statistics.fmean(ndcgs) if ndcgs else 0.0,
        hit_rate=hits / evaluated if evaluated else 0.0,
        coverage=len(recommended_items) / len(items) if items else 0.0,
        novelty=statistics.fmean(novelties) if novelties else 0.0,
    )


@dataclass
class RecoComparison:
    results: dict[str, RecoScore] = field(default_factory=dict)

    def render(self) -> str:
        if not self.results:
            return "no results"
        k = next(iter(self.results.values())).k
        header = (
            f"{'MODEL':<22}{'PREC@' + str(k):>9}{'RECALL':>9}{'MAP':>8}"
            f"{'NDCG':>8}{'HIT':>8}{'COVERAGE':>10}{'NOVELTY':>9}"
        )
        lines = [header, "-" * len(header)]
        for name, score in sorted(
            self.results.items(), key=lambda kv: -kv[1].ndcg
        ):
            lines.append(
                f"{name:<22}{score.precision:>9.4f}{score.recall:>9.4f}"
                f"{score.map_score:>8.4f}{score.ndcg:>8.4f}{score.hit_rate:>8.4f}"
                f"{score.coverage:>10.4f}{score.novelty:>9.3f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {name: score.to_dict() for name, score in self.results.items()}
