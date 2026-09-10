"""Item-item collaborative filtering for implicit feedback.

Implicit, not explicit: retail has purchases, not star ratings. That changes
the problem in a way that trips up naive implementations — there are no
negative signals. A customer who never bought an item might dislike it or might
never have seen it, and treating "not bought" as "disliked" is why so many
implicit recommenders confidently recommend nothing useful.

Item-item rather than user-item because in retail the item catalogue is far
more stable than the customer base: item neighbourhoods computed today are
still valid next week, whereas user vectors go stale as soon as someone shops.
It also gives a directly explainable recommendation — "because you bought X" —
which matters when a merchandiser asks why something surfaced.

The similarity is cosine over the item-user matrix with two corrections that
decide whether this works at all:

* **Popularity damping.** Raw cosine makes the best-selling item everyone's
  nearest neighbour, so every recommendation list collapses to the same top
  sellers. The damping exponent trades accuracy for catalogue coverage, and
  both are measured.
* **Inverse-user-frequency weighting.** A customer who bought 200 items says
  much less about any *pair* of them than a customer who bought three.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..data import Interaction


@dataclass
class RecommenderConfig:
    #: Neighbours kept per item. Beyond ~100 the tail is noise.
    max_neighbours: int = 60
    #: Similarity floor; pairs below this are dropped.
    min_similarity: float = 0.01
    #: Popularity damping exponent, in [0, 1]. 0 disables damping (raw cosine,
    #: which collapses to a bestseller list); 1 fully normalises popularity out.
    popularity_damping: float = 0.5
    #: Down-weight customers with very large baskets.
    inverse_user_frequency: bool = True
    #: Items with fewer interactions than this are excluded: a similarity based
    #: on two co-purchases is noise wearing a number.
    min_item_interactions: int = 3


@dataclass
class Recommendation:
    sku: str
    score: float
    #: The items in the customer's history that produced this suggestion.
    because_of: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku": self.sku,
            "score": round(self.score, 5),
            "because_of": self.because_of[:3],
        }


class ItemItemRecommender:
    """Item-item cosine similarity over implicit interactions."""

    name = "item_item_cf"

    def __init__(self, config: RecommenderConfig | None = None) -> None:
        self.config = config or RecommenderConfig()
        self.neighbours: dict[str, list[tuple[str, float]]] = {}
        self.item_popularity: dict[str, int] = {}
        self.user_items: dict[str, set[str]] = {}
        self._items: list[str] = []

    def fit(self, interactions: list[Interaction]) -> ItemItemRecommender:
        config = self.config

        users_by_item: dict[str, set[str]] = defaultdict(set)
        items_by_user: dict[str, set[str]] = defaultdict(set)
        for interaction in interactions:
            users_by_item[interaction.sku].add(interaction.customer)
            items_by_user[interaction.customer].add(interaction.sku)

        # Drop items too rare to compute a meaningful similarity for.
        users_by_item = {
            sku: users
            for sku, users in users_by_item.items()
            if len(users) >= config.min_item_interactions
        }
        self.item_popularity = {sku: len(users) for sku, users in users_by_item.items()}
        self.user_items = {user: set(items) for user, items in items_by_user.items()}
        self._items = sorted(users_by_item)

        # Inverse user frequency: a 200-item basket is weak evidence about any
        # particular pair within it.
        user_weight: dict[str, float] = {}
        for user, items in items_by_user.items():
            user_weight[user] = (
                1.0 / math.log(1 + len(items)) if config.inverse_user_frequency else 1.0
            )

        norms: dict[str, float] = {}
        for sku, users in users_by_item.items():
            norms[sku] = math.sqrt(sum(user_weight[u] ** 2 for u in users)) or 1.0

        # Accumulate co-occurrence through the user's basket rather than
        # comparing every item pair: retail catalogues are large and sparse, so
        # the pairwise loop would be quadratic in the catalogue for no reason.
        numerator: dict[tuple[str, str], float] = defaultdict(float)
        for user, items in items_by_user.items():
            present = sorted(item for item in items if item in users_by_item)
            weight = user_weight[user] ** 2
            for i, left in enumerate(present):
                for right in present[i + 1 :]:
                    numerator[(left, right)] += weight

        similarities: dict[str, list[tuple[str, float]]] = defaultdict(list)
        total_users = len(items_by_user) or 1
        for (left, right), shared in numerator.items():
            similarity = shared / (norms[left] * norms[right])
            if config.popularity_damping:
                # Divide out a power of the popularity product, so a bestseller
                # is not automatically everyone's nearest neighbour.
                popularity = (
                    self.item_popularity[left] * self.item_popularity[right]
                ) / (total_users**2)
                similarity /= max(popularity, 1e-9) ** config.popularity_damping
            if similarity < config.min_similarity:
                continue
            similarities[left].append((right, similarity))
            similarities[right].append((left, similarity))

        self.neighbours = {
            sku: sorted(pairs, key=lambda pair: -pair[1])[: config.max_neighbours]
            for sku, pairs in similarities.items()
        }
        return self

    def similar_items(self, sku: str, k: int = 10) -> list[tuple[str, float]]:
        """The nearest neighbours of one item. Useful on a product page."""
        return self.neighbours.get(sku, [])[:k]

    def recommend(
        self,
        customer: str,
        k: int = 10,
        history: set[str] | None = None,
        exclude_seen: bool = True,
    ) -> list[Recommendation]:
        """Recommend for a customer, aggregating over their purchase history."""
        seen = history if history is not None else self.user_items.get(customer, set())
        if not seen:
            return self._popular(k)

        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, list[str]] = defaultdict(list)
        for item in seen:
            for neighbour, similarity in self.neighbours.get(item, []):
                if exclude_seen and neighbour in seen:
                    continue
                scores[neighbour] += similarity
                reasons[neighbour].append(item)

        if not scores:
            return self._popular(k, exclude=seen if exclude_seen else set())

        ranked = sorted(scores.items(), key=lambda pair: -pair[1])[:k]
        return [
            Recommendation(
                sku=sku,
                score=score,
                because_of=sorted(
                    reasons[sku],
                    key=lambda item: -max(
                        (s for n, s in self.neighbours.get(item, []) if n == sku),
                        default=0.0,
                    ),
                ),
            )
            for sku, score in ranked
        ]

    def _popular(self, k: int, exclude: set[str] | None = None) -> list[Recommendation]:
        """Cold-start fallback.

        A new customer has no history, so there is nothing to collaborate on.
        Returning popular items is not a clever answer, but it is the honest
        one, and it is flagged in ``because_of`` so the caller can tell the
        difference between a personalised list and a fallback.
        """
        blocked = exclude or set()
        ranked = sorted(
            ((sku, count) for sku, count in self.item_popularity.items() if sku not in blocked),
            key=lambda pair: -pair[1],
        )[:k]
        total = sum(self.item_popularity.values()) or 1
        return [
            Recommendation(sku=sku, score=count / total, because_of=["<popularity fallback>"])
            for sku, count in ranked
        ]

    def describe(self) -> dict[str, Any]:
        neighbour_counts = [len(v) for v in self.neighbours.values()]
        return {
            "name": self.name,
            "items": len(self._items),
            "items_with_neighbours": len(self.neighbours),
            "mean_neighbours": round(
                sum(neighbour_counts) / len(neighbour_counts), 1
            )
            if neighbour_counts
            else 0.0,
            "customers": len(self.user_items),
            "popularity_damping": self.config.popularity_damping,
        }


class PopularityRecommender:
    """Recommend the best sellers to everybody.

    The baseline that must be beaten. It is embarrassingly hard to beat on
    recall in retail, because most purchases really are of popular items, and
    a collaborative filter that cannot beat it is not earning its complexity.
    """

    name = "popularity"

    def __init__(self) -> None:
        self.item_popularity: dict[str, int] = {}
        self.user_items: dict[str, set[str]] = {}

    def fit(self, interactions: list[Interaction]) -> PopularityRecommender:
        counts: dict[str, int] = defaultdict(int)
        users: dict[str, set[str]] = defaultdict(set)
        for interaction in interactions:
            counts[interaction.sku] += 1
            users[interaction.customer].add(interaction.sku)
        self.item_popularity = dict(counts)
        self.user_items = {u: set(items) for u, items in users.items()}
        return self

    def recommend(
        self,
        customer: str,
        k: int = 10,
        history: set[str] | None = None,
        exclude_seen: bool = True,
    ) -> list[Recommendation]:
        seen = history if history is not None else self.user_items.get(customer, set())
        blocked = seen if exclude_seen else set()
        ranked = sorted(
            ((sku, c) for sku, c in self.item_popularity.items() if sku not in blocked),
            key=lambda pair: -pair[1],
        )[:k]
        total = sum(self.item_popularity.values()) or 1
        return [Recommendation(sku=sku, score=c / total) for sku, c in ranked]

    def similar_items(self, sku: str, k: int = 10) -> list[tuple[str, float]]:
        return []

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "items": len(self.item_popularity)}
