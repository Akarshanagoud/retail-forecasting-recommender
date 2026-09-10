"""Synthetic retail data.

Real retail data cannot be shipped in a public repository, and a random walk
teaches nothing: a forecaster that beats noise has proved nothing, and a
recommender fitted to uniform random interactions has no structure to find.

So this generates data with the structure that actually makes retail
forecasting hard, and that a model can therefore be shown to capture or miss:

* **Multiplicative weekly seasonality** - Saturday is not "Tuesday plus a
  constant", it is Tuesday times a factor, and the difference matters as soon
  as the trend is non-zero.
* **Annual seasonality and holiday spikes** - a Black Friday spike that no
  amount of weekly seasonality explains.
* **Price elasticity** - demand responds to price, so a model that ignores
  price will systematically miss promotions.
* **Promotions with a pull-forward effect** - the week after a promotion is
  depressed because customers bought early. Models that treat promotions as a
  simple positive lift get the following week wrong.
* **Intermittent demand** - slow-moving SKUs with many zero days, where MAPE
  is undefined and naive accuracy metrics mislead.
* **A regime change** - a step change partway through the series, so drift
  detection has something real to detect.

Everything is seeded, so the same call always produces the same data.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

# Fixed US-style retail holidays, as (month, day) with a demand multiplier.
HOLIDAYS: dict[tuple[int, int], tuple[str, float]] = {
    (1, 1): ("new_year", 0.55),
    (7, 4): ("independence_day", 0.75),
    (11, 29): ("black_friday", 3.4),
    (12, 24): ("christmas_eve", 1.9),
    (12, 25): ("christmas", 0.25),
    (12, 26): ("boxing_day", 1.6),
    (12, 31): ("new_year_eve", 1.3),
}

CATEGORIES = ("grocery", "apparel", "electronics", "home", "beauty")


@dataclass
class SeriesSpec:
    """The generating parameters for one SKU-store series."""

    sku: str
    store: str
    category: str
    base_demand: float
    trend_per_day: float
    weekly_profile: list[float]
    annual_amplitude: float
    price: float
    #: Own-price elasticity. Negative: higher price, lower demand.
    elasticity: float
    promo_probability: float
    noise_scale: float
    intermittent: bool = False
    #: Day index at which a level shift occurs, if any.
    regime_change_day: int | None = None
    regime_multiplier: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku": self.sku,
            "store": self.store,
            "category": self.category,
            "base_demand": round(self.base_demand, 2),
            "trend_per_day": round(self.trend_per_day, 5),
            "elasticity": round(self.elasticity, 2),
            "intermittent": self.intermittent,
            "regime_change_day": self.regime_change_day,
        }


@dataclass
class SalesRecord:
    date: date
    sku: str
    store: str
    category: str
    units: float
    price: float
    on_promotion: bool
    holiday: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "sku": self.sku,
            "store": self.store,
            "category": self.category,
            "units": self.units,
            "price": round(self.price, 2),
            "on_promotion": self.on_promotion,
            "holiday": self.holiday,
        }


@dataclass
class Interaction:
    """A customer buying an item, for the recommender."""

    customer: str
    sku: str
    date: date
    quantity: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer": self.customer,
            "sku": self.sku,
            "date": self.date.isoformat(),
            "quantity": self.quantity,
        }


@dataclass
class Dataset:
    sales: list[SalesRecord] = field(default_factory=list)
    interactions: list[Interaction] = field(default_factory=list)
    specs: list[SeriesSpec] = field(default_factory=list)

    @property
    def skus(self) -> list[str]:
        return sorted({record.sku for record in self.sales})

    @property
    def stores(self) -> list[str]:
        return sorted({record.store for record in self.sales})

    def series(self, sku: str, store: str) -> list[SalesRecord]:
        """One SKU-store series in date order."""
        return sorted(
            (r for r in self.sales if r.sku == sku and r.store == store),
            key=lambda r: r.date,
        )

    def series_keys(self) -> list[tuple[str, str]]:
        return sorted({(r.sku, r.store) for r in self.sales})

    def summary(self) -> dict[str, Any]:
        units = [r.units for r in self.sales]
        return {
            "rows": len(self.sales),
            "series": len(self.series_keys()),
            "skus": len(self.skus),
            "stores": len(self.stores),
            "date_range": [
                min(r.date for r in self.sales).isoformat(),
                max(r.date for r in self.sales).isoformat(),
            ]
            if self.sales
            else [],
            "mean_units": round(sum(units) / len(units), 2) if units else 0.0,
            "zero_days": sum(1 for u in units if u == 0),
            "promotions": sum(1 for r in self.sales if r.on_promotion),
            "customers": len({i.customer for i in self.interactions}),
            "interactions": len(self.interactions),
        }


def _weekly_profile(rng: random.Random, category: str) -> list[float]:
    """Multiplicative day-of-week factors, Monday first, averaging ~1.0."""
    shapes = {
        # Grocery is flat midweek and peaks at the weekend.
        "grocery": [0.92, 0.90, 0.95, 1.02, 1.18, 1.30, 0.98],
        # Apparel is heavily weekend-weighted.
        "apparel": [0.80, 0.78, 0.85, 0.95, 1.25, 1.55, 1.20],
        # Electronics peaks Friday/Saturday but is steadier.
        "electronics": [0.95, 0.92, 0.95, 1.00, 1.15, 1.25, 1.05],
        "home": [0.88, 0.85, 0.92, 1.00, 1.15, 1.40, 1.15],
        "beauty": [0.90, 0.90, 0.95, 1.05, 1.20, 1.30, 1.02],
    }
    base = shapes.get(category, [1.0] * 7)
    jittered = [factor * rng.uniform(0.95, 1.05) for factor in base]
    mean = sum(jittered) / 7
    return [factor / mean for factor in jittered]


def make_specs(
    n_skus: int = 12, n_stores: int = 3, seed: int = 20260101
) -> list[SeriesSpec]:
    rng = random.Random(seed)
    specs: list[SeriesSpec] = []
    for i in range(n_skus):
        category = CATEGORIES[i % len(CATEGORIES)]
        sku = f"SKU-{i:03d}"
        # Every fifth SKU is a slow mover with intermittent demand. With a
        # catalogue smaller than five, the last SKU takes the role - otherwise
        # a small test dataset silently contains no zero-demand series at all,
        # and the intermittent-demand code paths go unexercised.
        intermittent = i % 5 == 4 or (n_skus < 5 and i == n_skus - 1)
        for j in range(n_stores):
            store = f"ST-{j:02d}"
            base = rng.uniform(2.0, 8.0) if intermittent else rng.uniform(25.0, 220.0)
            specs.append(
                SeriesSpec(
                    sku=sku,
                    store=store,
                    category=category,
                    base_demand=base,
                    trend_per_day=rng.uniform(-0.04, 0.09),
                    weekly_profile=_weekly_profile(rng, category),
                    annual_amplitude=rng.uniform(0.05, 0.30),
                    price=round(rng.uniform(3.0, 90.0), 2),
                    elasticity=rng.uniform(-2.4, -0.7),
                    promo_probability=rng.uniform(0.02, 0.10),
                    noise_scale=rng.uniform(0.06, 0.18),
                    intermittent=intermittent,
                    # One series in six shifts level partway through.
                    regime_change_day=rng.randint(300, 500) if (i + j) % 6 == 0 else None,
                    regime_multiplier=rng.choice([0.65, 0.72, 1.35, 1.45]),
                )
            )
    return specs


def generate_sales(
    specs: list[SeriesSpec] | None = None,
    start: date | None = None,
    days: int = 730,
    seed: int = 7,
) -> tuple[list[SalesRecord], list[SeriesSpec]]:
    """Generate daily sales for each series."""
    rng = random.Random(seed)
    specs = specs if specs is not None else make_specs()
    start = start or date(2024, 1, 1)
    records: list[SalesRecord] = []

    for spec in specs:
        # Promotions last several days and depress the following week, so
        # carry state across the loop rather than sampling independently.
        promo_days_left = 0
        pull_forward_debt = 0.0

        for offset in range(days):
            current = start + timedelta(days=offset)
            holiday_name = None
            multiplier = 1.0

            level = spec.base_demand + spec.trend_per_day * offset
            if spec.regime_change_day is not None and offset >= spec.regime_change_day:
                level *= spec.regime_multiplier

            multiplier *= spec.weekly_profile[current.weekday()]
            # Annual seasonality peaking in late November.
            day_of_year = current.timetuple().tm_yday
            multiplier *= 1 + spec.annual_amplitude * math.sin(
                2 * math.pi * (day_of_year - 60) / 365.25
            )

            holiday = HOLIDAYS.get((current.month, current.day))
            if holiday is not None:
                holiday_name, holiday_multiplier = holiday
                multiplier *= holiday_multiplier

            price = spec.price
            if promo_days_left > 0:
                promo_days_left -= 1
                on_promotion = True
                price = round(spec.price * rng.uniform(0.65, 0.85), 2)
            else:
                on_promotion = rng.random() < spec.promo_probability
                if on_promotion:
                    promo_days_left = rng.randint(1, 4)
                    price = round(spec.price * rng.uniform(0.65, 0.85), 2)

            # Constant-elasticity demand response to the price change.
            multiplier *= (price / spec.price) ** spec.elasticity

            if on_promotion:
                # Some of the promotional lift is borrowed from next week.
                pull_forward_debt += level * multiplier * 0.12
            elif pull_forward_debt > 0:
                repayment = min(pull_forward_debt, level * multiplier * 0.25)
                multiplier *= 1 - repayment / max(level * multiplier, 1e-9)
                pull_forward_debt -= repayment

            expected = max(level * multiplier, 0.0)
            noisy = expected * rng.gauss(1.0, spec.noise_scale)

            if spec.intermittent:
                # Slow movers: many days with no sale at all.
                units = 0.0 if rng.random() < 0.45 else float(max(0, round(noisy)))
            else:
                units = float(max(0, round(noisy)))

            records.append(
                SalesRecord(
                    date=current,
                    sku=spec.sku,
                    store=spec.store,
                    category=spec.category,
                    units=units,
                    price=price,
                    on_promotion=on_promotion,
                    holiday=holiday_name,
                )
            )

    return records, specs


def generate_interactions(
    sales: list[SalesRecord],
    n_customers: int = 600,
    seed: int = 11,
) -> list[Interaction]:
    """Customer-item interactions with genuine co-purchase structure.

    Customers are assigned an affinity for one or two categories, so items in
    the same category co-occur far more often than chance. Without that a
    collaborative filter has nothing to learn and every evaluation is noise.
    """
    rng = random.Random(seed)
    skus_by_category: dict[str, list[str]] = {}
    for record in sales:
        bucket = skus_by_category.setdefault(record.category, [])
        if record.sku not in bucket:
            bucket.append(record.sku)

    categories = sorted(skus_by_category)
    dates = sorted({record.date for record in sales})
    interactions: list[Interaction] = []

    for index in range(n_customers):
        customer = f"CUST-{index:04d}"
        primary = rng.choice(categories)
        secondary = rng.choice(categories)
        # A long tail: most customers buy a little, a few buy a lot.
        basket_count = max(3, int(rng.paretovariate(1.6) * 4))

        for _ in range(min(basket_count, 60)):
            roll = rng.random()
            if roll < 0.65:
                category = primary
            elif roll < 0.88:
                category = secondary
            else:
                category = rng.choice(categories)
            sku = rng.choice(skus_by_category[category])
            interactions.append(
                Interaction(
                    customer=customer,
                    sku=sku,
                    date=rng.choice(dates),
                    quantity=rng.randint(1, 3),
                )
            )

    return interactions


def generate_dataset(
    n_skus: int = 12,
    n_stores: int = 3,
    days: int = 730,
    n_customers: int = 600,
    start: date | None = None,
    seed: int = 7,
) -> Dataset:
    """The full synthetic dataset: sales series plus customer interactions."""
    specs = make_specs(n_skus=n_skus, n_stores=n_stores, seed=seed * 3 + 1)
    sales, specs = generate_sales(specs=specs, start=start, days=days, seed=seed)
    interactions = generate_interactions(sales, n_customers=n_customers, seed=seed + 4)
    return Dataset(sales=sales, interactions=interactions, specs=specs)


def to_series(records: list[SalesRecord]) -> tuple[list[date], list[float]]:
    """Split records into aligned date and value lists."""
    ordered = sorted(records, key=lambda r: r.date)
    return [r.date for r in ordered], [r.units for r in ordered]
