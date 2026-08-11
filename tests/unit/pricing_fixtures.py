"""Shared synthetic pricing fixtures for tests.

Builds a small transaction frame with per-SKU elasticities that the OLS
log-log estimator recovers cleanly, plus SKUs that exercise every coverage
status (insufficient_variation, insufficient_observations) and a noisy SKU
that lands in ``weak`` confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SKU_SPECS: dict[str, dict[str, object]] = {
    # sku: (true elasticity, intercept qty at price=1, noise (log-normal), n_weeks, price_scheme)
    # NOTE: revenue per week ~= intercept * price^(1+e), so a larger intercept is
    # needed to make a highly elastic SKU a high-revenue SKU. Estimable SKUs span
    # two years (104 weeks) so the week/month time-fixed-effects model is full rank.
    "ELASTIC_HI": {"e": -1.8, "qty": 100000, "noise": 0.02, "weeks": 104, "prices": (8.0, 9.0, 10.0, 11.0, 12.0)},
    "INELASTIC": {"e": -0.4, "qty": 5000, "noise": 0.02, "weeks": 104, "prices": (8.0, 9.0, 10.0, 11.0, 12.0)},
    "PRICE_LEVER": {"e": -1.5, "qty": 6000, "noise": 0.02, "weeks": 104, "prices": (8.0, 9.0, 10.0, 11.0, 12.0)},
    "REVIEW": {"e": -0.7, "qty": 1500, "noise": 0.02, "weeks": 104, "prices": (8.0, 9.0, 10.0, 11.0, 12.0)},
    "CONST_SKU": {"e": -1.0, "qty": 50000, "noise": 0.02, "weeks": 40, "prices": (10.0, 10.0, 10.0, 10.0, 10.0)},
    "SHORT_SKU": {"e": -1.0, "qty": 800, "noise": 0.02, "weeks": 3, "prices": (8.0, 10.0, 12.0)},
}

_CATEGORY = {
    "ELASTIC_HI": "drinks",
    "INELASTIC": "drinks",
    "PRICE_LEVER": "snacks",
    "REVIEW": "snacks",
    "CONST_SKU": "snacks",
    "SHORT_SKU": "drinks",
}


def build_pricing_df(seed: int = 42) -> pd.DataFrame:
    """Deterministic transaction frame exercising elasticity statuses.

    Statuses produced: estimated (4 SKUs, log-log elasticities recoverable),
    insufficient_variation (constant price) and insufficient_observations
    (3 weeks only).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    tid = 1
    for sku, spec in SKU_SPECS.items():
        prices = spec["prices"]
        weeks = int(spec["weeks"])
        for w in range(weeks):
            price = float(prices[w % len(prices)])
            base_qty = float(spec["qty"]) * price ** float(spec["e"])
            noise = float(np.exp(float(spec["noise"]) * float(rng.normal())))
            qty = max(base_qty * noise, 1.0)
            week_start = pd.Timestamp("2025-01-06") + pd.Timedelta(weeks=w)
            rows.append(
                {
                    "date": week_start,
                    "transaction_id": f"T{tid}",
                    "stockcode": sku,
                    "product": f"Product {sku}",
                    "customer_id": f"C{sku}_{w}",
                    "price": price,
                    "quantity": int(round(qty)),
                    "category": _CATEGORY[sku],
                }
            )
            tid += 1
    return pd.DataFrame(rows)


def build_kvi_fixture() -> pd.DataFrame:
    """Controlled KVI_SCORES-style frame with explicit elasticity status.

    Scores chosen so the median (4th of 7) lands at 0.40 and every decision
    type is reachable:
      - ELASTIC_HI / INELASTIC: high KVI -> invest / protect
      - PRICE_LEVER / REVIEW:   low KVI  -> price_lever / review
      - WEAK / CONST_SKU / SHORT_SKU: no usable evidence -> insufficient_evidence
    """
    return pd.DataFrame(
        {
            "stockcode": ["ELASTIC_HI", "INELASTIC", "WEAK", "CONST_SKU", "PRICE_LEVER", "REVIEW", "SHORT_SKU"],
            "category": ["drinks", "drinks", "snacks", "snacks", "snacks", "snacks", "drinks"],
            "kvi_score": [0.90, 0.80, 0.50, 0.40, 0.20, 0.10, 0.05],
            "abs_elasticity": [1.8, 0.4, 2.0, np.nan, 1.5, 0.7, np.nan],
            "elasticity_status": [
                "estimated",
                "estimated",
                "weak",
                "insufficient_variation",
                "estimated",
                "estimated",
                "insufficient_observations",
            ],
            "total_revenue": [632000.0, 478000.0, 2000.0, 80000.0, 76000.0, 120000.0, 2400.0],
            "basket_penetration": [0.5, 0.4, 0.3, 0.2, 0.2, 0.15, 0.1],
            "trip_incidence": [0.4, 0.3, 0.2, 0.15, 0.15, 0.1, 0.05],
        }
    )
