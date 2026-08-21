"""Product Decision Profile central store.

Centralized SKU decision profile storage with caching.
Provides per-SKU profile lookups with all decision fields,
computed from existing analytics pipelines and cached to avoid
recomputation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.analytics import promo
from src.analytics.performance import (
    abc_analysis,
    compute_repeat_rate,
    compute_sku_rationalization_df,
    compute_velocity,
    product_lifecycle_stage,
    xyz_analysis,
)
from src.analytics.pricing.elasticity import estimate_loglog_elasticity
from src.analytics.switching import compute_switching_status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile field definitions
# ---------------------------------------------------------------------------

PROFILE_FIELDS = [
    "revenue",
    "growth",
    "abc",
    "xyz",
    "lifecycle",
    "elasticity",
    "promo_effectiveness",
    "switching_risk",
    "substitutability",
    "customer_reach",
    "repeat_rate",
    "assortment_action",
    "price_action",
    "promo_action",
]


# ---------------------------------------------------------------------------
# Profile Service
# ---------------------------------------------------------------------------

class ProfileService:
    """Centralized SKU decision profile storage with caching.

    Computes and caches per-SKU decision profiles from existing analytics
    pipelines. Profile lookups are O(1) after initial computation.

    The service computes profiles in two stages:
    - Base profile: fast, computed from simple groupby operations (revenue, abc,
      xyz, lifecycle, velocity, repeat_rate, customer_reach, assortment_action)
    - Extended profile: computed on first cache miss (elasticity, promo,
      switching, price_action, promo_action) and then cached

    Caching: results are cached in-memory keyed by stockcode. A cache miss
    triggers computation from source data and populates the cache. Subsequent
    lookups hit the cache.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the profile service.

        Args:
            df: Transaction DataFrame with canonical columns
                (date, transaction_id, stockcode, product, customer_id, price, quantity).
        """
        self.df = df.copy()
        self._cache: Dict[str, dict[str, Any]] = {}
        self._base_computed: bool = False
        self._extended_cache: Dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Base profile computation (fast, groupby-based, computed once)
    # ------------------------------------------------------------------

    def _compute_base_profiles(self) -> None:
        """Compute base per-SKU profiles from simple groupby operations.

        These are inexpensive and computed in a small number of passes
        through the DataFrame. Results are cached internally and reused
        for all SKU lookups.
        """
        if self._base_computed:
            return

        n_baskets = int(self.df["transaction_id"].nunique()) or 1

        # --- revenue ---
        revenue_series = self.df.groupby("stockcode").apply(
            lambda x: float((x["price"] * x["quantity"]).sum()), include_groups=False
        )
        revenue_by_sku: Dict[str, float] = {}
        for k, v in revenue_series.items():
            revenue_by_sku[str(k)] = v

        # --- ABC classification ---
        abc_df = abc_analysis(self.df).set_index("stockcode")["abc_class"]
        abc_by_sku: Dict[str, str] = {}
        for k, v in abc_df.items():
            abc_by_sku[str(k)] = str(v)

        # --- XYZ classification and demand profile ---
        xyz_df = xyz_analysis(self.df).set_index("stockcode")
        xyz_by_sku: Dict[str, str] = {}
        demand_profile_by_sku: Dict[str, str] = {}
        for k, v in xyz_df["xyz_class"].items():
            xyz_by_sku[str(k)] = str(v)
        for k, v in xyz_df["demand_profile"].items():
            demand_profile_by_sku[str(k)] = str(v)

        # --- Lifecycle stage ---
        lifecycle_df = product_lifecycle_stage(self.df).set_index("stockcode")
        stage_by_sku: Dict[str, str] = {}
        growth_by_sku: Dict[str, float] = {}
        for k, v in lifecycle_df["stage"].items():
            stage_by_sku[str(k)] = str(v)
        for k, v in lifecycle_df["growth_pct"].items():
            growth_by_sku[str(k)] = float(v)

        # --- Velocity ---
        vel_df = compute_velocity(self.df).set_index("stockcode")
        velocity_by_sku: Dict[str, float] = {}
        for k, v in vel_df["velocity"].items():
            velocity_by_sku[str(k)] = float(v)

        # --- Repeat rate ---
        repeat_df = compute_repeat_rate(self.df).set_index("stockcode")
        repeat_rate_by_sku: Dict[str, float] = {}
        for k, v in repeat_df["repeat_rate"].items():
            repeat_rate_by_sku[str(k)] = float(v)

        # --- Customer reach (penetration) ---
        n_customers_by_sku = (
            self.df.groupby("stockcode")["customer_id"].nunique().to_dict()
        )
        penetration_by_sku: Dict[str, float] = {}
        for k, v in n_customers_by_sku.items():
            penetration_by_sku[str(k)] = min(float(v) / n_baskets, 1.0)

        # --- Assortment action (ABC × XYZ × velocity × repeat) ---
        rat_df = compute_sku_rationalization_df(self.df).set_index("stockcode")
        assortment_action_by_sku: Dict[str, str] = {}
        for k, v in rat_df["action"].items():
            assortment_action_by_sku[str(k)] = str(v)

        # Store all base results
        self._base: Dict[str, dict[str, Any]] = {}

        # Collect all SKU keys seen across any base computation
        all_skus: set[str] = set()
        for d in [revenue_by_sku, abc_by_sku, xyz_by_sku, demand_profile_by_sku,
                  stage_by_sku, growth_by_sku, velocity_by_sku, repeat_rate_by_sku,
                  penetration_by_sku, assortment_action_by_sku]:
            all_skus.update(d.keys())

        for sku in all_skus:
            self._base[sku] = {
                "revenue": revenue_by_sku.get(sku, 0.0),
                "abc": abc_by_sku.get(sku, "C"),
                "xyz": xyz_by_sku.get(sku, "Z"),
                "demand_profile": demand_profile_by_sku.get(sku, "Insufficient History"),
                "lifecycle": stage_by_sku.get(sku, "mature"),
                "growth": growth_by_sku.get(sku, 0.0),
                "velocity": velocity_by_sku.get(sku, 0.0),
                "repeat_rate": repeat_rate_by_sku.get(sku, 0.0),
                "customer_reach": penetration_by_sku.get(sku, 0.0),
                "assortment_action": assortment_action_by_sku.get(sku, "review"),
            }

        self._base_computed = True
        logger.info("Computed base profiles for %d SKUs", len(self._base))

    # ------------------------------------------------------------------
    # Extended profile computation (on-cache-miss, then cached)
    # ------------------------------------------------------------------

    def _compute_extended_profile(self, stockcode: str) -> dict[str, Any]:
        """Compute extended profile fields for a single SKU.

        These require more expensive operations (pricing elasticity,
        promo detection, switching analysis). Called on cache miss;
        results are cached so subsequent lookups are fast.

        Args:
            stockcode: The SKU code.

        Returns:
            Dict of extended profile fields.
        """
        # Return from extended cache if already computed
        if stockcode in self._extended_cache:
            return self._extended_cache[stockcode]

        result: dict[str, Any] = {}

        # --- Elasticity ---
        try:
            elast_df = estimate_loglog_elasticity(self.df, min_periods=5, min_price_variation=0.05, add_time_fe=False)
            elast_rows = elast_df[elast_df["stockcode"] == stockcode]
            if len(elast_rows) > 0 and not elast_rows.empty:
                elast_series = elast_rows["elasticity"]
                if len(elast_series) > 0:
                    elast_val = float(elast_series.iloc[0])
                else:
                    elast_val = 0.0
                result["elasticity"] = elast_val
            else:
                result["elasticity"] = 0.0
        except Exception:
            result["elasticity"] = 0.0

        # --- Promo effectiveness ---
        try:
            detected_promos = promo.detect_promotions(self.df)
            if detected_promos.empty:
                result["promo_effectiveness"] = 0.0
            else:
                sku_promo_filter = detected_promos["stockcode"] == stockcode
                filtered = detected_promos[sku_promo_filter]
                if filtered.empty:
                    result["promo_effectiveness"] = 0.0
                else:
                    lift_vals = filtered["revenue_lift"].tolist()
                    lift_nums: list[float] = []
                    for lv in lift_vals:
                        try:
                            fn = float(lv)
                            if fn == fn:
                                lift_nums.append(fn)
                        except (TypeError, ValueError):
                            pass
                    if lift_nums:
                        mean_lift = sum(lift_nums) / len(lift_nums)
                        result["promo_effectiveness"] = round(mean_lift, 2)
                    else:
                        result["promo_effectiveness"] = 0.0
        except Exception:
            result["promo_effectiveness"] = 0.0

        # --- Switching risk ---
        try:
            switching_status_df = compute_switching_status(self.df)
            ss_rows = switching_status_df[switching_status_df["stockcode"] == stockcode]
            if len(ss_rows) > 0 and not ss_rows.empty:
                result["switching_risk"] = str(ss_rows.iloc[0]["switching_status"])
            else:
                result["switching_risk"] = "no_switching_observed"
        except Exception:
            result["switching_risk"] = "unavailable"

        # --- Substitutability (SDP) ---
        try:
            from src.analytics.transference import (
                compute_demand_transference_matrix,
                compute_substitutable_demand_percentage,
            )
            transference_matrix = compute_demand_transference_matrix(self.df)
            sdp_df = compute_substitutable_demand_percentage(transference_matrix, self.df)
            sdp_rows = sdp_df[sdp_df["stockcode"] == stockcode]
            if len(sdp_rows) > 0 and not sdp_rows.empty:
                result["substitutability"] = float(sdp_rows["sdp"].iloc[0])
            else:
                result["substitutability"] = 0.5
        except Exception:
            result["substitutability"] = 0.5

        # --- Price action (from KVI + elasticity) ---
        try:
            from src.analytics.pricing.kvi import compute_kvi_score

            kvi_df = compute_kvi_score(self.df, method="heuristic")
            kvi_mask = kvi_df["stockcode"] == stockcode
            kvi_score = 0.5
            if kvi_mask.any():
                kvi_row = kvi_df.loc[kvi_mask]
                kvi_score = float(kvi_row.iloc[0, 0])

            elasticity = result.get("elasticity", 0.0)

            # Decision matrix: high KVI + elastic = invest; high KVI + inelastic = protect;
            # low KVI + elastic = price_lever; else review
            if kvi_score >= 0.6 and elasticity < -1.0:
                result["price_action"] = "invest"
            elif kvi_score >= 0.6 and elasticity >= -1.0:
                result["price_action"] = "protect"
            elif kvi_score < 0.6 and elasticity < -1.0:
                result["price_action"] = "price_lever"
            else:
                result["price_action"] = "review"
        except Exception:
            result["price_action"] = "review"

        # --- Promo action ---
        try:
            detected_promos = promo.detect_promotions(self.df)
            if len(detected_promos) > 0:
                sku_promos = detected_promos[detected_promos["stockcode"] == stockcode]
                if len(sku_promos) > 0:
                    # Simple fallback: has promos -> price_lever, no promos -> review
                    result["promo_action"] = "price_lever"
                else:
                    result["promo_action"] = "review"
            else:
                result["promo_action"] = "review"
        except Exception:
            result["promo_action"] = "review"

        # Cache the result
        self._extended_cache[stockcode] = result
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_profile(self, stockcode: str) -> dict[str, Any]:
        """Get the full decision profile for a given SKU.

        Uses cache if available. On cache miss, computes base profile
        from pre-computed data, then computes extended fields and caches
        the complete profile.

        Args:
            stockcode: The SKU code (e.g., "85050").

        Returns:
            Dict mapping profile field names to values. Always includes
            at minimum the base fields. On cache miss, extended fields
            are computed from source data and may be partial if some
            computations failed.
        """
        # Check full cache first (base + extended)
        if stockcode in self._cache:
            return self._cache[stockcode]

        # Ensure base profiles are computed
        self._compute_base_profiles()

        # Build base profile for this SKU
        base = self._base.get(stockcode, {})
        profile: dict[str, Any] = dict(base)

        # Compute extended fields on cache miss and cache result
        extended = self._compute_extended_profile(stockcode)
        profile.update(extended)

        # Cache the complete profile (base + extended)
        self._cache[stockcode] = profile

        logger.info("Profile lookup for %s: %d fields (cache miss, computed from source)", stockcode, len(profile))
        return profile

    def get_profile_cached(self, stockcode: str) -> dict[str, Any]:
        """Get profile, forcing cache usage.

        If the profile is in cache, returns it. If not, returns whatever
        base fields are available (may be incomplete). Use
        ``get_profile()`` for full computation on cache miss.

        Args:
            stockcode: The SKU code.

        Returns:
            Dict of profile fields from cache, or partial base fields.
        """
        if stockcode in self._cache:
            return self._cache[stockcode]

        # Ensure base profiles are computed
        self._compute_base_profiles()

        base = self._base.get(stockcode, {})
        self._cache[stockcode] = dict(base)
        return self._cache[stockcode]

    def refresh(self) -> None:
        """Clear all caches and recompute from source.

        Useful when the underlying data has changed and stale profiles
        need to be refreshed.
        """
        self._cache.clear()
        self._extended_cache.clear()
        self._base_computed = False
        if hasattr(self, "_base"):
            self._base.clear()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_profile_service_instance: Optional[ProfileService] = None


def init_profile_service(df: pd.DataFrame) -> ProfileService:
    """Initialize the global profile service instance.

    Should be called once after data loading (e.g., after
    :func:`src.analytics.data.load_transactions`).

    Args:
        df: Transaction DataFrame.

    Returns:
        The initialized ProfileService instance.
    """
    global _profile_service_instance
    _profile_service_instance = ProfileService(df)
    return _profile_service_instance


def get_profile_service() -> Optional[ProfileService]:
    """Get the global profile service instance.

    Returns:
        The ProfileService instance, or None if not initialized.
    """
    return _profile_service_instance


def get_profile(stockcode: str) -> dict[str, Any]:
    """Convenience function to get a profile via the global instance.

    Args:
        stockcode: The SKU code.

    Returns:
        Profile dict.

    Raises:
        ValueError: If the profile service is not initialized.
    """
    svc = _profile_service_instance
    if svc is None:
        raise ValueError("Profile service not initialized. Call init_profile_service() first.")
    return svc.get_profile(stockcode)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "ProfileService",
    "init_profile_service",
    "get_profile_service",
    "get_profile",
    "PROFILE_FIELDS",
]
