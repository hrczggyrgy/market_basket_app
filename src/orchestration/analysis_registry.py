"""Analysis registry managing all 15+ analyses with tiers and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# AnalysisSpec dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnalysisSpec:
    """Specification for a single analysis operation."""

    key: str
    tier: str  # A=instant, B=lazy cached, C=explicit
    dependencies: list[str] = field(default_factory=list)
    cacheable: bool = True
    max_memory_mb: int = 512
    expected_seconds: float = 30.0
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# AnalysisRegistry class managing all analyses
# ---------------------------------------------------------------------------

class AnalysisRegistry:
    """Manages all 15+ analysis specifications with tiers and metadata.

    Provides registry lookups, tier grouping, and analysis discovery.
    """

    def __init__(self) -> None:
        self._specs: dict[str, AnalysisSpec] = {}

    def register(self, spec: AnalysisSpec) -> None:
        """Register an AnalysisSpec in the registry."""
        self._specs[spec.key] = spec

    def get(self, key: str) -> AnalysisSpec:
        """Get an AnalysisSpec by key."""
        if key not in self._specs:
            raise KeyError(f"Analysis '{key}' not found in registry")
        return self._specs[key]

    def all_specs(self) -> dict[str, AnalysisSpec]:
        """Get all registered analysis specs."""
        return dict(self._specs)

    def get_tiers(self) -> dict[str, list[str]]:
        """Group analysis keys by tier."""
        tiers: dict[str, list[str]] = {"A": [], "B": [], "C": []}
        for spec in self._specs.values():
            if spec.tier in tiers:
                tiers[spec.tier].append(spec.key)
        return tiers

    def get_by_tier(self, tier: str) -> list[AnalysisSpec]:
        """Get all specs for a given tier."""
        return [spec for spec in self._specs.values() if spec.tier == tier]


# Global registry instance
_global_registry: AnalysisRegistry | None = None


def get_global_registry() -> AnalysisRegistry:
    """Get the global analysis registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AnalysisRegistry()
        # Register all analyses automatically
        # The module-level register function will populate the global state
    return _global_registry


# Module-level storage (kept for backward compatibility)
ANALYSIS_SPECS: dict[str, AnalysisSpec] = {}


def register(spec: AnalysisSpec) -> None:
    """Register an AnalysisSpec in the global registry."""
    ANALYSIS_SPECS[spec.key] = spec


def get(key: str) -> AnalysisSpec:
    """Get an AnalysisSpec by key."""
    if key not in ANALYSIS_SPECS:
        raise KeyError(f"Analysis '{key}' not found in registry")
    return ANALYSIS_SPECS[key]


def all_specs() -> dict[str, AnalysisSpec]:
    """Get all registered analysis specs."""
    return dict(ANALYSIS_SPECS)


def get_tiers() -> dict[str, list[str]]:
    """Group analysis keys by tier."""
    tiers: dict[str, list[str]] = {"A": [], "B": [], "C": []}
    for spec in ANALYSIS_SPECS.values():
        if spec.tier in tiers:
            tiers[spec.tier].append(spec.key)
    return tiers


# ---------------------------------------------------------------------------
# Tier A: Instant analyses (recompute every time, no caching overhead)
# ---------------------------------------------------------------------------

register(AnalysisSpec(
    key="overview",
    tier="A",
    dependencies=[],
    cacheable=False,
    max_memory_mb=256,
    expected_seconds=5.0,
    version="1.0.0",
))

register(AnalysisSpec(
    key="performance",
    tier="A",
    dependencies=[],
    cacheable=False,
    max_memory_mb=256,
    expected_seconds=8.0,
    version="1.0.0",
))

# ---------------------------------------------------------------------------
# Tier B: Lazy cached analyses (re-run when params/data change, cached between runs)
# ---------------------------------------------------------------------------

register(AnalysisSpec(
    key="basket",
    tier="B",
    dependencies=["overview"],
    cacheable=True,
    max_memory_mb=512,
    expected_seconds=15.0,
    version="1.0.0",
))

register(AnalysisSpec(
    key="pricing",
    tier="B",
    dependencies=["overview"],
    cacheable=True,
    max_memory_mb=512,
    expected_seconds=20.0,
    version="1.0.0",
))

register(AnalysisSpec(
    key="switching",
    tier="B",
    dependencies=["overview"],
    cacheable=True,
    max_memory_mb=512,
    expected_seconds=25.0,
    version="1.0.0",
))

register(AnalysisSpec(
    key="promotion",
    tier="B",
    dependencies=["overview"],
    cacheable=True,
    max_memory_mb=512,
    expected_seconds=18.0,
    version="1.0.0",
))

register(AnalysisSpec(
    key="segmentation",
    tier="B",
    dependencies=["overview"],
    cacheable=True,
    max_memory_mb=512,
    expected_seconds=30.0,
    version="1.0.0",
))

register(AnalysisSpec(
    key="cross_sell",
    tier="B",
    dependencies=["basket", "segmentation"],
    cacheable=True,
    max_memory_mb=512,
    expected_seconds=25.0,
    version="1.0.0",
))

register(AnalysisSpec(
    key="cohorts",
    tier="B",
    dependencies=["overview"],
    cacheable=True,
    max_memory_mb=512,
    expected_seconds=20.0,
    version="1.0.0",
))

# ---------------------------------------------------------------------------
# Tier C: Explicit analyses (manual trigger, full recomputation, heavy resources)
# ---------------------------------------------------------------------------

register(AnalysisSpec(
    key="cdt",
    tier="C",
    dependencies=["overview", "segmentation"],
    cacheable=False,
    max_memory_mb=2048,
    expected_seconds=60.0,
    version="1.0.0",
))

register(AnalysisSpec(
    key="clv",
    tier="C",
    dependencies=["cohorts"],
    cacheable=False,
    max_memory_mb=1024,
    expected_seconds=45.0,
    version="1.0.0",
))

register(AnalysisSpec(
    key="assortment",
    tier="C",
    dependencies=["basket", "pricing"],
    cacheable=False,
    max_memory_mb=2048,
    expected_seconds=60.0,
    version="1.0.0",
))

register(AnalysisSpec(
    key="network",
    tier="C",
    dependencies=["switching", "cross_sell"],
    cacheable=False,
    max_memory_mb=1024,
    expected_seconds=40.0,
    version="1.0.0",
))

register(AnalysisSpec(
    key="markov",
    tier="C",
    dependencies=["switching", "cohorts"],
    cacheable=False,
    max_memory_mb=1024,
    expected_seconds=50.0,
    version="1.0.0",
))

register(AnalysisSpec(
    key="rules",
    tier="C",
    dependencies=["cross_sell"],
    cacheable=False,
    max_memory_mb=512,
    expected_seconds=35.0,
    version="1.0.0",
))
