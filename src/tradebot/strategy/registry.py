"""Strategy registry: name -> constructor.

Selecting a strategy by string is what lets config, the CLI and the dashboard all refer to
the same thing without importing it, and it is the seam that makes adding a seventh family
a one-line change rather than an edit to the runner.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import Strategy
from .orb_breakout import OpeningRangeBreakout
from .sr_breakout_retest import SupportResistanceBreakoutRetest
from .sr_rejection import SupportResistanceRejection
from .trend_pullback import TrendPullback
from .vwap_continuation import VwapContinuation
from .vwap_reversion import VwapReversion

_REGISTRY: dict[str, Callable[..., Strategy]] = {
    "orb_breakout": OpeningRangeBreakout.build,
    "sr_rejection": SupportResistanceRejection.build,
    "sr_breakout_retest": SupportResistanceBreakoutRetest.build,
    "vwap_reversion": VwapReversion.build,
    "vwap_continuation": VwapContinuation.build,
    "trend_pullback": TrendPullback.build,
}


class UnknownStrategy(KeyError):
    pass


def build_strategy(name: str, params: dict | None = None) -> Strategy:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise UnknownStrategy(
            f"{name!r} is not a registered strategy. Known: {known_strategies()}"
        )
    strategy = _REGISTRY[key](**(params or {}))
    if strategy.name != key:
        raise ValueError(
            f"{key!r} built a strategy calling itself {strategy.name!r}; "
            "the registry key and StrategySpec.name must agree"
        )
    return strategy


def register_strategy(name: str, factory: Callable[..., Strategy]) -> None:
    key = name.strip().lower()
    if key in _REGISTRY:
        raise ValueError(f"{key} is already registered")
    _REGISTRY[key] = factory


def known_strategies() -> list[str]:
    return sorted(_REGISTRY)
