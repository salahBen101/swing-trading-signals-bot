"""Contract specifications, in one place.

The multiplier is the highest-leverage constant in the project: get it wrong and every
P&L, every position size and every risk limit is off by a factor, while the output still
looks entirely plausible. So it lives here, once, and `tests/tradebot/test_instruments.py`
pins the numbers against the published CME specs.

Adding an instrument means adding a row. Nothing else in the system knows what MNQ is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    symbol: str
    name: str
    exchange: str
    multiplier: float  # USD per 1.0 index point, per contract
    tick_size: float  # minimum price increment, in index points
    currency: str = "USD"
    timezone: str = "US/Eastern"
    rth_start: time = time(9, 30)
    rth_end: time = time(16, 0)
    # Round-turn commission per contract, all-in (broker + exchange + NFA). A starting
    # figure for retail micro futures; override per broker in config rather than editing.
    default_commission_round_trip_usd: float = 1.24
    contract_months: str = "HMUZ"

    @property
    def tick_value(self) -> float:
        """USD per tick, per contract."""
        return self.tick_size * self.multiplier

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def round_to_tick(self, price: float) -> float:
        """Snap a price to the contract's tick grid.

        Every price that will be compared against a market level or used as a fill goes
        through here. Without it, an ATR-derived stop lands at 18234.187321 — a price that
        cannot exist — and comparisons against real prices become subtly path-dependent on
        floating-point noise.
        """
        if self.tick_size <= 0:
            return price
        ticks = round(price / self.tick_size)
        # Re-round the product: 73 * 0.25 is exact, but many tick sizes are not, and the
        # multiplication reintroduces the representation error we just removed.
        return round(ticks * self.tick_size, 10)

    def round_away(self, price: float, *, direction: int) -> float:
        """Snap to the tick grid in a fixed direction.

        Used for protective stops, where rounding must never move the level *closer* to
        the entry: a stop nudged 0.24 points nearer costs real money and does so silently.
        `direction` is +1 to round up, -1 to round down.
        """
        if self.tick_size <= 0:
            return price
        ticks = math.ceil(price / self.tick_size) if direction > 0 else math.floor(price / self.tick_size)
        return round(ticks * self.tick_size, 10)

    def points_to_usd(self, points: float, quantity: int = 1) -> float:
        return points * self.multiplier * quantity

    def usd_to_points(self, usd: float, quantity: int = 1) -> float:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        return usd / (self.multiplier * quantity)

    def ticks(self, points: float) -> float:
        return points / self.tick_size


# CME Globex equity index futures. Multipliers and tick sizes are the published contract
# specifications; do not "fix" these to make a backtest look better.
_SPECS: dict[str, InstrumentSpec] = {
    "MNQ": InstrumentSpec(
        symbol="MNQ",
        name="Micro E-mini Nasdaq-100",
        exchange="CME",
        multiplier=2.0,
        tick_size=0.25,
        default_commission_round_trip_usd=1.24,
    ),
    "NQ": InstrumentSpec(
        symbol="NQ",
        name="E-mini Nasdaq-100",
        exchange="CME",
        multiplier=20.0,
        tick_size=0.25,
        default_commission_round_trip_usd=4.28,
    ),
    "MES": InstrumentSpec(
        symbol="MES",
        name="Micro E-mini S&P 500",
        exchange="CME",
        multiplier=5.0,
        tick_size=0.25,
        default_commission_round_trip_usd=1.24,
    ),
    "ES": InstrumentSpec(
        symbol="ES",
        name="E-mini S&P 500",
        exchange="CME",
        multiplier=50.0,
        tick_size=0.25,
        default_commission_round_trip_usd=4.28,
    ),
}


class UnknownInstrument(KeyError):
    pass


def get_instrument(symbol: str) -> InstrumentSpec:
    key = symbol.upper().strip()
    if key not in _SPECS:
        raise UnknownInstrument(
            f"{symbol!r} is not in the instrument registry. Known: {sorted(_SPECS)}"
        )
    return _SPECS[key]


def register_instrument(spec: InstrumentSpec) -> None:
    """Extension point for instruments not shipped here. Refuses to shadow a known one."""
    key = spec.symbol.upper().strip()
    if key in _SPECS:
        raise ValueError(f"{key} is already registered; edit the registry rather than shadowing it")
    _SPECS[key] = spec


def known_instruments() -> tuple[str, ...]:
    return tuple(sorted(_SPECS))
