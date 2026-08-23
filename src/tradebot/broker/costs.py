"""Fill economics: commission and slippage.

Lives in the broker layer because that is where costs are physically incurred — a real
broker charges commission and gives you the fill you actually got. Modelling them anywhere
else invites a backtest that charges different costs than the paper run.

Slippage is applied **against** the order in every case: buys fill higher, sells fill
lower. There is no configuration that makes it favourable, because a cost model that can
help you is a cost model that will eventually be tuned until it does.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import CostConfig
from ..core.types import Side
from ..instruments.registry import InstrumentSpec


@dataclass(frozen=True, slots=True)
class CostModel:
    instrument: InstrumentSpec
    commission_round_trip_usd: float
    slippage_ticks_per_side: float
    stress_multiplier: float = 1.0

    @classmethod
    def from_config(
        cls, config: CostConfig, instrument: InstrumentSpec, *, stress: bool = False
    ) -> CostModel:
        return cls(
            instrument=instrument,
            commission_round_trip_usd=config.commission_round_trip_usd,
            slippage_ticks_per_side=config.slippage_ticks_per_side,
            stress_multiplier=config.slippage_stress_multiplier if stress else 1.0,
        )

    @property
    def slippage_points(self) -> float:
        return self.slippage_ticks_per_side * self.stress_multiplier * self.instrument.tick_size

    def commission_per_fill(self, quantity: int) -> float:
        """Half a round turn, per contract. Charged on entry and again on exit."""
        return (self.commission_round_trip_usd / 2.0) * quantity

    def apply_slippage(self, price: float, side: Side) -> float:
        """Move the price against the side taking liquidity, then snap to the tick grid."""
        slipped = price + self.slippage_points * int(side)
        return self.instrument.round_to_tick(slipped)

    def pnl_usd(self, entry: float, exit_price: float, quantity: int, side: Side) -> float:
        return (exit_price - entry) * int(side) * quantity * self.instrument.multiplier

    def round_trip_cost_usd(self, quantity: int) -> float:
        """Total expected cost of a round trip: both commissions plus both slippages.

        Useful for stating cost as a share of the move a strategy is trying to capture —
        the number that killed the 1-minute order-flow work in this repository, where a
        real +3.65-tick gross effect met a 3.2-tick fee.
        """
        commission = self.commission_per_fill(quantity) * 2
        slippage = self.slippage_points * 2 * quantity * self.instrument.multiplier
        return commission + slippage

    def describe(self) -> str:
        stress = f" (stress x{self.stress_multiplier:g})" if self.stress_multiplier != 1 else ""
        return (
            f"${self.commission_round_trip_usd:.2f} round turn per contract, "
            f"{self.slippage_ticks_per_side:g} tick(s) slippage per side{stress} "
            f"= ${self.round_trip_cost_usd(1):.2f} per contract round trip"
        )
