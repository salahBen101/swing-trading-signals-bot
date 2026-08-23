"""Position sizing.

Volatility-targeted: the dollar risk per trade is held roughly constant, so a wider
ATR-derived stop mechanically buys fewer contracts. That is the whole idea — the contract
count is an output of the stop distance, not an input chosen separately.

Two guards that matter more than the formula:

* **A floor, not a rounding-down.** If the budget cannot pay for `min_contracts`, the trade
  is *skipped*. Taking it anyway at one contract would silently exceed the per-trade risk
  cap, which is precisely the limit the sizer exists to enforce.
* **A cushion throttle.** As the trailing-drawdown allowance is consumed, size scales down
  in proportion. Walking into the floor at full size is how a prop evaluation ends on a
  single trade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import PerTradeRisk
from ..core.types import RejectReason
from ..instruments.registry import InstrumentSpec


@dataclass(frozen=True, slots=True)
class SizingResult:
    contracts: int
    risk_usd: float
    risk_per_contract_usd: float
    budget_usd: float
    throttle: float
    reason: RejectReason | None = None
    detail: str = ""

    @property
    def approved(self) -> bool:
        return self.contracts > 0


class VolatilityTargetSizer:
    def __init__(self, config: PerTradeRisk, instrument: InstrumentSpec) -> None:
        self.config = config
        self.instrument = instrument

    def size(
        self,
        *,
        equity: float,
        stop_distance_points: float,
        cushion_fraction: float = 1.0,
        cushion_threshold: float = 0.4,
    ) -> SizingResult:
        cfg = self.config

        if stop_distance_points <= 0:
            return SizingResult(
                0, 0.0, 0.0, 0.0, 1.0,
                RejectReason.INVALID_ORDER,
                "stop distance is zero; risk per contract would be undefined",
            )

        budget = min(equity * (cfg.risk_pct_of_equity / 100.0), cfg.max_risk_per_trade_usd)

        throttle = 1.0
        if cushion_threshold > 0 and cushion_fraction < cushion_threshold:
            throttle = max(0.0, cushion_fraction / cushion_threshold)
            budget *= throttle

        risk_per_contract = stop_distance_points * self.instrument.multiplier
        if risk_per_contract <= 0:
            return SizingResult(
                0, 0.0, 0.0, budget, throttle,
                RejectReason.INVALID_ORDER, "risk per contract is not positive",
            )

        raw = math.floor(budget / risk_per_contract)
        contracts = min(raw, cfg.max_contracts)

        if contracts < cfg.min_contracts:
            return SizingResult(
                0, 0.0, risk_per_contract, budget, throttle,
                RejectReason.SIZE_BELOW_MINIMUM,
                f"budget ${budget:,.2f} buys {raw} contracts at ${risk_per_contract:,.2f} "
                f"risk each; minimum is {cfg.min_contracts}. Skipping rather than "
                f"exceeding the per-trade risk cap.",
            )

        return SizingResult(
            contracts=contracts,
            risk_usd=contracts * risk_per_contract,
            risk_per_contract_usd=risk_per_contract,
            budget_usd=budget,
            throttle=throttle,
        )
