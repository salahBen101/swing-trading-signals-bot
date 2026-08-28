"""The single place risk decisions are made for the RSI options trial.

Every limit lives here. Nothing else in the pipeline is permitted to decide that a trade is
acceptable, and nothing else may raise a limit to accommodate a trade. When one contract costs
more than the per-trade cap, the answer is REJECTED_RISK_LIMIT and a recorded opportunity - not
a larger cap and not a smaller notional achieved by pretending fractional contracts exist.

Premium at risk for a long option is the entire debit, because that is exactly what can be lost:

    premium_at_risk = price * contract_multiplier * quantity + transaction_costs

If the risk engine cannot evaluate a trade - unknown multiplier, unknown cost, missing account
state - the answer is NO TRADE. An unavailable risk engine is not permission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from .experiment import RiskLimits
from .states import RejectReason, Rejection


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """What the risk engine needs to know about the account right now."""

    cash: float
    equity: float
    open_positions: tuple[dict, ...] = ()      # each: ticker, sector, premium_at_risk
    trades_this_session: int = 0
    realized_pnl_today: float = 0.0
    session_date: date | None = None

    @property
    def open_count(self) -> int:
        return len(self.open_positions)

    @property
    def aggregate_premium_at_risk(self) -> float:
        return sum(float(p.get("premium_at_risk", 0.0)) for p in self.open_positions)

    def count_ticker(self, ticker: str) -> int:
        return sum(1 for p in self.open_positions if p.get("ticker") == ticker)

    def count_sector(self, sector: str) -> int:
        return sum(1 for p in self.open_positions if p.get("sector") == sector)


@dataclass(frozen=True, slots=True)
class ProposedTrade:
    ticker: str
    sector: str
    option_symbol: str
    price_per_contract: float       # the price we expect to pay, per share of the contract
    contract_multiplier: int
    quantity: int
    transaction_costs: float = 0.0

    @property
    def premium_at_risk(self) -> float:
        return (self.price_per_contract * self.contract_multiplier * self.quantity
                + self.transaction_costs)


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    rejection: Rejection | None = None
    premium_at_risk: float = 0.0
    checks: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.approved


class RiskEngine:
    """Centralized, fail-closed. Every check must pass; the first failure stops evaluation."""

    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    def max_affordable_quantity(self, price_per_contract: float, multiplier: int,
                                costs_per_contract: float = 0.0) -> int:
        """Largest whole quantity that fits the per-trade cap. Often zero, and that is a valid
        answer - it means this contract cannot be traded within policy at any size."""
        per = price_per_contract * multiplier + costs_per_contract
        if per <= 0:
            return 0
        return int(self.limits.max_premium_risk_per_trade // per)

    def evaluate(self, trade: ProposedTrade, portfolio: PortfolioSnapshot,
                 now: datetime | None = None) -> RiskDecision:
        now = now or datetime.now()
        ts = now.isoformat()
        L = self.limits
        par = trade.premium_at_risk
        checks: dict = {"premium_at_risk": par}

        def reject(reason: RejectReason, detail: str, **ctx) -> RiskDecision:
            return RiskDecision(
                approved=False, premium_at_risk=par, checks=checks,
                rejection=Rejection(reason, detail, ticker=trade.ticker,
                                    occurred_at=ts, context={**checks, **ctx}))

        if trade.contract_multiplier is None or trade.contract_multiplier <= 0:
            return reject(RejectReason.CONTRACT_MULTIPLIER_UNKNOWN,
                          "contract multiplier unknown; risk cannot be computed")
        if trade.quantity < 1:
            return reject(RejectReason.RISK_PREMIUM_EXCEEDED,
                          "quantity is zero - one contract already exceeds the per-trade cap")

        # 1. per-trade premium cap
        checks["max_premium_risk_per_trade"] = L.max_premium_risk_per_trade
        if par > L.max_premium_risk_per_trade:
            return reject(
                RejectReason.RISK_PREMIUM_EXCEEDED,
                f"premium at risk ${par:,.2f} exceeds the ${L.max_premium_risk_per_trade:,.2f} "
                f"per-trade limit. One contract of {trade.option_symbol} costs "
                f"${trade.price_per_contract * trade.contract_multiplier:,.2f}.")

        # 2. concurrent positions
        checks["open_positions"] = portfolio.open_count
        checks["max_open_positions"] = L.max_open_positions
        if portfolio.open_count >= L.max_open_positions:
            return reject(RejectReason.RISK_MAX_OPEN_POSITIONS,
                          f"{portfolio.open_count} position(s) already open, limit is "
                          f"{L.max_open_positions}")

        # 3. new trades this session
        checks["trades_this_session"] = portfolio.trades_this_session
        if portfolio.trades_this_session >= L.max_new_trades_per_session:
            return reject(RejectReason.RISK_MAX_TRADES_PER_SESSION,
                          f"{portfolio.trades_this_session} trade(s) already taken this session, "
                          f"limit is {L.max_new_trades_per_session}")

        # 4. daily loss limit - a breach stops new risk for the rest of the session
        checks["realized_pnl_today"] = portfolio.realized_pnl_today
        if portfolio.realized_pnl_today <= -abs(L.max_daily_strategy_loss):
            return reject(RejectReason.RISK_DAILY_LOSS_REACHED,
                          f"realized loss today ${portfolio.realized_pnl_today:,.2f} has reached "
                          f"the ${L.max_daily_strategy_loss:,.2f} daily limit")

        # 5. concentration
        if portfolio.count_ticker(trade.ticker) >= L.max_positions_per_ticker:
            return reject(RejectReason.RISK_TICKER_CONCENTRATION,
                          f"already holding {trade.ticker}")
        if trade.sector and portfolio.count_sector(trade.sector) >= L.max_positions_per_sector:
            return reject(RejectReason.RISK_SECTOR_CONCENTRATION,
                          f"already holding a position in sector {trade.sector}")

        # 6. aggregate premium across the book
        agg = portfolio.aggregate_premium_at_risk + par
        checks["aggregate_premium_after"] = agg
        if agg > L.max_aggregate_premium:
            return reject(RejectReason.RISK_AGGREGATE_PREMIUM,
                          f"aggregate premium at risk would be ${agg:,.2f}, limit is "
                          f"${L.max_aggregate_premium:,.2f}")

        # 7. cash - the account must actually be able to pay
        checks["cash"] = portfolio.cash
        if par > portfolio.cash:
            return reject(RejectReason.INSUFFICIENT_CASH,
                          f"needs ${par:,.2f}, cash is ${portfolio.cash:,.2f}")

        return RiskDecision(approved=True, premium_at_risk=par, checks=checks)
