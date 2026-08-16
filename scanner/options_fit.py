"""For a triggered RSI(2) signal, pull a live option chain and show its mechanics.

This is an *illustrative scenario panel*, not an options backtest or an expected-value model.
The RSI research covers shares, not option contracts; it does not contain historical implied
volatility, fills, or a per-ticker option-return distribution. A fixed-IV Black-Scholes
calculation can demonstrate spread, theta, and break-even risk, but must never be interpreted
as a prediction or a contract recommendation.

Two costs matter more than most people expect and are shown explicitly:

  * The bid-ask spread. Paying the ask and selling the bid is a real, immediate loss. On thin
    contracts it can exceed the entire expected move.
  * Time decay across the hold. Short-dated options lose value fastest, so the cheapest
    contract is frequently the worst one for a 3-4 day trade.

A third cost cannot be shown from a static chain and is flagged instead: this signal fires
after a selloff, when implied volatility is elevated. As price recovers, IV typically falls,
which hurts long premium. The projections here hold IV constant and are therefore optimistic
for long calls.

This prints data. It does not recommend a contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import erf, exp, isfinite, log, sqrt
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

MARKET_TZ = ZoneInfo("America/New_York")
RISK_FREE = 0.045

# Scenario move shown only to make the scale of option costs legible. It is not transferred
# from the Mag 7 or any historical share backtest into an options "EV".
EXPECTED_MOVE_PCT = 1.17
EXPECTED_HOLD_DAYS = 3.5

# A downside stress scenario, not a calibrated loss distribution.
ADVERSE_MOVE_PCT = -2.0


def _finite_float(value: object, default: float = 0.0) -> float:
    """Return a finite float without treating NaN as a usable market quote."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if isfinite(parsed) else default


def _nonnegative_int(value: object) -> int:
    """Option-chain volume fields are sometimes blank/NaN before the open."""
    return max(0, int(_finite_float(value)))


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)


def call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return norm_cdf(d1)


@dataclass
class Candidate:
    expiry: str
    dte: int
    strike: float
    bid: float
    ask: float
    mid: float
    spread_pct: float
    iv: float
    delta: float
    open_interest: int
    volume: int
    moneyness: str
    intrinsic: float
    extrinsic: float
    projected_value: float
    projected_return_pct: float
    adverse_return_pct: float
    breakeven: float
    breakeven_move_pct: float

    @property
    def cost_per_contract(self) -> float:
        return self.ask * 100

    @property
    def tradeable(self) -> bool:
        """Wide spreads and empty books make a contract unusable regardless of its theory."""
        return self.spread_pct < 15 and self.open_interest >= 50 and self.ask > 0


def fetch_candidates(
    ticker: str,
    spot: float,
    *,
    min_dte: int = 5,
    max_dte: int = 45,
    strike_window_pct: float = 8.0,
) -> list[Candidate]:
    if not isfinite(spot) or spot <= 0:
        raise ValueError("spot must be a finite positive price")

    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        return []

    today = datetime.now(MARKET_TZ).date()
    out: list[Candidate] = []

    for expiry in expiries:
        try:
            exp_date = datetime.strptime(str(expiry), "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        dte = (exp_date - today).days
        if not (min_dte <= dte <= max_dte):
            continue

        try:
            chain = tk.option_chain(expiry).calls
        except Exception:
            continue
        if chain is None or chain.empty:
            continue

        lo = spot * (1 - strike_window_pct / 100)
        hi = spot * (1 + strike_window_pct / 100)
        if "strike" not in chain:
            continue
        strikes = pd.to_numeric(chain["strike"], errors="coerce")
        chain = chain.loc[(strikes >= lo) & (strikes <= hi)]

        for _, row in chain.iterrows():
            bid = max(0.0, _finite_float(row.get("bid")))
            ask = _finite_float(row.get("ask"))
            if ask <= 0:
                continue
            mid = (bid + ask) / 2 if bid > 0 else ask
            spread_pct = (ask - bid) / mid * 100 if mid > 0 else 100.0
            iv = _finite_float(row.get("impliedVolatility"))
            if iv <= 0:
                iv = 0.30
            strike = _finite_float(row.get("strike"))
            if strike <= 0:
                continue

            T_now = dte / 365.0
            T_exit = max(dte - EXPECTED_HOLD_DAYS, 0) / 365.0
            delta = call_delta(spot, strike, T_now, RISK_FREE, iv)

            def value_after(move_pct: float) -> float:
                target = spot * (1 + move_pct / 100)
                theo = bs_call(target, strike, T_exit, RISK_FREE, iv)
                # Bought at the ask, sold into the bid: charge the quoted spread both ways.
                return theo * (1 - (spread_pct / 100) / 2)

            exit_proceeds = value_after(EXPECTED_MOVE_PCT)
            adverse_proceeds = value_after(ADVERSE_MOVE_PCT)
            projected_return = (exit_proceeds - ask) / ask * 100 if ask > 0 else 0.0
            adverse_return = (adverse_proceeds - ask) / ask * 100 if ask > 0 else 0.0
            intrinsic = max(0.0, spot - strike)
            moneyness = "ITM" if strike < spot * 0.995 else ("ATM" if strike <= spot * 1.005 else "OTM")

            out.append(
                Candidate(
                    expiry=expiry,
                    dte=dte,
                    strike=strike,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    spread_pct=spread_pct,
                    iv=iv,
                    delta=delta,
                    open_interest=_nonnegative_int(row.get("openInterest")),
                    volume=_nonnegative_int(row.get("volume")),
                    moneyness=moneyness,
                    intrinsic=intrinsic,
                    extrinsic=ask - intrinsic,
                    projected_value=exit_proceeds,
                    projected_return_pct=projected_return,
                    adverse_return_pct=adverse_return,
                    breakeven=strike + ask,
                    breakeven_move_pct=(strike + ask) / spot * 100 - 100,
                )
            )
    return out


def render_options(ticker: str, spot: float, candidates: list[Candidate]) -> None:
    print(f"\n  {'-' * 78}")
    print(f"  OPTIONS FIT - {ticker} @ {spot:,.2f}")
    print(f"  {'-' * 78}")

    if not candidates:
        print("    No chain data available (market may be closed, or no listed expiries in range).")
        return

    tradeable = [c for c in candidates if c.tradeable]
    rejected = len(candidates) - len(tradeable)

    print(f"    Scenario only: +{EXPECTED_MOVE_PCT:.2f}% underlying move over "
          f"{EXPECTED_HOLD_DAYS:.1f} sessions, with IV held constant.")
    print(f"    {len(candidates)} contracts in range, {len(tradeable)} pass liquidity "
          f"(spread <15%, OI >=50).")
    if rejected:
        print(f"    {rejected} rejected as untradeable - a wide spread is a guaranteed loss "
              f"before the trade starts.")

    if not tradeable:
        print("\n    Nothing passes the liquidity filter. Anything here would be bought at a")
        print("    price that already consumes the expected move.")
        return

    # Rank only by execution quality. An option profitability ranking would require historical
    # IV, actual bid/ask fills, and per-ticker option outcomes, none of which are available.
    ranked = sorted(tradeable, key=lambda c: (c.spread_pct, -c.open_interest, abs(c.delta - 0.7)))[:12]

    print(f"\n    {'expiry':<12} {'DTE':>4} {'strike':>9} {'M':>4} {'ask':>7} {'spr%':>6} "
          f"{'delta':>6} {'OI':>7} {'if +1.2%':>9} {'if -2%':>8} {'B/E move':>9}")
    print("    " + "-" * 96)
    for c in ranked:
        print(
            f"    {c.expiry:<12} {c.dte:>4} {c.strike:>9,.1f} {c.moneyness:>4} {c.ask:>7.2f} "
            f"{c.spread_pct:>5.1f}% {c.delta:>6.2f} {c.open_interest:>7,} "
            f"{c.projected_return_pct:>+8.1f}% {c.adverse_return_pct:>+7.1f}% "
            f"{c.breakeven_move_pct:>+8.1f}%"
        )

    most_liquid = ranked[0]
    print(f"\n    Reading this:")
    print(f"      if +1.2%  = constant-IV scenario, not a forecast "
          f"(+{EXPECTED_MOVE_PCT:.2f}% over {EXPECTED_HOLD_DAYS:.1f} sessions)")
    print(f"      if -2%    = downside stress scenario, not a worst case")
    print(f"      B/E move  = expiration break-even, which understates the move needed over "
          f"a {EXPECTED_HOLD_DAYS:.1f}-session hold because theta still applies")
    print(f"\n    Most liquid displayed contract: {most_liquid.expiry} "
          f"{most_liquid.strike:,.1f} strike, ${most_liquid.cost_per_contract:,.0f} per contract")
    print(f"      ${most_liquid.extrinsic * 100:,.0f} is time value and can decay even if the")
    print("      underlying is unchanged. No contract is labelled positive-EV or recommended.")

    print(f"\n    Two things not modelled, both unfavourable to buying calls:")
    print(f"      1. IV crush. This signal fires after a selloff when IV is elevated; IV")
    print(f"         typically falls as price recovers. These figures hold IV fixed.")
    print(f"      2. The -2% stress case is not a worst case. A gap")
    print(f"         down on news can take a short-dated call to near zero.")
