"""Would options survive this edge, or consume it?

The RSI(2) signal produces a small underlying move over a short hold: roughly +0.7% over 3
days on average. Options add three costs that a share position does not pay - premium
(extrinsic value), bid-ask spread, and time decay across the hold. Whether the strategy
survives depends on how those compare to the move being captured.

Option prices here come from Black-Scholes with realistic implied volatilities rather than a
live chain, which is enough to answer a question about orders of magnitude. The conclusion is
not sensitive to a point or two of IV.
"""

from __future__ import annotations

import argparse
from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd

from rsi2_validation import backtest, stats  # noqa: E402


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, K - S)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def simulate_option_trade(
    entry_price: float,
    exit_price: float,
    days_held: int,
    *,
    strategy: str,
    dte: int,
    iv: float,
    r: float = 0.045,
    spread_pct: float = 0.02,
) -> dict:
    """Price the option at entry and exit, charging half the bid-ask spread on each side.

    IV is held constant. In practice the signal fires after a selloff, when IV is ELEVATED and
    then falls as price recovers - so a long-option position also fights vol crush. Holding IV
    flat is therefore generous to the long-option cases below.
    """
    T_entry = dte / 365.0
    T_exit = max(dte - days_held, 0) / 365.0
    S0, S1 = entry_price, exit_price

    if strategy == "long_call_atm":
        K = S0
        cost = bs_call(S0, K, T_entry, r, iv)
        value = bs_call(S1, K, T_exit, r, iv)
    elif strategy == "long_call_itm":
        K = S0 * 0.95  # deep in the money: mostly intrinsic, little time value
        cost = bs_call(S0, K, T_entry, r, iv)
        value = bs_call(S1, K, T_exit, r, iv)
    elif strategy == "short_put_spread":
        # Bullish,収 premium: short an ATM put, long one 3% below as protection.
        short_k, long_k = S0, S0 * 0.97
        credit = bs_put(S0, short_k, T_entry, r, iv) - bs_put(S0, long_k, T_entry, r, iv)
        close_cost = bs_put(S1, short_k, T_exit, r, iv) - bs_put(S1, long_k, T_exit, r, iv)
        entry_fee = credit * spread_pct
        exit_fee = close_cost * spread_pct
        pnl = credit - close_cost - entry_fee - exit_fee
        max_risk = (short_k - long_k) - credit
        return {"pnl": pnl, "cost_basis": max_risk, "return_pct": pnl / max_risk * 100 if max_risk > 0 else 0.0}
    else:
        raise ValueError(strategy)

    entry_fee = cost * spread_pct
    exit_fee = value * spread_pct
    pnl = value - cost - entry_fee - exit_fee
    return {"pnl": pnl, "cost_basis": cost, "return_pct": pnl / cost * 100 if cost > 0 else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--entry-rsi", type=float, default=5.0)
    parser.add_argument("--iv", type=float, default=0.20)
    args = parser.parse_args()

    bars = pd.read_parquet(f"data_cache/{args.ticker}_daily.parquet").dropna()
    trades = backtest(bars, entry_rsi=args.entry_rsi)
    base = stats(trades, bars)

    print("=" * 78)
    print(f"RSI(2) VIA OPTIONS - {args.ticker}")
    print("=" * 78)
    print(f"Underlying signal: {base['n']} trades, win {base['win_pct']}%, "
          f"avg {base['avg_pct']:+.3f}%, avg hold {base['avg_days']} days")
    print(f"Assumed IV {args.iv:.0%}, bid-ask 2% of premium, rate 4.5%\n")

    print("The question: does a +0.7% average move over ~3 days survive option costs?\n")

    rows = []
    for strategy, dte, label in (
        ("long_call_atm", 7, "long ATM call, 7 DTE"),
        ("long_call_atm", 30, "long ATM call, 30 DTE"),
        ("long_call_itm", 30, "long 5% ITM call, 30 DTE"),
        ("short_put_spread", 7, "short put spread, 7 DTE"),
        ("short_put_spread", 30, "short put spread, 30 DTE"),
    ):
        results = []
        for _, t in trades.iterrows():
            res = simulate_option_trade(
                t["entry"], t["exit"], int(t["days"]), strategy=strategy, dte=dte, iv=args.iv
            )
            results.append(res["return_pct"])
        arr = np.array(results)
        rows.append(
            {
                "structure": label,
                "n": len(arr),
                "win%": round((arr > 0).mean() * 100, 1),
                "avg_ret%": round(arr.mean(), 2),
                "median%": round(float(np.median(arr)), 2),
                "worst%": round(arr.min(), 1),
                "best%": round(arr.max(), 1),
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print("\n" + "=" * 78)
    print("SHARES BASELINE, for comparison")
    print("=" * 78)
    print(f"  buying the underlying: win {base['win_pct']}%, avg {base['avg_pct']:+.3f}% per trade")
    print("  (no premium, no theta, no vol crush - just spread and commission)")

    print("\n" + "=" * 78)
    print("WHAT THIS MEANS")
    print("=" * 78)
    print("  A long option must overcome premium, spread and decay before the underlying move")
    print("  counts. The signal produces roughly +0.7% over three days, which is comparable to")
    print("  what a short-dated option costs outright.")
    print("\n  There is a further problem the numbers above understate: this signal fires AFTER")
    print("  a selloff, precisely when implied volatility is elevated. As price recovers, IV")
    print("  falls. A long option is then fighting vol crush at the same time as theta, while")
    print("  a short-premium structure is helped by it. Holding IV constant, as done here,")
    print("  flatters the long-option cases.")
    print("\n  Structures that SELL premium fit a high-win-rate, small-move signal far better")
    print("  than structures that buy it - but they carry the mirror-image risk profile:")
    print("  many small wins and occasional large losses, which is exactly the shape that")
    print("  ends accounts if position sizing is wrong.")


if __name__ == "__main__":
    main()
