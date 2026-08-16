# RSI(2) Pullback Scanner — research / paper-trading only

This is a 120-symbol daily RSI(2) **scanner**. It does not connect to a broker, place
orders, track positions, or enforce exits. Treat it as a research console until it has a
locked forward paper-trading record.

## Run it

```powershell
.venv\Scripts\python.exe scanner\rsi2_scanner.py --tickers all --no-notify
```

Useful presets:

```powershell
.venv\Scripts\python.exe scanner\rsi2_scanner.py --tickers core --no-notify
.venv\Scripts\python.exe scanner\rsi2_scanner.py --tickers index --no-notify
.venv\Scripts\python.exe scanner\rsi2_scanner.py --tickers AAPL MSFT SPY --no-notify
```

`all` remains a 120-name observation universe. Default notifications are deliberately
restricted to the pre-specified diversified ETF `core` (`SPY`, `QQQ`, `IWM`, `DIA`, `MDY`).
Use `--include-research-signals` only if you explicitly want alerts for the broader,
retrospectively labelled watchlist.

Options are off by default. `--options` shows only a constant-IV **scenario** for an enabled
signal; it does not calculate option expected value or recommend a contract.
`--force-options SPY` is available for a manual chain inspection.

## Volatility-compression watch

The scanner also displays a separate **volatility-compression watch** by default. It is not
an RSI alert, does not send notifications, and does not place orders. Hide it with
`--no-squeeze`.

This is a daily OHLCV **volatility squeeze**, not a literal short-squeeze detector. It marks:

1. `WATCH` — at least three consecutive closes with 20-period Bollinger Bands inside a
   20-period / 1.5 ATR Keltner Channel, with Bollinger width in the lowest 20% of the prior
   126 sessions and price above the 200-day SMA.
2. `ARMED` — a qualified squeeze occurred in the prior five sessions; wait for confirmation.
3. `CONFIRMED` — at a completed close, price is above the prior 20-day high with at least
   1.2× its prior 20-day average volume and above the 200-day SMA. The research model would
   enter at the following open, never at the signal close.

These labels are condition states, **not probability scores**. The repository has no timely
historical point-in-time short-interest, borrow/utilization, options-positioning, or
earnings-calendar dataset, so it cannot backtest a literal short-squeeze thesis or claim that
the extra data makes the setup profitable.

## Strict market-context gate

A `CONFIRMED` OHLCV breakout is now marked `QUALIFIED` only when the scanner finds a
separate, current, explicitly verified snapshot for **all four** inputs below. Otherwise it
is marked `SUPPRESSED` and remains only a technical observation. A qualified row still does
not send a notification, place an order, or mean “high probability”; it says only that the
data-quality checks passed.

| Input | Required snapshot fields | Freshness rule |
|---|---|---:|
| Short interest | Percent of float (decimal, e.g. `0.12` = 12%), days-to-cover, source, report date | ≤21 days |
| Borrow | Shares available, annualized fee %, source, observed timestamp | ≤24 hours |
| Earnings | Next report date, source, observed timestamp | ≤24 hours; no entry within 10 calendar days |
| Options | Total call and put open interest, source, observed timestamp | ≤24 hours |

The gate fails closed for a missing, malformed, future-dated, stale, or unverified component.
It also rejects an attempt to set `*_verified: true` without supplying that component's own
values, date, and source. That prevents a Yahoo convenience field from being relabelled as a
broker/vendor-verified fact.

### Populate the snapshots

The checked-in template is intentionally blank and disabled:
[`config/market_context_override.example.json`](../config/market_context_override.example.json).
Create a private per-symbol copy under `data_cache/market_context/overrides/`, fill all four
groups from sources you are entitled to use, and then run:

```powershell
.venv\Scripts\python.exe scripts\refresh_market_context.py --tickers AAPL NVDA
```

The refresh script writes dated JSON snapshots to `data_cache/market_context/`. That directory
is ignored by Git so vendor data and source details stay local. Then scan normally (or point
to another cache with `--context-dir`):

```powershell
.venv\Scripts\python.exe scanner\rsi2_scanner.py --tickers all --no-notify
```

Use a dated FINRA/exchange/licensed feed for short interest; a broker securities-lending or
locate feed (for example, an entitled IBKR feed) for borrow; an entitled option-chain source
for open interest; and company investor relations or an entitled earnings-calendar vendor for
earnings dates. The refresh helper can fetch current Yahoo earnings/option convenience values,
but marks them **unverified**. Do not set the template's verification flags to true for Yahoo
data alone: Yahoo has no borrow feed and does not give the timestamp provenance needed for a
strict short-interest/options qualification.

The enrichment gate has no historical backtest in this repository because the required
point-in-time borrow, options, and earnings inputs are not present in the historical cache.
Treat it as a source/freshness safety check and collect a forward paper-trading log before
judging whether it improves the technical baseline.

Re-run the frozen historical check with:

```powershell
.venv\Scripts\python.exe scripts\squeeze_validation.py --universe all --cost-bps 20 --hold-days 10
```

On the cached 120-name survivor universe, using a next-open entry, fixed 10-session open
exit, and a 20 bp round-trip cost, compression breakouts did **not** show a stable incremental
advantage over the same trend/volume/20-day-high breakout without recent compression:

| Date block | Compression mean net return | HAC t-stat | No-compression baseline mean | HAC t-stat |
|---|---:|---:|---:|---:|
| 2009–2018 | +49.0 bp | 2.56 | +7.6 bp | 1.19 |
| 2019–2024 | +13.3 bp | 0.87 | +44.3 bp | 2.50 |
| 2025–2026-08-05 | +70.4 bp | 0.89 | +89.6 bp | 1.53 |

That mixed result is why the panel remains an observation/paper-trading feature rather than a
“high probability” signal or alert. Do not retune the thresholds from these results; record
every future watch/armed/confirmed state and compare it with the frozen baseline.

## Rules actually tested

1. At a completed daily close, identify `RSI(2) < 5` while price is above its 200-day SMA.
2. Fill at the **next session's open**. This avoids assuming an executable fill at the same
   close that created the signal.
3. Once open, confirm an exit at a close where RSI(2) is above 65, close is above the 5-day
   SMA, or ten sessions have elapsed; fill that exit at the next session's open.

The console can show provisional intraday readings, but it suppresses notifications for scans
that start before 16:00 ET. The research model is after-close / next-open, not a 15:30
market-on-close strategy.

## Corrected historical check

Run the reproducible portfolio test against the local cache:

```powershell
.venv\Scripts\python.exe scripts\rsi2_portfolio_validation.py `
  --universe all --max-positions 10 --max-per-sector 2 --cost-bps 4
```

On the repository's cached data through 2026-08-05, this model produced the following
historical results:

| Assumption | 2009–2026 CAGR | Max drawdown | HAC t-stat |
|---|---:|---:|---:|
| 10 equal capital slots, 2 per sector, 4 bp round trip | +9.27% | 17.7% | 4.88 |
| Same rules, 20 bp round-trip cost stress | +5.46% | 23.3% | 3.05 |
| 2019–2026 later historical window, 4 bp | +8.85% | 13.4% | 2.99 |
| 2019–2026 later historical window, 20 bp | +5.01% | 14.2% | 1.81 |

These figures are more realistic than the earlier trade-by-trade headline because they use
next-open fills, a cash-constrained portfolio, a sector cap, dividend-aware returns, and
daily Newey-West-adjusted inference. They are still **not proof of live profitability**.

The broad index result is weaker: the corrected post-2009 SPY test was only `+0.309%` per
trade at 4 bp cost, with a 1.28% strategy CAGR versus 12.80% buy-and-hold. That regime
dependence matters more than any single attractive portfolio statistic.

The deliberately conservative five-ETF alert core produced only a 1.17% historical CAGR
(HAC t-stat 2.04) under its own five-slot model. It is the safer starting universe because it
was specified without ranking current stocks by their backtest, not because it has the most
attractive historical return.

## What remains unproven

- The 120-name set is today's survivor/liquidity universe, not a point-in-time universe with
  delisted securities. That creates survivorship bias.
- The 2019–2026 period informed previous `strong` / sector labels, so it is not an untouched
  holdout. Those labels are now shown as retrospective context only.
- Historical tests cannot fully model spreads, market impact, earnings gaps, halted stocks,
  tax, or live data failures. A single-stock RSI signal cannot tell a temporary dip from a
  permanent repricing.
- The shares research does not validate long calls, short premium, or any other option
  structure. The option panel is deliberately informational only.

## Recommended next step

Paper trade a pre-committed version for 6–12 months: the five-ETF core, 10 capital slots,
maximum two positions per sector, next-open fills, and no new single-stock trades through an
earnings window. Save each after-close signal, assumed fill, and actual next-open fill. Do not
retune thresholds or the universe during that period; that log is the first genuinely
out-of-sample evidence this project can produce.
