# Trading research workspace

This repository contains two independent projects:

- `scanner/`: the original end-of-day RSI(2) stock/ETF research scanner.
- `src/tradebot/`: a survival-first, deterministic MNQ/MES intraday futures system for
  researching $50,000 prop-account constraints.

The futures system currently exposes **Stage 0 backtests** and a bounded **Stage 1 local
market replay**. It cannot automatically enable paper connectivity, a prop evaluation,
a funded account, or a live route, and no included strategy is claimed to have a validated
edge.

## MNQ prop-account system

The futures path separates deterministic strategy logic, personal risk, dated prop-firm
rules, execution, and broker access. Defaults are one open position, one trade per session,
at most $200 planned risk, and a $200 daily strategy loss. Contract size is derived from
stop distance and MNQ's $0.50 tick value, then capped; strategies never select size.

Stage 3/4 requires separate human approval bound to the exact code, strategy, runtime
configuration, profile, and official-rule snapshot. Current Tradeify ambiguities and the
absence of an approved Evaluation/Sim-Funded API route keep those stages blocked.

Setup for development:

```powershell
uv sync --locked --extra dev
uv run --locked --extra dev pytest
```

These commands use the checked-in `uv.lock`; install
[uv](https://docs.astral.sh/uv/getting-started/installation/) first if it is not already
available. Without uv, `pip install -r requirements.txt` then `pytest` works from the
repository root. Both paths are verified from a clean checkout and give 866 passed with 3
skipped — the three skips are integrity checks against the nine-year NQ archive, which is
gitignored and absent from a fresh clone.

The operator CLI can be invoked with `uv run --locked tradebot`. Its first three commands
are safety checks (rule verification can append audit/alert artifacts), while
`market-replay` writes a local SQLite journal and daily reports:

```powershell
uv run --locked tradebot config-check
uv run --locked tradebot prop-profile-check
uv run --locked tradebot verify-rules
uv run --locked tradebot market-replay --help
```

`market-replay` can submit orders only to the local simulated broker. No CLI command can
contact an external broker or grant Stage 2, Stage 3, or Stage 4 approval.

The shipped configuration is Stage 0 (backtest), so a replay has to be asked for
explicitly. It needs local DEV bars, a calendar classifying every session date, and a
rule-verification timestamp:

```powershell
$env:TRADEBOT__DEPLOYMENT__STAGE=1; $env:TRADEBOT__MODE="PAPER"; uv run --locked tradebot market-replay --data bars.parquet --calendar calendar.json --rules-at 2026-08-22T12:00:00-04:00 --minimum-rr 1.5 --journal logs/replay.sqlite3 --report-dir output/replay
```

A twelve-session run over March 2018 produces roughly 4,700 equity marks, 4,800 events,
4,400 reason-coded rejections, 11 trades and a per-session report pair, and ends
broker-confirmed flat with no working orders.

Start with [the architecture](docs/ARCHITECTURE.md),
[known limitations](docs/LIMITATIONS.md), and
[the dated Tradeify rule ledger](docs/PROP_RULES.md). Project priorities and unfinished
work live in `PROJECT_SPEC.md`, `PLAN.md`, and `TODO.md`.

## RSI(2) scanner

A little end-of-day scanner I built to flag short-term pullbacks in US stocks
and ETFs. It runs the classic RSI(2) mean-reversion idea (Connors) over a
watchlist, prints what's oversold/overbought, and can email or toast me the
alerts. There's also a volatility-squeeze watch bolted on.

It's a research console, not a trading bot. It doesn't touch a broker, place
orders, or track positions. Nothing here is financial advice.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# run it (no notifications, just print)
python scanner\rsi2_scanner.py --tickers all --no-notify
```

A few presets:

```powershell
python scanner\rsi2_scanner.py --tickers core --no-notify
python scanner\rsi2_scanner.py --tickers AAPL MSFT SPY --no-notify
```

`all` is a ~120-name observation universe. By default, notifications are limited
to a small diversified ETF core (SPY, QQQ, IWM, DIA, MDY) so I'm not spammed by
the wider watchlist. See `scanner/README.md` for the full flag list.

## What's in here

- `scanner/` – the scanner itself: RSI(2) logic, the squeeze watch, the ticker
  universe, market-context filter, and notifications.
- `scripts/` – the validation stuff I used to convince myself the edge was real
  (holdout tests, portfolio checks, universe checks).
- `tests/` – unit tests for the strategy and market-context code.
- `scan.bat` / `setup_daily_scan.ps1` – run it daily via Windows Task Scheduler.

## Notifications

Copy `.env.example` to `.env` and fill in your email settings if you want alerts
(Gmail needs an app password). `.env` is gitignored, so your credentials stay
local. Without it the scanner just prints to the console. See `EMAIL_SETUP.md`.

## A note on the results

I tried a bunch of ideas before this one and most of them died out of sample —
that's kind of the point of the validation scripts. RSI(2) is the one that held
up on data I hadn't looked at. Take the signals as a starting point for your own
homework, not a green light.

## Tests

```powershell
pip install -r requirements.txt
pytest
```
