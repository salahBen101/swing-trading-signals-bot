# RSI(2) scanner

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
