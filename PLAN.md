# PLAN — how the system in `PROJECT_SPEC.md` gets built

Version 2.0, 2026-08-22.

---

## 0. Starting position

This repository already contains a working end-of-day RSI(2) equity scanner
(`scanner/`, `scripts/`, 27 passing tests). **That stays untouched.** The futures system
lands beside it in `src/tradebot/`, with its own tests under `tests/tradebot/`. Nothing in
`scanner/` is imported by the new package and nothing new is imported by the scanner.

Assets already present that the plan uses:

| asset | what it gives us |
|---|---|
| `data_cache/nq_1m.parquet` | 898,186 NQ 1-minute RTH bars, 2017-06-30 to 2026-07-29 |
| `data_cache/nq_15s.parquet` | 788,370 15-second bars, 2024-08-05 to 2026-08-03 |
| `full-history-backup` branch | a previous `src/nq_scalper/` implementation and its research notes |
| `.venv` | Python 3.14, pandas 3.0.5, numpy 2.5.1, pyarrow, databento, pytest 9.1.1 |

The old `nq_scalper` package is **not** resurrected wholesale. Its good ideas are carried
over deliberately (next-bar fills, pessimistic stop-first, causal features, locked splits,
the data-integrity guard and its calibration note). Its gaps are exactly what this build
adds: an execution engine, a broker adapter, a persistent journal, monitoring, a dashboard,
and the structural non-bypass property.

The stale `config/risk.yaml` and `config/strategy.yaml` reference `src/nq_scalper/...`,
which no longer exists, and carry NQ's $20 multiplier where this project needs MNQ's $2.
They get replaced, not edited around.

---

## 1. Sequencing principle

Build **inward-out along the safety path**, not feature-first. The order is chosen so that
each milestone can be tested with only what came before it, and so that the components that
can lose money exist *after* the components that constrain them:

```
core types -> instruments -> data -> features -> strategy
                                                    |
                                      risk <--------+
                                        |
                          broker (sim) -+-> execution -> journal
                                                    |
                                    analytics ------+-> monitoring -> dashboard
                                        |
                                    backtest
```

Risk is built before execution. Execution is built before any real broker. The Tradovate
adapter is built last among the runtime pieces, so that by the time real HTTP exists,
everything it could do wrong is already fenced by tested code.

---

## 2. Milestones

Each milestone lands with its tests. `TODO.md` carries the live checklist; this section
explains the intent and the risk of each.

### M1 — Foundations: `core`, `instruments`, config

Value types (`Side`, `OrderType`, `Bar`, `OrderIntent`, `Order`, `Fill`, `Position`,
`Trade`, `ExitReason`, `RejectReason`), the contract registry, and typed config loading
from YAML with environment overrides.

*Risk:* getting the multiplier wrong silently scales every P&L number in the project. The
registry is therefore the single source, and a test pins MNQ at $2/point, 0.25 tick,
$0.50/tick.

### M2 — Data: store, integrity, splits, feeds

Parquet-backed `BarStore` with idempotent import and a manifest; `validate_bars` carried
over and re-calibrated; locked `splits.py`; `MarketDataFeed` protocol with `ReplayFeed`.

*Risk:* silently delivering an unclosed bar. The feed API only ever yields closed bars, and
a test asserts a partial bar is withheld until its interval elapses.

### M3 — Features: causal indicator library

ATR, RSI, EMA/SMA, VWAP (session-anchored), Bollinger, Donchian, ADX, efficiency ratio,
realized volatility, opening range, rolling swing highs/lows.

*Risk:* an accidentally centred or forward-filled window. Every indicator gets a
prefix-equality test: computing on `bars[:k]` must match the first `k` values of computing
on all bars.

### M4 — Strategy framework and six families

`StrategySpec` as declarative data; `Strategy` base with `on_bar(ctx) -> OrderIntent | None`
and `manage(ctx, position) -> ExitDecision | StopUpdate | None`; a registry so a strategy is
selected by name from config; the six families in §5 of the spec.

*Risk:* discretionary logic leaking in. Conditions are named predicates recorded on the
intent, so every trade carries the reason it was taken.

### M5 — Risk engine

Limits, volatility sizing, session gating, consecutive-loss cooldown, kill switch, and the
`RiskToken` mint. Rejections carry machine-readable reason codes.

*Risk:* this is the module where a bug costs money. Every limit gets a test that drives the
engine to the boundary and one tick past it.

### M6 — Broker layer: protocol, `SimulatedBroker`, `GuardedBroker`

The adapter protocol; a deterministic local simulator with configurable latency, partial
fills, rejects and disconnects; and the guard wrapper that makes a tokenless order
unrepresentable and re-validates independently.

*Risk:* the guard being wrapped *around the wrong thing* or bypassable by holding the inner
adapter. The execution engine only ever receives the guarded instance, and a test asserts
that calling the inner adapter directly is not reachable from a strategy.

### M7 — Execution engine and journal

Intent to approval to order to fill to trade, with position reconciliation, restart
recovery from the journal plus broker truth, and SQLite persistence for signals,
rejections, orders, fills, trades, equity and events.

*Risk:* restart divergence — the journal saying one thing and the broker another. Broker
state wins; the difference is journalled as a reconciliation event.

### M8 — Backtest engine and analytics

Event-driven next-bar-open simulation reusing the same `Strategy`, `RiskEngine` and cost
model as live, so a backtest and a paper run cannot drift apart. Full statistics module.

*Risk:* a backtest that flatters itself relative to the live path. Mitigated by sharing the
code path: the backtest drives the *same* execution and risk objects through a
`BacktestBroker`.

### M9 — Monitoring and dashboard

Health state, staleness detection, error counting, heartbeat; stdlib HTTP server exposing a
JSON API and a single-page dashboard with the emergency STOP.

*Risk:* a dashboard that can place orders. The API is read-only apart from STOP and
resume-request, and STOP only ever *reduces* permission.

### M10 — Tradovate demo adapter

Real REST against the demo host, token lifecycle with renewal, `isAutomated: true`,
WebSocket frame handling, and the live-host lockout.

*Risk:* accidentally reaching a live endpoint. The live branch raises unconditionally in
v1, credentials are separate, and tests assert both.

### M11 — Paper-trading runner

The daemon that wires feed to strategy to risk to execution to broker to journal to
monitor, with graceful shutdown and end-of-session flatten.

### M12 — Documentation and clean-machine verification

`docs/ARCHITECTURE.md`, `docs/LIMITATIONS.md`, README section, `.env.example`, and a
from-scratch setup walked through in a clean virtual environment.

---

## 3. Cross-cutting decisions, decided now

**No new runtime dependencies.** Everything ships on pandas / numpy / pyarrow / PyYAML /
stdlib. The dashboard uses `http.server`, the journal uses `sqlite3`, the Tradovate client
uses `urllib.request` behind a thin transport seam so tests inject a fake without needing
`requests` or a live socket. This keeps clean-machine setup to one `pip install -r`.

**One code path for backtest and live.** The strategy, the risk engine, the cost model and
the execution engine are the same objects in both. Only the broker and the feed differ.
This is the main structural defence against backtest/live divergence, and it is why the
broker protocol is deliberately narrow.

**Config is data, and typed.** YAML into frozen dataclasses, validated on load, with an
explicit error naming the offending key. No dictionary lookups scattered through the
runtime.

**Everything time-aware.** All timestamps are timezone-aware `US/Eastern`. Naive datetimes
are rejected at the boundary rather than coerced.

**Money in integers where it matters.** Prices are rounded to the instrument tick before
any comparison or fill, so floating-point drift cannot produce a fill at 18000.000000001.

---

## 4. What is deliberately deferred

Recorded here so it is a choice rather than an omission; carried into
`docs/LIMITATIONS.md` at M12.

- **Live trading.** Not implemented. Requires separate credentials, an explicit config
  change, and removal of a hard raise.
- **Real-time market data from Tradovate's market-data socket.** Paper trading in v1 is
  driven by the replay feed and by the simulator. The live-feed seam exists; the
  subscription implementation does not.
- **Bracket (OSO/OCO) orders at the broker.** v1 manages the stop and target in the
  execution engine and sends flattening orders. That is correct for a simulator and for a
  supervised paper run, but a production live system wants exchange-resident protective
  orders so a process crash cannot leave a naked position.
- **News-calendar blackout.** The config windows exist; no feed populates them.
- **Multi-instrument and multi-strategy concurrency.** The registry and instrument
  abstraction make it additive, but v1 runs one strategy on one instrument.
- **Tick-level fill modelling.** Fills are modelled from OHLCV bars with a pessimistic
  path assumption. Sub-bar sequencing is not recoverable from this data.

---

## 5. How "done" is judged

The twelve criteria in `PROJECT_SPEC.md` §13, each demonstrated by something runnable:

| criterion | demonstrated by |
|---|---|
| import historical data | `python -m tradebot import-data --dry-run` and the store tests |
| deterministic strategy | `StrategySpec` dump in the backtest report |
| backtestable | `python -m tradebot backtest --strategy orb_breakout --split dev` |
| automatic statistics | the report emitted by that command |
| risk not bypassable | the non-bypass test group, including the static import check |
| demo broker connection | `python -m tradebot broker-check` against the simulator, and against Tradovate demo when credentials are present |
| paper trades automatically | `python -m tradebot paper --replay ...` producing journalled trades |
| everything logged | journal row counts asserted in the runner test |
| dashboard | `python -m tradebot dashboard` serving the live JSON and page |
| tests pass | `pytest` green, including the 27 pre-existing scanner tests |
| clean-machine setup | README walkthrough executed in a fresh venv |
| documented | `docs/ARCHITECTURE.md` and `docs/LIMITATIONS.md` |

A profitable strategy is **not** on this list, by design.

---

## 6. Mission-v2 milestones

The recovered work was ahead of its checklist, but a safety audit found gaps hidden by
green tests. The next milestones harden the path before adding more broker functionality.

### M13 — Repository memory and official prop profiles

Create the durable ledgers, verify current official Tradeify sources, document conflicts,
and load strict current-cohort Growth/Select/Lightning 50K phase profiles. Profiles expire
for production use after the configured verification interval. Legacy and Elite Live terms
require separate files.

### M14 — Three-layer risk and authoritative account state

Replace the guessed percentage drawdown in runtime decisions with explicit strategy,
personal, and prop gates. Pin personal defaults at $200 / one trade / $200 / one position.
Track EOD HWM, hard floor/lock, DLL, safety buffer, consistency, payout, contracts, and
hours. The final guard uses fresh broker account/position/order truth, reserves exposure,
and fails closed across restart.

Before integrating with a broker, fix approval-token field binding, typed order purposes,
protective cancellation/replacement, and exposure-reducing proof.

### M15 — Backtest safety corrections

Correct same-entry-bar protection, commission/equity reconciliation, partial exits,
end-of-session/data flattening, feature/split alignment, and DEV-by-default behavior.
Bound strategy context to causal history and add prefix equivalence across every strategy.

### M16 — Prop simulator and Monte Carlo

Run every phase from a fresh account, then support evaluation-to-funded journeys. Apply
real-time breach and EOD update/lock semantics, DLL pauses, consistency, scaling and payout
tests. Bootstrap whole Tradeify-session blocks and report survival-first metrics.

### M17 — Stage manifests and immutable production

Model stages 0–4. Stage 3/4 requires a human-approved manifest binding code, strategy,
config, official rule snapshot, account, and execution route. Preserve the hard live lock.
Reject Tradeify Evaluation/Sim use of the unavailable Tradovate API route.

### M18 — Operator loop

Complete CLI and replay/paper runner, dashboard prop state and STOP control, daily report,
weekly research report, official-source change detector, and clean-machine walkthrough.
Only after all preceding safety gates may an operator consider a Stage 3 manifest.
