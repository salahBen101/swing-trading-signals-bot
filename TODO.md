# TODO — milestone checklist

Live tracker for `PLAN.md`. `[x]` means implemented **and** covered by a test that fails if
the behaviour regresses.

Baseline at start: 27 pre-existing scanner tests passing. That number must never go down.

Mission-v2 baseline recovered 2026-08-22: 535 tests passed before new prop-profile work.
Stage 3 and Stage 4 remain blocked.

---

## M1 — Foundations  *(done)*
- [x] `core/types.py` — Side, OrderType, TimeInForce, OrderStatus, ExitReason, RejectReason
- [x] `core/models.py` — Bar, OrderIntent, Order, Fill, Position, Trade
- [x] `core/clock.py` — tz-aware clock seam (real / simulated)
- [x] `instruments/registry.py` — MNQ, NQ, MES, ES specs; tick rounding helpers
- [x] `config/` loader — YAML into frozen dataclasses, validated, env overrides
- [x] tests: MNQ spec pinned ($2/pt, 0.25 tick, $0.50/tick); tick rounding; config errors name the key

## M2 — Data  *(done: 72 tests)*
- [x] `data/schema.py` — bar integrity validation
- [x] `data/store.py` — parquet BarStore, idempotent import, manifest with content hash
- [x] `data/splits.py` — locked DEV / VALIDATION / HOLDOUT constants
- [x] `data/feed.py` — MarketDataFeed protocol, ReplayFeed, closed-bars-only guarantee
- [x] tests: corrupt OHLC rejected; duplicate timestamps rejected; partial bar never delivered; splits are disjoint and cover the sample
- [x] bad-tick threshold scales as sqrt(bar duration) — a flat 2% is right at 15s and wrong at 5min
- [x] the real 9-year archive validates at 1min/5min/15min (skipped when absent)

## M3 — Features  *(done)*
- [x] `features/` — ATR, RSI, EMA/SMA, session VWAP, Bollinger, Donchian, ADX, efficiency ratio, realized vol, opening range, swing highs/lows
- [x] `features/pipeline.py` — assemble the causal frame for a strategy
- [x] tests: prefix-equality for every indicator; no NaN leakage past warmup; VWAP resets at the session boundary

## M4 — Strategy framework  *(done)*
- [x] `strategy/base.py` — StrategySpec, Strategy ABC, StrategyContext
- [x] `strategy/registry.py` — name to class
- [x] `strategy/orb_breakout.py`
- [x] `strategy/sr_rejection.py`
- [x] `strategy/sr_breakout_retest.py`
- [x] `strategy/vwap_reversion.py`
- [x] `strategy/vwap_continuation.py`
- [x] `strategy/trend_pullback.py`
- [x] tests: each family fires on a hand-built bar sequence and stays silent on a near-miss; max-trades-per-session enforced; stop and target placed on the correct side

## M5 — Risk engine  *(done)*
- [x] `risk/limits.py` — RiskEngine, RiskState, all hard limits
- [x] `risk/sizing.py` — volatility-targeted sizer with cushion throttle
- [x] `risk/tokens.py` — RiskToken mint, binding hash, single use, expiry
- [x] `risk/killswitch.py` — flag file, trip and clear
- [x] tests: one test per limit at the boundary and one tick past; sizing floors and caps; duplicate intent rejected; kill switch blocks and flattens

## M6 — Broker layer  *(done)*
- [x] `broker/base.py` — BrokerAdapter protocol, BrokerError taxonomy
- [x] `broker/simulated.py` — deterministic sim: latency, partial fills, rejects, disconnects
- [x] `broker/guarded.py` — GuardedBroker token gate plus independent re-validation
- [x] tests: tokenless order rejected; forged/replayed/expired token rejected; stale approval past a fresh breach rejected; partial fill accumulates; disconnect surfaces as a typed error
- [x] OCO groups so the protective stop/target pair rests at the venue and self-cancels
- [x] stops matched before limits, so a bar spanning both takes the pessimistic branch

## M7 — Execution and journal  *(done: 401 tests)*
- [x] `journal/db.py` — SQLite schema, migrations, append API
- [x] `journal/queries.py` — read side for analytics and dashboard
- [x] `execution/engine.py` — intent to approval to order to fill to trade; reconciliation; recovery
- [x] tests: every decision lands a row; restart rebuilds state; broker truth wins on divergence; API failure mid-order does not double-send
- [x] regression: events drained in exactly one place (double-drain doubled every position)
- [x] cancelling protection is best-effort and can never abort a flatten

## M8 — Backtest and analytics
- [x] `broker/costs.py` — shared commission and slippage model
- [x] `broker/simulated.py` — deterministic simulator driving the shared execution path
- [x] `backtest/engine.py` — event loop, next-bar fills, pessimistic stop-first
- [x] `analytics/metrics.py` — the full statistic set from spec §6
- [x] `analytics/report.py` — text and HTML report
- [x] tests: no-look-ahead guard (NaN-poisoned future); fill never on the signal bar; stop-before-target when a bar spans both; gap fills at the open; metrics verified against a hand-computed trade set

## M9 — Monitoring and dashboard
- [x] `monitoring/health.py` — heartbeat, staleness, error counter, state machine
- [x] `dashboard/server.py` — stdlib HTTP, JSON API, static page
- [x] `dashboard/static/` — prop/bot/trade/statistics page, audit views, large STOP
- [x] tests: staleness trips at the threshold; STOP endpoint trips the kill switch; API is read-only otherwise and cannot resume

## M10 — Tradovate demo adapter
- [x] `broker/tradovate/transport.py` — injectable HTTP seam
- [x] `broker/tradovate/auth.py` — access token, expiry, renewal
- [x] `broker/tradovate/adapter.py` — place/cancel/list, isAutomated, symbol mapping
- [x] `broker/tradovate/ws.py` — frame parsing (o / h / a / c), authorize, reconnect
- [x] tests: auth request shape; renewal before expiry; live host raises LiveTradingDisabled; 401/429/500 map to typed errors with backoff; WS heartbeat and reconnect; no credential literals in source

## M11 — Paper-trading runner
- [ ] `app/paper.py` — the daemon
- [ ] `app/cli.py` — `import-data`, `backtest`, `paper`, `dashboard`, `broker-check`
- [ ] tests: a full replay session produces journalled trades, flattens before the close, and holds no overnight position

## M12 — Documentation
- [x] `docs/ARCHITECTURE.md`
- [x] `docs/LIMITATIONS.md`
- [x] README section for the futures system
- [x] `.env.example` extended with Tradovate demo keys
- [ ] clean-venv setup walkthrough executed and corrected

---

## Definition of done (spec §13)

| # | criterion | status |
|---|---|---|
| 1 | Historical data can be imported | [x] |
| 2 | A strategy can be expressed deterministically | [x] |
| 3 | It can be backtested | [x] |
| 4 | Statistics are generated automatically | [x] |
| 5 | Risk rules cannot be bypassed by the strategy | [x] |
| 6 | The system can connect to a demo broker | [x] adapter/test seam; not an approved Tradeify route |
| 7 | It can paper trade automatically | [x] Stage-1 `market-replay` through the simulated broker |
| 8 | Every action is logged | [x] signals, reason-coded rejections, orders, order events, fills, trades, equity marks and events |
| 9 | Results appear on a web dashboard | [x] |
| 10 | Automated tests pass | [x] reverified after each integration milestone |
| 11 | Setup instructions work from a clean machine | [x] uv and pip paths both verified on a fresh checkout |
| 12 | Architecture and limitations documented | [x] |

---

## M13 — Repository memory and official Tradeify profiles

- [x] `AGENTS.md`, `DECISIONS.md`, `EXPERIMENTS.md`, `CHANGELOG.md`
- [x] `docs/PROP_RULES.md` with official sources checked 2026-08-22
- [x] strict current-cohort Growth / Select / Lightning 50K YAML profiles
- [x] phase-specific target, EOD drawdown/lock, DLL, contracts, consistency, payout,
  hours, prohibited conduct, and automation restrictions
- [x] unresolved source conflicts and unavailable Evaluation/Sim Tradovate API recorded
- [x] strict profile-loader boundaries and daily freshness check
- [x] official-source content change detector and unapproved proposal/alert artifacts
- [x] reviewed source-text hash baselines captured from official help content and verified
  unchanged on 2026-08-22

## M14 — Three-layer and broker-authoritative risk

- [x] explicit StrategyGate decision trace
- [x] personal defaults pinned: $200/trade, 1 trade/session, $200/day, 1 position
- [x] explicit PropFirmRiskGate consuming selected profile/phase
- [x] EOD HWM/floor/lock, DLL escalation, internal buffer, contracts, consistency/payout
- [x] broker guard fetches fresh account/positions/orders and reserves worst-case risk
- [ ] persisted/reconciled risk state keeps restart locked until authoritative recovery
- [x] typed order purpose and non-forgeable approvals bind the full canonical order/risk state
- [x] protective stop cannot be removed/widened; replace-before-cancel or flatten
- [ ] conservative post-gap/slippage/fee risk stays at or below $200

## M15 — Backtest safety hardening

- [x] entry-bar stop/target protection, pessimistic when both hit
- [x] engine/broker/trade equity includes entry and exit commissions identically
- [x] partial exits accumulate complete trade and daily-risk accounting
- [x] every session and end-of-data finishes broker-confirmed flat with no working entries
- [x] DEV is default; holdout requires explicit unlock/spend record
- [x] precomputed feature index/spec must match the selected split exactly
- [x] causal bounded context and prefix-equivalence across all strategy families

## M16 — Prop simulator and Monte Carlo

- [x] fresh-account phase simulator
- [x] evaluation-to-funded journey simulator
- [x] session-block bootstrap/permutation with deterministic seeds
- [x] pass/fail/internal-lock, days-to-pass, drawdown, payout, lifetime, expected profit,
  losing-sequence metrics
- [x] normal, stressed cost/slippage, and parameter-perturbation reports

## M17 — Stage control

- [x] explicit Stage 0–4 model and ordinary config hard-block for Stage 3/4
- [x] human-approved, hash-bound deployment manifest gate for Stage 3/4
- [x] production env overrides can tighten but never weaken pinned risk
- [x] Tradeify bot ownership/exclusivity/proof attestations in the manifest gate
- [ ] approved platform-native Evaluation route verified; Tradovate API remains blocked

## M18 — Operator loop and reporting

- [x] paper/replay runner and CLI (`market-replay`; journal + per-session reports)
- [x] dashboard shows complete prop/bot/current-trade/statistics state plus STOP
- [x] deterministic daily session report and abnormality flags (runner wiring pending)
- [x] deterministic weekly accepted/rejected-trade and survival research report (runner wiring pending)
- [x] daily official-rule verification record/alert command; broker-dependency polling pending
- [x] clean-machine verification (uv and pip, 866 passed / 3 skipped)
- [ ] live-runner wiring for the daily/weekly reporting and rule-polling commands
