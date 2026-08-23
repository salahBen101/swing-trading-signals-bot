# PROJECT_SPEC — MNQ intraday futures trading system

Status: living document. Version 2.0, 2026-08-22.

This specifies *what the system must do and must never do*. `PLAN.md` says how it gets
built; `TODO.md` tracks milestones; `docs/ARCHITECTURE.md` describes what was actually
built and where it falls short.

---

## 1. Purpose and non-purpose

**Purpose.** A small, understandable, thoroughly tested futures trading system whose
behaviour can be measured. It ingests historical and live bars, evaluates deterministic
strategies, forces every order through a risk engine, executes against a paper broker,
records everything, and reports the results on a local dashboard.

**Non-purpose.** This project does not claim, assume, or aim to produce a profitable
strategy. Prior work in this repository screened thirteen intraday NQ strategy families
over nine years and none survived out-of-sample (see §12). The deliverable is the
*measurement apparatus* and the *safety apparatus*. A strategy that measures as
unprofitable is a successful output of this system, not a failure of it.

**Explicitly out of scope for v1:** live real-money trading, multi-instrument portfolios,
options, overnight/swing holds, discretionary override, ML signal generation.

---

## 2. Instrument and session

Primary instrument: **MNQ — Micro E-mini Nasdaq-100 futures (CME Globex).**

| property | value |
|---|---|
| multiplier | $2.00 x index |
| tick size | 0.25 index points |
| tick value | $0.50 |
| listed months | Mar / Jun / Sep / Dec (H, M, U, Z) |
| Globex hours | Sun 17:00 CT to Fri 16:00 CT, daily halt 16:00-17:00 CT |
| session traded | RTH only, 09:30-16:00 US/Eastern |

Contract specs live in a registry (`instruments/`), not in strategy code, so ES/MES/NQ can
be added by adding a row.

**Price data note.** The historical archive in `data_cache/` is E-mini **NQ** front-month
OHLCV, while risk and P&L use MNQ's $2 multiplier. This is a useful price-path proxy, not
MNQ execution evidence: the contracts can have different prints, depth, spread, and volume,
and every volume-derived feature here observes NQ rather than MNQ. Reports must state the
proxy explicitly. Actual contract-level MNQ data with documented provenance and roll logic
is required before a strategy can advance beyond preliminary research.

### Session rules (hard)

- Intraday only. **No overnight positions.** Any open position is force-flattened before
  the RTH close.
- **One open position at a time**, one instrument, in v1.
- No new entries in a configurable buffer after the open and before the close.
- Optional news-blackout windows, configurable, empty by default (no calendar feed wired).

---

## 3. Architecture — module boundaries

Eleven modules, each independently testable, with a one-directional dependency flow.

```
        historical store            market data feed          data/
                |                          |
                +------------+-------------+
                             | bars
                       features                                features/
                             |
                     strategy engine                           strategy/
              (pure: bars + features -> OrderIntent)
                             |
                             | OrderIntent (inert data)
                             v
                        risk engine                            risk/
                 hard limits, sizing, approval tokens
                             |
                             | RiskToken + Order
                             v
                      execution engine                         execution/
                             |
                             v
                   guarded broker adapter                      broker/
                 Simulated | Tradovate demo
                             |
                             v
     journal (SQLite) <-> monitoring -> analytics -> dashboard
```

| module | responsibility | must NOT |
|---|---|---|
| `core` | shared value types, enums, clock | contain strategy- or broker-specific logic |
| `instruments` | contract spec registry | be bypassed by hardcoding MNQ elsewhere |
| `data` | parquet store, integrity validation, feeds, locked splits | expose future bars to a strategy |
| `features` | causal indicator library | use `shift(-n)`, `center=True`, or whole-series statistics |
| `strategy` | deterministic signal generation | import `broker` or `execution`, or hold any order-sending handle |
| `backtest` | event-driven simulation, realistic fills and costs | fill on the signal bar |
| `risk` | hard limits, sizing, approval tokens | be optional or bypassable |
| `execution` | intent to risk to order to broker, reconciliation, recovery | place an order without a valid risk token |
| `broker` | adapter protocol, Simulated and Tradovate demo | reach live endpoints without explicit opt-in *and* separate credentials |
| `journal` | append-only record of every decision and event | be the source of live truth (the broker is) |
| `analytics` | statistics from a trade set | recompute or alter trades |
| `monitoring` | heartbeat, staleness, error counting, kill switch | be able to open a position |
| `dashboard` | read-only view plus emergency STOP | contain business logic |

### 3.1 The non-bypass property (central safety requirement)

> *"The trading strategy must NEVER have direct unrestricted access to the broker. All
> orders must pass through the risk engine. The broker execution layer must reject any
> order violating these limits regardless of what the strategy requests."*

Enforced **structurally, in three independent layers**, not by convention:

1. **The strategy cannot express an order.** `Strategy.on_bar()` returns an `OrderIntent`
   — an inert dataclass. Strategies are constructed with no broker, no execution engine,
   and no journal. A static test asserts that nothing under `strategy/` imports `broker`
   or `execution`.

2. **The risk engine mints a single-use token.** `RiskEngine.evaluate(request)` returns an
   `Approval` carrying a `RiskToken` bound to a hash of the exact order fields (side,
   quantity, type, prices, instrument, timestamp) plus a nonce.

3. **The broker refuses tokenless orders.** Every adapter is wrapped in `GuardedBroker`,
   whose `place_order(order, token)` signature makes an unapproved order
   *unrepresentable*. It verifies the token — correct binding hash, unexpired, unspent,
   issued by this risk engine instance — and then **independently re-validates the order
   against the live limits**, so a stale approval issued before a limit breach still
   cannot execute.

Single-use tokens also give duplicate-order rejection for free.

---

## 4. Market data

**Historical.** Parquet store under `data_cache/`, addressed by `(instrument, timeframe)`.
Every load runs integrity validation: monotonic non-duplicate timestamps, OHLC consistency
(`low <= open, close <= high`), non-negative volume, and a bad-tick guard on bar range as a
fraction of price. Import is idempotent and writes a manifest recording row counts, date
range, and a content hash.

Available today: 898,186 NQ 1-minute RTH bars, 2017-06-30 to 2026-07-29 (2,341 sessions),
and 788,370 15-second bars, 2024-08-05 to 2026-08-03. They are represented as front-month,
volume-rolled, US/Eastern archives, but the Parquet metadata does not embed vendor,
contract-map, correction, or roll provenance. Content hashes can make experiments
reproducible; they cannot independently establish source authenticity.

**Live / replay.** A `MarketDataFeed` protocol emits **closed bars only**. An incomplete
bar is never delivered to a strategy — the single most common source of accidental
look-ahead in intraday systems. Two implementations: `ReplayFeed` (drives paper trading
from historical bars, optionally faster than real time) and a live feed behind the broker
adapter.

**Staleness.** The feed timestamps every bar on arrival. Monitoring raises `STALE_DATA`
when the gap since the last bar exceeds a configured multiple of the bar interval; the
execution engine refuses new entries while stale and flattens beyond a hard threshold.

---

## 5. Strategy framework

Every strategy declares, as data, on a `StrategySpec`:

- `entry_conditions` — named boolean predicates over the causal feature frame
- `invalidation_conditions` — what makes an open position's premise false
- `stop_loss` — rule (ATR multiple, structural level, or fixed points)
- `profit_target` — rule (R multiple, structural level, or fixed points)
- `trading_hours` — earliest entry, latest entry, force-flat time
- `filters` — regime, volatility and volume gates
- `max_trades_per_session` — hard integer cap

**No discretionary language may appear in execution logic.** Any term like "looks bullish"
must be translated into a named, testable predicate before it can be evaluated. The spec
object is serialisable, so the exact rule set behind any trade is recoverable from the
journal.

Families implemented in v1 — each deterministic, each a hypothesis to be *measured*, not
asserted:

1. `orb_breakout` — opening-range breakout
2. `sr_rejection` — support/resistance rejection
3. `sr_breakout_retest` — level breakout followed by a retest that holds
4. `vwap_reversion` — fade a stretched deviation from session VWAP
5. `vwap_continuation` — trade with trend on a VWAP touch-and-hold
6. `trend_pullback` — trend-following pullback entry

---

## 6. Backtesting

| hazard | mitigation |
|---|---|
| look-ahead bias | strategy sees `bars[:i+1]` only; a guard test poisons future rows with NaN and asserts identical results |
| future-data leakage | features computed causally; a test recomputes on truncated data and asserts prefix-equality |
| incomplete candles | feed emits closed bars only; a test asserts a partial bar is never delivered |
| unrealistic fills | a signal on bar *i* fills at bar *i+1* open plus slippage; gaps fill at the open, not the level |
| stop/target ambiguity | when one bar spans both, the **stop** is assumed hit first (pessimistic) |
| ignored commissions | round-trip commission charged per contract per fill |
| ignored slippage | configurable ticks per side, plus a stress multiplier |

**Reported statistics (minimum):** trade count, win rate, average winner, average loser,
expectancy, profit factor, maximum drawdown, average R multiple, Sharpe (where the trade
count supports it), max consecutive winners, max consecutive losers, performance by hour,
performance by weekday, long vs short breakdown, equity curve.

**Splits.** `DEV < 2023-01-01 <= VALIDATION < 2025-01-01 <= HOLDOUT`, locked as constants
and imported everywhere. DEV deliberately contains the 2018 Q4 selloff, the 2020 crash and
the 2022 bear. Parameters are never optimised on HOLDOUT. If a HOLDOUT number ever informs
a design decision, the holdout is spent and must be declared so.

**Multiple testing.** The report states how many configurations were evaluated and applies
a selection bar scaled to that count. A t-statistic quoted without its search count is not
evidence.

---

## 7. Risk engine

Configurable hard limits, all enforced in code and all covered by tests:

| limit | key |
|---|---|
| max dollars risked per trade | `max_risk_per_trade_usd` |
| max daily loss | `max_daily_loss_usd` (and an R-multiple twin) |
| max trades per day | `max_trades_per_day` |
| max position size | `max_contracts` |
| max consecutive losses | `max_consecutive_losses` plus cooldown |
| no duplicate orders | single-use tokens plus intent fingerprinting |
| no position outside allowed hours | session windows, force-flatten |
| emergency kill switch | flag file plus dashboard button; flattens and refuses new entries until a human clears it |

Position sizing is volatility-targeted: risk budget divided by (stop distance x
multiplier), capped at `max_contracts`, floored at `min_contracts` — below which the trade
is skipped rather than silently oversized. Size throttles as the trailing-drawdown cushion
shrinks.

Every rejection is journalled with a machine-readable reason code.

---

## 8. Broker integration

`BrokerAdapter` protocol: `connect`, `disconnect`, `place_order`, `cancel_order`,
`get_orders`, `get_positions`, `get_account`, `poll_events`.

**`SimulatedBroker`** — the default, used by tests and by paper trading. Local,
deterministic, no credentials. Models order acceptance and rejection, latency, partial
fills, stop and limit triggering against subsequent bars, and connection loss.

**`TradovateDemoBroker`** — real HTTP against Tradovate's demo environment.

- REST base `https://demo.tradovateapi.com/v1`
- Auth `POST /auth/accesstokenrequest` returns `accessToken` with a ~90-minute lifetime;
  renew via `/auth/renewaccesstoken` ahead of expiry rather than re-authenticating —
  concurrent sessions are capped and re-auth storms cause 4xx/5xx.
- Orders `POST /order/placeorder` with `accountSpec`, `accountId`, `action` (`Buy`/`Sell`),
  `symbol`, `orderQty`, `orderType` (`Market`, `Limit`, `Stop`, `StopLimit`, ...), and
  **`isAutomated: true`** — required by exchange policy for any order not physically
  triggered by a human. Cancel via `POST /order/cancelorder`.
- State `GET /order/list`, `/position/list`, `/account/list`.
- WebSocket `wss://demo.tradovateapi.com/v1/websocket`, frame-typed (`o` open,
  `h` heartbeat, `a` data array, `c` close); authorize after the open frame.

**Live trading is not implemented and not reachable.** The live environment branch raises
`LiveTradingDisabled` unconditionally; there is no boolean or environment-variable escape
hatch. Only demo credential names are documented in `.env.example`, `.env` is gitignored,
and tests assert that credential-shaped literals are absent from source. The demo adapter
is an integration seam, not a Tradeify Evaluation or Sim-Funded route.

---

## 9. Observability

Every strategy decision must be reproducible from the record alone. For each evaluated bar
the system can log: timestamp, market state (OHLCV), the indicator and level values the
rules actually read, the signal or the rejection with its reason code, the order request,
the broker response, fills, stop and target, resulting P&L, and any error.

Storage is a SQLite database, append-only in spirit, with tables for `signals`,
`rejections`, `orders`, `fills`, `trades`, `equity`, and `events`. Structured JSON logs go
to `logs/` alongside it. The journal is the audit record; a future approved broker remains
the source of truth for position state, and the two are reconciled on every restart.

---

## 10. Dashboard

Local web dashboard, Python standard library only (no web framework dependency), served on
`127.0.0.1`. Shows the selected Stage 0–4 mode, prop balance/high-water mark/failure floor,
daily and remaining risk, consistency/payout progress, bot lock state, current trade,
strategy statistics, and rejected signals. It includes a prominent **STOP TRADING** button
that trips the kill switch: flatten and refuse new entries until a human clears the flag.

---

## 11. Testing

Automated coverage required for strategy logic, every risk limit, position sizing, stop
placement, target placement, API failures, WebSocket disconnects, duplicate orders,
partial fills, stale market data, and restart recovery — plus the look-ahead and
non-bypass guards above.

"It compiles" and "it ran without an exception" are not completion criteria. A milestone is
done when its behaviour is asserted by a test that fails if the behaviour regresses.

---

## 12. Prior findings this system must not forget

Carried forward from earlier work in this repository, because they set the prior for
anything this system measures:

- Thirteen intraday NQ strategy families, well over a hundred configurations, 2017-2026:
  **all failed out-of-sample.**
- On 5-minute bars, eight of nine families had **no gross edge before costs** — the failure
  was upstream of the cost hurdle, which is a worse result than being killed by fees.
- The one apparent survivor (a narrow-open 30-minute ORB) drew 140% of its net profit from
  2022 alone, and the "it is really a volatility regime" rescue hypothesis was falsified:
  the same rule lost money on 2020's high-volatility days.
- The published intraday-momentum effect (Gao/Han/Li/Zhou, JFE 2018) does not replicate on
  NQ futures 2017-2022.
- Cost as a share of available movement: ~20% at a 1-minute hold, ~3.9% at 5 minutes,
  ~1.2% at 60 minutes, under 1% daily.
- Independent corroboration: arXiv:2605.04004 tested 14 signal families on MNQ 5-minute
  data over 947 sessions and found none met deployment criteria.

The correct posture: build the apparatus well, measure honestly, and expect the answer to
be "no edge" unless the evidence survives the search count that produced it.

---

## 13. Definition of done (v1)

| # | criterion | status |
|---|---|---|
| 1 | Historical data can be imported | implemented and tested |
| 2 | A strategy can be expressed deterministically | six families implemented and tested |
| 3 | It can be backtested | implemented; prop-gated integration verification in progress |
| 4 | Statistics are generated automatically | implemented and tested |
| 5 | Risk rules cannot be bypassed by the strategy | three-layer path and broker guard implemented and tested |
| 6 | The system can connect to a demo broker | demo adapter implemented/tested; no Tradeify route approved |
| 7 | It can paper trade automatically | pending |
| 8 | Every action is logged | partial; runtime wiring of complete three-layer traces is pending |
| 9 | Results appear on a web dashboard | implemented and tested |
| 10 | Automated tests pass | implemented; re-run after every integration milestone |
| 11 | Setup instructions work from a clean machine | pending |
| 12 | Architecture and remaining limitations are documented | implemented |

---

## 14. Survival-first 50K prop mission (v2)

The v1 measurement and non-bypass apparatus remains required, but it is now one layer of
a larger mission: operate a boring, measurable intraday MNQ system whose primary objective
is account survival under a versioned 50K prop-firm profile. MES is the secondary research
instrument. Production must never contain discretionary or generative-AI entries.

Priority is fixed:

1. protect the prop account;
2. preserve consistency and measurable positive expectancy;
3. pass evaluation and preserve payout eligibility;
4. raw profit.

The initial official profiles are current Tradeify Growth, Select, and Lightning 50K
accounts. Each phase has its own rule set. A generic percentage drawdown must never be
reported as Tradeify behaviour.

## 15. Three independent order gates

Every entry must produce an audit trace through all three gates. Any missing or uncertain
fact rejects the order.

1. **Strategy gate:** deterministic setup evidence, instrument, causal data, freshness,
   entry/stop/target geometry, expected R:R, tick alignment, and strategy session.
2. **Personal gate:** at most $200 planned worst-case loss including costs, one trade per
   Tradeify session, $200 daily strategy loss, one position, no averaging down, no
   martingale, no post-loss size increase, and no stop removal/widening.
3. **Prop-firm gate:** authoritative account/net-liq state, phase-specific EOD trailing
   floor and lock, remaining drawdown, dollar safety buffer, firm daily limit, aggregate
   contracts, consistency/payout implications, session/holiday deadline, compliance,
   profile freshness, and verified execution route.

The token/guard/broker boundary from §3.1 remains the structural **non-bypass control**; it
does not substitute for these three semantic gates.

The broker-side check must use a fresh authoritative account/positions/orders snapshot and
reserve risk atomically. A strategy or caller-supplied `purpose` string cannot self-declare
an exposure-increasing order to be an exit.

## 16. Prop account state and safety buffer

Continuously expose and journal:

- realized balance and real-time net liquidation;
- highest completed EOD balance;
- current official failure floor and remaining official drawdown;
- internal stop threshold (`official floor + configured dollar buffer`);
- realized/unrealized/daily P&L and remaining personal/firm daily risk;
- current and reserved aggregate exposure;
- evaluation target/consistency/pass state;
- payout cycle, eligibility, buffer, hold-duration test, and estimated request cap;
- explicit bot state and lock reason.

An entry is rejected when its conservative post-stop equity, including adverse entry/exit
slippage and fees, would reach or cross the internal stop threshold. The firm's floor is
the last defence, never the strategy stop.

## 17. Prop account simulation and robustness

Every prop simulation begins from a fresh nominal 50K account and uses a named, hashed
profile/phase. It applies the actual EOD high-water methodology, real-time breach test,
funded lock behaviour, soft daily pause, scaling, consistency, pass, and payout rules.

Monte Carlo operates on session blocks so intraday dependence is not destroyed. Reports
must include pass rate, hard-failure rate, internal-lock rate, median days to pass, average
and tail drawdown, first-payout probability, expected lifetime, expected account profit,
and losing-streak survival. Strategy selection prioritizes survival and payout probability,
not the best ordered backtest path.

## 18. Deployment stages and production immutability

| stage | name | automatic transition permitted? |
|---|---|---|
| 0 | backtest | yes, within research controls |
| 1 | market replay | no transition beyond completion |
| 2 | paper/demo | no transition beyond completion |
| 3 | prop evaluation | **never**; explicit human-approved manifest required |
| 4 | funded/sim-funded/live | **never**; separate explicit approval required |

A Stage 3/4 manifest binds account ID, firm/profile/phase and source snapshot, code commit,
strategy/config hashes, personal policy, approved execution route, and operator
attestations. Environment overrides may tighten risk but may not weaken or swap a pinned
production artifact.

Tradeify's current rules say Tradovate API access is unavailable for Evaluation and Sim
Funded accounts. The demo adapter in this repository is not a Stage 3 route. Evaluation
automation remains disabled until an approved platform-native route is verified.

Changes follow: idea → backtest → validation → untouched OOS → paper → production
comparison → human approval → production. A funded configuration is immutable until that
path and approval are complete.

## 19. Periodic verification and reporting

Daily rule verification checks official firm sources and broker/API dependencies. An
unchanged result records the date. Any source change emits an alert and proposed profile
diff; it never silently mutates Stage 3/4 behaviour.

Every trading session produces a report containing trades, P&L/R, setup/execution,
slippage/fees, balance/floor distance, payout progress, signal and rejection counts,
rejection reasons, and rule violations. It compares the day with historical distributions
and flags abnormal slippage, frequency, loss, or signal behaviour.

Weekly research analyses accepted and rejected trades, expectancy, setup/time/direction,
stops, slippage, loss sequences, and account-survival probability. At most one or two
changes advance as isolated experiments; none modifies production automatically.

## 20. Mission definition of done

In addition to §13, the mission is not deployment-ready until:

- official current profiles are verified, source-linked, machine-readable, and selected by
  exact account cohort/phase;
- all three gates and the authoritative broker-side recheck pass boundary/one-past tests;
- the personal $200 / one trade / $200 / one position defaults are pinned by tests;
- protective orders survive process failure and cannot be removed or widened by an
  unapproved path;
- risk state and reservations recover across restart before trading unlocks;
- fresh-account and Monte Carlo survival reports are reproducible;
- stage manifests and human approval prevent automatic Stage 3/4 activation;
- daily/weekly verification and reporting are operational; and
- no strategy has been promoted without evidence that survives search count, validation,
  holdout, paper comparison, and approval.
