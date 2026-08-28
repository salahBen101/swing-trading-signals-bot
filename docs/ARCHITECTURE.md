# Futures-system architecture

Status: Stage 0 research. This document describes `src/tradebot`; the older end-of-day
equity scanner in `scanner/` is independent and unchanged.

## Safety boundary

The strategy never owns a broker handle and never chooses contract quantity. It emits an
immutable `OrderIntent`. An entry can reach an adapter only through all three risk gates
and a final broker guard:

```text
closed bars -> causal features -> deterministic strategy -> inert OrderIntent
                                                     |
                                    StrategyGate (setup, prices, R:R,
                                      session, signal/data freshness)
                                                     |
                                    Personal RiskEngine ($200 maximum,
                                      one trade, one position, daily lock)
                                                     |
                                    PropFirmRiskGate (dated profile,
                                      account floor/buffer, DLL, contracts,
                                      hours, stage and account state)
                                                     |
                                  exact signed, expiring, single-use approval
                                                     |
                                    GuardedBroker -> broker adapter
```

The order, purpose, quantity, prices, instrument, strategy identity, risk, protective stop,
and timestamps are cryptographically bound to the approval. The final guard refreshes the
outer gates, obtains broker account/order/position state, and serializes entry verification
with adapter submission before it spends the personal-risk token. Unknown, altered,
expired, replayed, stale, already-exposed, or outcome-unknown entries fail closed. Exit and
protective-stop authorizations have separate typed purposes so an entry-only rule cannot
trap exposure.

Quantity is derived from all-in bounded risk. A market intent is converted to a protective
limit no farther than the configured entry-gap allowance. Per-contract risk is measured
from that worst signed entry to the stop, then adds round-trip commission and stressed
exit slippage before flooring and capping quantity. STOP and STOP_LIMIT entries are not
accepted because their entry risk is unbounded. This is a conservative planning bound,
not a guarantee against a later market gap through the protective stop.

No module is a substitute for the others:

| layer | owns | does not own |
|---|---|---|
| strategy | deterministic setup and intended entry/stop/target | size, firm rules, broker |
| strategy gate | setup shape, R:R, tick alignment, freshness/session facts | account limits |
| personal risk | sizing and personal quotas/locks | Tradeify policy |
| prop risk | selected profile/phase and immutable account-state transitions | signals |
| execution | order/protection lifecycle and reconciliation | permission policy |
| broker guard | final approval verification and adapter boundary | strategy logic |

## Runtime modules

- `data/`: validates and stores timestamped futures bars; replay exposes only closed bars.
- `features/`: causal indicators and price levels.
- `strategy/`: six deterministic strategy families behind a common interface.
- `risk/`: three-layer coordinator, personal limits, prop state/gate, sizing, kill switch,
  and signed approval tokens.
- `prop_firms/`: strict YAML model/loader and official-source change verification. Firm
  rules are data here, never constants in a strategy.
- `deployment/`: Stage 0-4 model and hash-bound human approval manifests.
- `broker/`: adapter protocol, local simulator, guard, and demo-only Tradovate seam.
- `execution/`: intent, fill, protective OCO, flattening, and recovery lifecycle.
- `journal/`: append-oriented SQLite records and dashboard/report queries, including
  rejected signals.
- `backtest/`: next-bar execution through the shared risk/execution path.
- `prop_simulator/`: fresh-account phase and Monte Carlo survival simulation.
- `analytics/`: performance metrics and research reports.
- `monitoring/` and `dashboard/`: health state, read API, account/bot/trade statistics,
  and the one mutating control: `STOP TRADING`.
- `app/`: operator entry points. These must preserve the stage gate.

## Prop-account state

`PropAccountState` is immutable. Reconciliation returns a new state containing balance,
real-time net liquidation, completed-session high-water mark, firm floor, internal floor
plus buffer, remaining cushion, daily realized/unrealized P&L, contracts, consistency,
evaluation progress, and payout eligibility.

For Tradeify's documented end-of-day trail, the high-water mark advances only at session
close while the resulting failure floor is enforced against real-time net liquidation.
Hard breaches latch. A new session can clear a soft DLL lock but cannot clear an account
breach. Payouts and phase changes are explicit transitions; neither occurs because a
strategy asks for one.

## Stage control

| stage | meaning | automatic reachability |
|---:|---|---|
| 0 | historical backtest | yes |
| 1 | market replay | yes |
| 2 | paper/demo | only after the adapter proves the complete recovery capability set; no shipped adapter does |
| 3 | prop evaluation | no |
| 4 | funded/sim-funded/live | no |

Every Stage 2+ broker guard requires a durable personal-risk ledger, a durable pending-entry
ledger, a separately completed bootstrap, canonical deployment-context verification,
exact account/broker/paper/route pins, native atomic protection/OCO, venue reduce-only or
close-position semantics, authoritative cancellation and terminal history, complete
firm-session execution replay, and account-owner fencing. Both shipped adapters advertise
at least one missing capability and are refused. A test-only adapter exercises the
positive constructor path without qualifying any real route.

Stage 3/4 startup additionally requires a short-lived signed human authorization that pins
the account, profile phase, profile hash, official-source snapshot hash, runtime
configuration, strategy artifact, actual code revision, execution route, and
ownership/exclusivity attestations. The repository must be clean; official-source
verification must be complete, current, and unchanged; and profiles with unresolved
ambiguity notes are refused. The supplied configuration is Stage 0 and the example
manifest is not an authorization.

## Research integrity

Data is divided into locked DEV, VALIDATION, and HOLDOUT ranges. DEV is the default. A
caller must explicitly unlock HOLDOUT or ALL, and doing so spends the final unbiased test.
Signals use history ending at the current closed bar, entries fill no earlier than the
next bar, and same-entry-bar stops are active after an opening fill. When an OHLC bar spans
both stop and target, the pessimistic stop is selected.

The signed entry price is a bounded IOC limit, not an indefinitely resting market proxy.
Strategy stop/target geometry and minimum reward/risk are recomputed at that executable
bound before approval and final verification. In the simulator, IOC has one eligible
opening print; a non-marketable order or unfilled remainder is terminally cancelled before
the bar's later range can affect it.

Backtest equity includes entry and exit costs. Partial fills aggregate into one completed
trade. Session end and end-of-data attempt a confirmed flatten and treat unresolved
position/order state as an error rather than silently finishing flat.

## Persistence and recovery

The SQLite journal records signals, accepted and rejected decisions, orders, fills,
trades, equity, and operational events. On restart, the execution layer compares local
state with broker positions and orders; broker state is authoritative and divergence is
journalled.

Personal risk and unresolved entry submissions also have separate, strictly versioned
durable ledgers. Each ledger uses exact bindings, atomic replace and fsync, a sibling
advisory writer lock, monotonic revision/CAS rules, and strict corruption rejection. The
personal ledger persists equity/session watermarks, trade quota, loss state, and active
entry/fill identity. Its initialization marker prevents a deleted file from being silently
re-created as a fresh account. The entry ledger persists account/route plus local and
broker order identities and cumulative fill evidence; a stale writer cannot erase stronger
fill evidence.

Bootstrap is an explicit, one-time operator workflow. A new runtime restores the state but
remains entry-locked until a fresh exact broker snapshot reconciles it. Persistence faults
latch new entries and the kill switch, while typed reduce-only flatten/exit paths remain
available. If a fill cannot be persisted, the guard attempts a snapshot-authorized full
flatten rather than exposing the fill as safely accounted.

These local ledgers are not an atomic transaction with each other or with a venue. A
cancellation can race a fill, and a flatten decision can race another account owner; even
a fresh sequence of account/order/position reads is not a versioned atomic snapshot.
Consequently Stage 2+ stays blocked until the adapter proves atomic protected entry,
authoritative full-session replay, and venue-backed account-owner fencing.

## Configuration ownership

- `config/tradebot.yaml`: personal risk, stage, data, costs, broker, and selected profile.
- `config/prop_firms/*.yaml`: dated program/phase rules with official URLs.
- `config/prop_firms/source_snapshots.yaml`: reviewed official-page fingerprints.
- `config/deployment.example.yaml`: schema example only; never an approval.

Unknown configuration keys fail validation. Environment overrides use
`TRADEBOT__SECTION__KEY`; they are convenient for research. A manifest-bound runner must
call the existing loader with `production_pinned=True`, where identity/route changes and
risk weakening are refused while only unambiguously tighter limits are accepted.
