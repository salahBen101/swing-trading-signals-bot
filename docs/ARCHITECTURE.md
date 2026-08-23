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
| 2 | paper/demo | yes |
| 3 | prop evaluation | no |
| 4 | funded/sim-funded/live | no |

Stage 3/4 startup requires a human-created manifest that pins the account, profile phase,
profile hash, official-source snapshot hash, runtime configuration, strategy artifact,
code revision, execution route, and ownership/exclusivity attestations. The repository
must be clean; official-source verification must be complete, current, and unchanged; and
profiles with unresolved ambiguity notes are refused. The supplied configuration is Stage
0 and the example manifest is not an authorization.

## Research integrity

Data is divided into locked DEV, VALIDATION, and HOLDOUT ranges. DEV is the default. A
caller must explicitly unlock HOLDOUT or ALL, and doing so spends the final unbiased test.
Signals use history ending at the current closed bar, entries fill no earlier than the
next bar, and same-entry-bar stops are active after an opening fill. When an OHLC bar spans
both stop and target, the pessimistic stop is selected.

Backtest equity includes entry and exit costs. Partial fills aggregate into one completed
trade. Session end and end-of-data attempt a confirmed flatten and treat unresolved
position/order state as an error rather than silently finishing flat.

## Persistence and recovery

The SQLite journal records signals, accepted and rejected decisions, orders, fills,
trades, equity, and operational events. On restart, the execution layer compares local
state with broker positions and orders; broker state is authoritative and divergence is
journalled. Runtime risk/account persistence and production-grade transactional recovery
remain incomplete and are listed in `docs/LIMITATIONS.md`.

## Configuration ownership

- `config/tradebot.yaml`: personal risk, stage, data, costs, broker, and selected profile.
- `config/prop_firms/*.yaml`: dated program/phase rules with official URLs.
- `config/prop_firms/source_snapshots.yaml`: reviewed official-page fingerprints.
- `config/deployment.example.yaml`: schema example only; never an approval.

Unknown configuration keys fail validation. Environment overrides use
`TRADEBOT__SECTION__KEY`; they are convenient for research. A manifest-bound runner must
call the existing loader with `production_pinned=True`, where identity/route changes and
risk weakening are refused while only unambiguously tighter limits are accepted.
