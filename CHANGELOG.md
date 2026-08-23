# CHANGELOG

All notable, safety-relevant, and operator-visible changes are recorded here.

## Unreleased

### Added

- Durable repository operating instructions and decision/experiment ledgers for the
  survival-first $50K prop-firm mission.
- Standing requirement for dated, official-source prop profiles and fail-closed Stage 3/4
  rule verification.
- Personal-policy defaults now cap risk at $200 per trade, $200 per session, one trade,
  and one open position; every built-in strategy defaults to the same one-trade quota.
- Risk entry and broker-side approval checks now honor the internally tracked open
  position even when a caller omits its local position object.
- Strict, source-linked current Tradeify Growth, Select, and Lightning 50K profiles plus a
  documented conflict ledger and profile loader.
- Fail-closed official-page change detection that can append verification records and emit
  an alert/unapproved review proposal without mutating a live profile.
- Explicit Stage 0–4 authorization model. Stage 3/4 requires a human manifest binding
  account, profile/phase, code revision, strategy/config hashes, execution route, and bot
  ownership/exclusivity attestations; ordinary configuration refuses those stages.
- The shipped runtime now starts in Stage 0/BACKTEST with a $400 prop-floor safety buffer
  setting and the current Growth Evaluation profile selected for research.
- A deterministic three-layer coordinator now records Strategy, Personal, and Prop-Firm
  decisions independently; stale/unknown market, session, holiday, account, profile, or
  stage facts reject an entry.
- Immutable prop-account state transitions model EOD trailing drawdown, real-time breach,
  lock levels, daily-loss escalation, contract scaling, evaluation consistency, and
  phase-specific payout eligibility.
- Entry approvals are signed, typed, expiring, single-use, and bound to the exact order.
  The final guard obtains authoritative broker account/order/position state, serializes
  entry verification and submission, and reserves unresolved exposure.
- Protective stops and targets have separately validated permissions. Missing/refused
  initial protection trips the kill switch and requests an immediate flatten; stop
  replacement is replace-before-cancel and cannot widen risk.
- Historical backtesting now protects fills during the entry bar, charges both sides of
  commissions, aggregates partial exits, rejects unresolved terminal exposure, defaults
  to DEV, and requires explicit holdout access.
- Fresh-account phase and evaluation-to-funded prop simulators, session-block Monte Carlo,
  survival metrics, and auditable baseline/stress/policy scenarios.
- A complete prop-account/Bot/Current Trade/Statistics dashboard projection with a large
  `STOP TRADING` control, plus deterministic daily reporting and anomaly flags.
- A research-only operator CLI for typed configuration checks, profile checks, and
  append-only official-source verification. It imports no execution path and cannot place
  orders or authorize Stage 3/4.
- Production-pinned configuration loading that refuses identity/route overrides and risk
  weakening while accepting only explicitly safer environment changes.
- Architecture, limitation, rule-ledger, setup, decision, experiment, and project-memory
  documentation.

### Changed

- Personal defaults are now authoritative ceilings rather than strategy suggestions.
- Prop-firm policy is loaded by exact profile and phase; no strategy module contains firm
  constants.
- Research output distinguishes planned stop risk, actual commissions, and execution
  slippage, and records rejected candidates as evidence of the safety gates.

### Fixed

- The test suite could not be collected: `deployment/stages.py` referenced
  `dataclass_field` without importing it, and `broker/guarded.py` carried an orphaned
  block from an interrupted edit that raised `IndentationError`. The orphan was the
  pending-entry FILL transition and has been restored to `_observe_events`, where it
  belongs, rather than deleted.
- The execution engine still called the removed generic `GuardedBroker.cancel_order`. It
  now routes by purpose: entries through `cancel_entry`, protection through
  `retire_protective`, and stop/target swaps name the already-working replacement so the
  guard can authorise the retirement. A cancellation request is journalled as
  `CANCEL_REQUESTED`, not `CANCELLED`, because only a terminal broker event establishes
  that the venue actually cancelled.
- A restarted guard could not cancel the entry it had just recovered: the in-memory order
  ledger is empty after a restart, so the persisted reservation was unusable and the
  in-flight order could never be resolved. `cancel_entry` now also accepts an id matching
  the restored reservation, and still refuses every other unknown id.
- `requirements.txt` omitted `pyarrow` and `PyYAML`, which the runtime imports directly
  and which only resolved through pandas' and yfinance's transitive graphs.
- Stale test fixtures: deployment manifests used placeholder git revisions that the
  40-character commit rule now rejects, and three tests still called the removed
  `cancel_order`.

### Verified

- Clean-checkout setup works by both documented paths — `uv sync --locked --extra dev`
  and `pip install -r requirements.txt` — each giving 866 passed and 3 skipped. The skips
  are integrity checks against the gitignored nine-year NQ archive.
- A twelve-session Stage-1 `market-replay` over March 2018 completes broker-confirmed flat
  with no working orders, writing 4,680 equity marks, 4,787 events, 4,453 reason-coded
  rejections, 11 signals, 35 orders, 116 order events, 22 fills, 11 trades and 24
  per-session report files.

### Safety status

- Stage 3 and Stage 4 remain unavailable. No strategy has validated positive expectancy,
  unresolved official-rule conflicts remain, and no permitted Tradeify Evaluation/Sim
  Funded execution route has been approved.

### Existing implementation recovered

- Typed core models, instrument registry, causal data/features, six deterministic strategy
  families, guarded risk tokens, simulated broker with venue-side OCO behaviour, execution
  journal, backtesting/analytics, monitoring/dashboard, and a demo-only Tradovate adapter.
- Full baseline on 2026-08-22: 535 tests passed.
