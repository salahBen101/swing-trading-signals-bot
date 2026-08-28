# CHANGELOG

All notable, safety-relevant, and operator-visible changes are recorded here.

## Unreleased

### Added

- Versioned durable personal-risk state and pending-entry reservations with strict
  account/route/context bindings, atomic fsync/replace, advisory writer locks, monotonic
  fill evidence, revisions, and compare-and-swap updates. Personal risk also uses a
  one-time bootstrap marker so deleting its state cannot create a fresh account history.
- Canonical deployment-context and exact broker identity pins are now part of durable risk
  policy. Restarted entries remain locked until fresh broker reconciliation succeeds.
- Stage 2+ broker capability certification now requires atomic native protection/OCO,
  reduce-only/close-position semantics, authoritative cancellation and terminal history,
  authoritative firm-session execution replay, and account-owner fencing. No shipped
  adapter currently qualifies.
- All-in entry sizing uses the worst signed entry bound plus an eight-tick protective-stop
  gap reserve, round-trip commission, and stressed exit slippage before deriving quantity.
  Market intents become bounded IOC limits; unbounded STOP/STOP_LIMIT entries, fills
  beyond the signed entry envelope, and realized loss beyond the approved pro-rata
  envelope fail closed and latch the kill switch.
- Strategy geometry and minimum reward/risk are repriced at the final signed executable
  entry before approval and again before submission. The generic Stage-0 runner applies
  the same finite, tick-aligned, protective target/stop checks independently.
- Snapshot-authorized emergency flattening for recovered broker exposure, monotonic
  cumulative-fill reconciliation, and simulator-side reduction caps that cannot reverse a
  position through flat.
- Short-lived signed deployment authorization now binds the actual clean code revision,
  artifacts, runtime configuration, profile/source snapshot, account, phase, and exact
  execution route through Stage 3/4 startup.
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

- The simulator now gives IOC entries one eligible opening print only. A non-marketable
  opening cancels without a later intrabar-wick or future-bar fill, while an opening
  partial fill cancels its remainder. Tradovate orders carry the exact IOC instruction.
- Numeric configuration rejects booleans, non-finite values, fractional integers, and
  invalid timeouts/seeds instead of allowing language coercions to weaken policy.
- The 2026-08-25 official-source audit repeated the four prior mismatches and added the
  shared Daily Loss Limit page, for five distinct changed pages. New alerts were recorded;
  no reviewed hash, profile, or execution behavior was automatically changed.
- The 2026-08-24 official-source audit detected changed content on Growth Evaluation,
  Lightning Funded Accounts, Select Evaluation, and Select funded/payout policy pages.
  Alerts and empty unapproved proposals were emitted; the 2026-08-22 profiles and source
  baselines remain unchanged and are now stale for Stage 2+.
- Tradovate projections are exact-account and account-scoped, resolve the precise contract
  maturity/product instrument, preserve foreign identities for rejection, and treat
  malformed/corrected/cumulative fill evidence conservatively. The adapter still does not
  advertise Stage 2 recovery safety.
- Protective and flatten orders are risk-reducing in the simulator: flat, same-side, or
  oversized concurrent exits cancel or cap at the opposing exposure instead of opening a
  reverse position.
- Personal defaults are now authoritative ceilings rather than strategy suggestions.
- Prop-firm policy is loaded by exact profile and phase; no strategy module contains firm
  constants.
- Research output distinguishes planned stop risk, actual commissions, and execution
  slippage, and records rejected candidates as evidence of the safety gates.

### Fixed

- `STOP TRADING` now requests entry cancellation and consumes immediately available
  broker events before matching another bar; raced terminal events apply fill economics
  exactly once before cancellation/rejection handling, so a late fill cannot disappear.
- Durable pending-entry reservations bind immutable side, order type, limit, and stop
  facts and validate cumulative fill identity, time, price, quantity, and cost evidence
  before advancing personal risk state.
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

- Stage 2, Stage 3, and Stage 4 are unavailable with every shipped adapter. Atomic
  protected-entry semantics, authoritative session replay, and account-owner fencing are
  not implemented; the official profiles also became stale when four Tradeify pages
  changed on 2026-08-24. No strategy has validated positive expectancy, no permitted
  Tradeify Evaluation/Sim Funded route is approved, and no live execution was enabled.

### Existing implementation recovered

- Typed core models, instrument registry, causal data/features, six deterministic strategy
  families, guarded risk tokens, simulated broker with venue-side OCO behaviour, execution
  journal, backtesting/analytics, monitoring/dashboard, and a demo-only Tradovate adapter.
- Full baseline on 2026-08-22: 535 tests passed.
