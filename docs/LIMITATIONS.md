# Limitations and live-readiness blockers

Status checked: 2026-08-24. The last human-reviewed official-rule baseline remains
2026-08-22; the 2026-08-24 result is a change alert, not a replacement verification.

The repository is a Stage 0 research system. It is not approved for a Tradeify evaluation,
funded account, or unattended live order routing. Passing tests demonstrates specified
software behaviour, not a trading edge or permission to trade.

## Hard blockers

1. **No strategy has established a production edge.** The deterministic candidates exist,
   but selection still requires honest DEV research, frozen validation, a single final
   holdout, stressed costs, market replay, and paper comparison. No profitability claim is
   made.
   The present archive is an NQ price/volume proxy for MNQ and lacks embedded vendor and
   roll provenance; actual MNQ contract-level data is required before promotion.
2. **No approved Tradeify execution route is configured.** Tradeify's official material
   says Tradovate API access is unavailable for Evaluation and Sim Funded accounts. The
   included Tradovate adapter is a demo seam and hard-refuses its live host; it is not a
   route around that restriction.
3. **Profiles deliberately contain unresolved source conflicts.** The safest current
   deadlines are encoded for research, but any ambiguity blocks Stage 3/4 until a human
   resolves it with current official support/documentation.
4. **Every current profile is stale after an official-source change alert.** On
   2026-08-24, content changed on the Growth Evaluation, Lightning Funded Accounts,
   Select Evaluation, and Select funded/payout pages. Hash changes do not identify or
   approve a rule change. The 2026-08-22 profiles/baselines remain unchanged and cannot
   authorize Stage 2+ until human review produces a new versioned, tested profile.
5. **No shipped adapter meets the Stage 2 recovery contract.** Stage 2+ requires atomic
   protected entry, reduce-only close semantics, authoritative cancel/terminal state and
   complete firm-session execution replay, plus account-owner fencing. The simulator
   lacks durable session replay and owner fencing; the Tradovate seam lacks additional
   native protections. Constructor gates refuse both.
6. **No human deployment approval has been issued.** The fail-closed manifest mechanism
   exists, but the checked-in example carries no authority. Stage 3/4 additionally require
   current unchanged rule verification, exact artifact hashes, a clean repository, an
   approved account/phase and execution route, and immutable production artifacts.
7. **Paper and replay operation has not completed the required observation period.** No
   stage may be skipped automatically.

## Execution and broker limits

- The broker protocol has no atomic bracket or replace operation. A successful protective
  replacement can briefly leave two same-OCO stops, and a cancel failure can leave a
  duplicate working order. The engine keeps the older stop until the replacement is
  accepted and flattens on missing initial protection, but venue atomicity is unavailable.
- Cancellation and emergency flattening retain a time-of-check/time-of-use race. An entry
  can fill after its snapshot but before cancellation, and another owner can change
  exposure between a flatten snapshot and submission. Durable reservations preserve the
  uncertainty; recovery confirmation now requires successful full guarded reconciliation
  followed by two order-to-position flat/no-working-order passes, and the simulator caps
  reductions through flat. Those sequential reads still cannot exclude a second owner
  submitting and filling entirely between reads. Only venue-native atomic protection,
  reduce-only semantics, and account-owner fencing close this class of race for an
  external route, so Stage 2 remains blocked.
- Emergency flatten submission cannot guarantee a fill while a broker or network is
  disconnected. This is an unavoidable external-system risk, not a reason to relax the
  fail-safe response.
- The local simulator models bar-level execution. It cannot recover tick ordering inside
  an OHLC bar, queue position, exchange throttling, limit-down behaviour, or every form of
  partial fill. Pessimistic ordering and explicit slippage are approximations.
- NQ and MNQ may track the same index closely, but they do not share order books, prints,
  or volume. NQ-derived volume filters and a fixed MNQ slippage assumption cannot validate
  MNQ fill quality. The current Parquet file is content-addressable but does not carry
  independently auditable vendor/contract-roll provenance.
- A real-time market-data subscription and exchange-calendar/holiday service are not yet
  production integrated. Unknown holiday state rejects entries; the ordinary research
  session is intentionally narrower than the firm's outer window.
- The final guard reads account, orders, and positions immediately before entry,
  serializes entry verification/submission, and stores a durable conservative
  pending-entry reservation. The reads remain sequential rather than a broker-versioned
  atomic snapshot. An outcome-unknown or incompletely replayed submission intentionally
  stays locked.
- Stage 2+ requires exact broker/account/paper/route pins and a canonical material-runtime
  context. This prevents accidental identity drift, but it cannot detect a second process
  trading the same account; venue-backed owner fencing is still absent.
- Reconciliation imports broker equity and visible positions/orders, while durable ledgers
  retain local daily state and cumulative fill evidence. No shipped adapter can yet replay
  an authoritative account-wide firm-session execution cursor with complete protective
  provenance. The personal-risk and pending-entry files are also separate fail-closed
  transactions, not one atomic commit with the broker.

## Risk-model limits

- `risk.drawdown` in `config/tradebot.yaml` is a legacy research circuit breaker, not the
  Tradeify rule. Exact firm logic comes only from the selected prop profile and
  `PropAccountState`.
- Commission schedules and exchange/regulatory fees can change. Configured costs are
  assumptions and must be refreshed and stress-tested before each research report.
- Slippage is a model input, not a bound. Sizing includes the worst permitted entry limit,
  an eight-tick protective-stop gap reserve, round-trip commission, and stressed exit
  slippage, and rejects all-in planned risk above $200. A realized loss above its approved
  pro-rata envelope latches the kill switch, but that is detection rather than prevention.
  A larger gap through the protective stop, changed fees, or worse liquidity can still
  lose more than $200; no stop order can mathematically guarantee a maximum realized loss.
- Firm dashboards may calculate balance, consistency, winning days, or contract scaling
  differently during corrections. The official profile must be reverified and the broker
  account reconciled before relying on local status.
- Only one instrument/strategy/position is intended per runtime. Cross-account correlated
  exposure and firm-wide limits cannot be inferred from one broker connection.

## Prop-simulation limits

- Monte Carlo resamples historical session blocks; it does not create new market regimes
  and cannot prove future survival.
- Results inherit every bias in the source trades, costs, and sample period. A high pass
  rate on optimized trades is not independent evidence.
- Payout probability is policy-conditional. It is invalid after an unreviewed firm-rule
  change, and it excludes discretionary firm enforcement not expressible in the profile.
- Parameter perturbation and stressed execution should be reported beside the baseline;
  a single best backtest is never a deployment criterion.

## Security and operations

- The dashboard binds to loopback by default and has no authentication. Do not expose it
  to another host or the public network. `STOP TRADING` only reduces permission.
- Risk-token privacy is enforced by API design and signatures, but Python attribute
  privacy cannot defend against hostile arbitrary code already running in the process.
- Secrets come from environment variables. They must never be committed, written to the
  journal, or embedded in approval manifests.
- Automated daily rule checks compare official text fingerprints. Anti-bot challenges,
  unavailable pages, missing snapshots, or changed text produce a failure/alert; they do
  not authorize an automatic rule or production change.

## Evidence still required

Before a human can even consider Stage 3, the repository must show:

- ~~complete green tests from a clean environment~~ — **done.** A fresh checkout passes by
  both documented install paths (`uv sync --locked --extra dev` and
  `pip install -r requirements.txt`): 866 passed, 3 skipped. The skips are integrity
  checks against the gitignored nine-year NQ archive, which a clean clone does not carry.
- ~~market-replay journals demonstrating rejected-signal capture, daily reports, and
  forced flattening~~ — **partly done.** A twelve-session Stage-1 replay finishes
  broker-confirmed flat with no working orders and writes a full audit trail (4,453
  reason-coded rejections, 11 trades, 24 per-session report files). Protective-recovery
  and restart paths are covered by tests rather than by a recorded live journal.
- a frozen strategy and configuration with causal DEV/VALIDATION/HOLDOUT evidence;
- normal and stressed prop-simulation reports with survival metrics;
- a paper (Stage 2) journal over a meaningful sample, which does not yet exist;
- current official rules with no unresolved conflict;
- a verified platform-native execution route that Tradeify permits for the selected
  account phase; and
- explicit human approval bound to the exact artifacts.

**None of this is evidence of an edge.** Every criterion above is about the apparatus
being trustworthy, not about the strategies being profitable. No strategy in this
repository has demonstrated positive out-of-sample expectancy, and the measured behaviour
so far is consistent with the project's prior: on a 2018 DEV sample the opening-range
family returned −$2,977 over 395 trades with costs consuming 97.5% of the gross move at a
4.8-minute average hold.
