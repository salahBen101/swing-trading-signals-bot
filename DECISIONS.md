# DECISIONS — durable engineering and policy choices

Newest decisions appear first. Each entry records the choice, why it was made, and what
would justify revisiting it.

## 2026-08-25 — D013: Repeated source changes preserve, rather than reset, the block

The next daily check still found all four August 24 page mismatches and added Tradeify's
shared Daily Loss Limit page, affecting every current profile. Repeated or expanded hash
differences do not make the current text self-approving. The dated profiles and reviewed
hashes remain untouched, append-only alerts preserve both observations, and Stage 2+
remains blocked.

Revisit only through human comparison of the complete official text, cohort resolution,
a newly versioned profile/baseline, tests, and explicit approval. Never roll a profile's
verification date forward merely because the latest fetch completed successfully.

## 2026-08-24 — D012: Entry lifetime and reward/risk are executable facts

Every approved entry is now a bounded IOC limit. The simulator gives it exactly one
eligible opening print: a non-marketable order expires without later wick or future-bar
fills, and an opening partial fill cancels the remainder. The Tradovate projection carries
the IOC instruction explicitly. The complete strategy geometry is repriced at the final
signed execution bound before both approval and verification, so a setup that is 2R only
at the signal reference cannot pass as 2R at the executable price. The generic Stage-0
runner independently enforces the same finite, tick-aligned, protective geometry.

This deliberately trades fill rate for bounded stale-order risk. Revisit resting-entry
lifetime only after it is explicitly signed into the approval, survives final executable
R:R checks, has deterministic cancellation/restart semantics, and is validated without
using the locked holdout.

## 2026-08-24 — D011: A changed official page invalidates the reviewed profile

The official-source checker detected new content hashes on four Tradeify pages: Growth
Evaluation, Lightning Funded Accounts, Select Evaluation, and Select funded/payout
policies. This is a change alert, not a new rule verification. The 2026-08-22 profiles and
source baselines remain deliberately unchanged and are stale for deployment; no rule is
inferred from a hash difference and no live behaviour changes automatically.

Revisit only after a human reviews the new official text, resolves every ambiguity,
creates and tests a new dated profile/baseline, and explicitly approves it. Until then,
all affected profiles fail closed for Stage 2+ and no live execution is enabled.

## 2026-08-24 — D010: Every shipped adapter is blocked from Stage 2+

Durable local files are necessary but cannot prove that the venue accepted a protected
entry exactly once or that a second process cannot trade the same account. Stage 2 and
later construction therefore requires all of: native atomic protection/OCO, venue-side
reduce-only or close-position semantics, authoritative cancellation and exact terminal
history, authoritative replay of the complete firm-session execution history, and an
account-owner fencing generation. The simulator and Tradovate demo adapter truthfully
advertise missing capabilities, so both are unconditionally refused at Stage 2+.

Revisit only when an adapter implements and tests the complete capability set against a
durable venue boundary. A test double may exercise the positive constructor path, but it
does not qualify a real adapter or authorize live trading.

## 2026-08-24 — D009: The $200 ceiling includes bounded entry movement and costs

Sizing now begins with the worst executable entry price, not the signal reference. A
market intent becomes a protective limit no farther than the configured entry-gap bound;
the risk per contract is bound-to-stop loss plus a configured protective-stop gap reserve,
round-trip commission, and stressed exit slippage. Quantity is floored from that all-in
amount and then capped. Unbounded STOP and STOP_LIMIT entries are rejected, a broker fill
outside the signed entry envelope latches the risk engine, and a realized loss above the
approved pro-rata all-in envelope trips the kill switch.

This protects planned risk under the signed execution assumptions. It is not a promise
that realized loss cannot exceed $200: a market gap through the protective stop, fee
changes, or worse-than-modelled exit liquidity can still exceed it. Revisit parameters
only through configuration validation and stressed evidence, never by choosing quantity
first.

## 2026-08-24 — D008: Personal risk and unresolved entries are durable fail-closed ledgers

Personal daily/session state and pending-entry state use separate versioned files with
strict schemas, exact account/route/context bindings, atomic replace plus fsync, advisory
writer locks, monotonic revisions, and compare-and-swap transitions. A one-time
initialization marker prevents silent re-bootstrap after deletion. Restart remains locked
until a fresh broker snapshot reconciles state, and malformed, missing, stale-writer, or
ambiguous fill evidence preserves the reservation and blocks another entry. Durable
failures latch entries while retaining narrowly validated risk-reducing actions.

The two ledgers are not one cross-system transaction and cannot replace authoritative
venue replay or ownership fencing. Revisit only to strengthen them or replace them with a
transactional store that preserves the same bindings, monotonic fill evidence, and
fail-closed recovery.

## 2026-08-22 — D007: Scenario results are an audit ledger, not an optimizer

Prop-account Monte Carlo resamples complete trading sessions so intraday signal clusters,
zero-signal days, and the one-trade policy remain intact. Every run starts a fresh account
and reports baseline, stressed execution costs, and separately perturbed safety policies.
The scenario runner records inputs and outputs but deliberately does not rank or select a
winner.

Revisit the sampling unit only with evidence that a different block construction preserves
the dependence relevant to prop-account failure. Never let a scenario report promote a
production configuration automatically.

## 2026-08-22 — D006: Production configuration may only become stricter at runtime

Research configuration can use environment overrides. A manifest-bound runner must load
configuration in production-pinned mode: account/profile/phase, stage, route, strategy,
instrument, and artifact identity cannot change, and runtime overrides may only tighten
unambiguous risk, session, cost, or data-safety values. Ambiguous changes fail closed.

Revisit only to reduce the runtime override surface further.

## 2026-08-22 — D005: Entry approval is exact, typed, and broker-authoritative

Approvals bind the complete canonical order and authorization purpose to a signed,
expiring, single-use token. Immediately before entry submission, the guard reads broker
orders, positions, and account state, serializes snapshot → verification → submission, and
holds a worst-case pending-entry reservation. Exit, flatten, stop, and target permissions
use separately validated authorization kinds so an entry rule cannot trap exposure.

The current reservation is process-local and broker snapshots are sequential reads. These
are explicit Stage 3/4 blockers until durable recovery and route-specific atomicity are
proven; they are not reasons to relax the gate.

## 2026-08-22 — D004: Current official evidence may still be deployment-blocking

Official Tradeify pages were reviewed and fingerprinted on 2026-08-22. The machine-readable
profiles encode the conservative interpretation when official pages conflict, but retain
the conflict as an ambiguity note. A successful content-hash verification means the
reviewed pages are unchanged; it does not resolve their contradiction or authorize Stage
3/4. Tradeify's documented lack of Tradovate API access for Evaluation and Sim Funded is
also a hard route blocker.

Revisit only after current official documentation or written Tradeify support resolves
the ambiguity and an approved account-specific execution route is independently verified.

## 2026-08-22 — D003: Prop rules are versioned data, never strategy constants

Tradeify rules will be represented as dated, source-linked YAML profiles selected by
program and phase. The strategy receives no prop-firm identity. Risk and simulation consume
the selected profile. Missing, ambiguous, changed, or stale rules fail closed for Stage 3
and Stage 4.

Revisit only if a future broker provides a trustworthy machine-readable rule API; even
then, retain a pinned snapshot for reproducibility.

## 2026-08-22 — D002: Survival metrics outrank raw backtest profit

Strategy comparison must report fresh-account pass rate, failure rate, median days to
pass, drawdown, payout probability, expected account lifetime/profit, and losing-streak
survival. Net profit alone cannot select a production candidate.

Revisit only if the target ceases to be a prop-firm account.

## 2026-08-22 — D001: One guarded execution path

Backtest, replay, paper, and any future approved execution share strategy, sizing, risk,
execution, and journal components. Strategies emit inert `OrderIntent` values; a guarded
broker accepts only current, single-use risk approvals.

Revisit only to make the boundary stricter, never to add a bypass.
