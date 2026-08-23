# DECISIONS — durable engineering and policy choices

Newest decisions appear first. Each entry records the choice, why it was made, and what
would justify revisiting it.

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
