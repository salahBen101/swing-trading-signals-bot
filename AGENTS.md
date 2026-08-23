# AGENTS — operating instructions for this repository

This repository is the durable memory for the $50K prop-firm trading-system mission.
Do not restart the project in a new session and do not treat chat history as the source of
truth.

## Required session startup

Before changing code:

1. Read this file completely.
2. Read `PROJECT_SPEC.md` completely.
3. Read `PLAN.md` completely.
4. Read `TODO.md` completely.
5. Read the newest entries in `DECISIONS.md`.
6. Inspect `git status --short --branch` and preserve unrelated user changes.
7. Inspect the implementation relevant to the highest-priority unfinished task.
8. Run the smallest useful baseline test, then continue that task.

## Priority order

1. Protect the prop account.
2. Preserve deterministic, measurable behaviour and auditability.
3. Establish positive expectancy with honest train/validation/holdout evidence.
4. Pass evaluations and preserve payout eligibility.
5. Raw profit is last.

An unprofitable strategy result is valid research. A flattering or unsafe backtest is not.

## Non-negotiable safety rules

- Strategy code may emit inert intents only. It may never import or call broker/execution
  code.
- Every entry passes strategy, personal-risk, and prop-firm gates. Any uncertainty is a
  rejection.
- Personal defaults are at most $200 risk per trade, one trade per Tradeify session,
  $200 daily strategy loss, and one open position.
- Never average down, martingale, increase size after a loss, revenge trade, remove a
  protective stop, or hold through the configured flatten deadline.
- Contract quantity is derived from risk and stop distance, then capped. It is never
  selected first.
- Stage 0 backtest, Stage 1 replay, and Stage 2 paper are the only automatically reachable
  stages. Stage 3 evaluation and Stage 4 funded/live require explicit human approval.
- Production strategy/configuration is immutable without the documented validation path
  and explicit human approval.
- Official prop rules live only in dated profiles under `config/prop_firms/` and in
  `docs/PROP_RULES.md`; strategy code must contain no firm-specific rules.
- A stale, changed, incomplete, or ambiguous official-rule profile blocks Stage 3/4.
- Never enable live trading, spend money, submit a payout, or perform another irreversible
  external action without explicit human approval.

## Engineering workflow

Use the loop: observe, plan, implement, test, measure, review, document, select next task.
After each milestone run its targeted tests; after a major milestone run the full suite.
Fix failures when they are in scope rather than merely reporting them.

Record:

- architectural and policy choices in `DECISIONS.md`;
- research runs and results in `EXPERIMENTS.md`;
- user-visible or safety-relevant changes in `CHANGELOG.md`;
- live task state in `TODO.md`.

Do not silently spend the locked out-of-sample dataset. If it influences a design choice,
record that the holdout is spent before making the choice.

