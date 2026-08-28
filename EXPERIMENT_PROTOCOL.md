# Experiment Protocol — RSI(2) Long-Call Paper Trial

**Experiment ID:** `RSI2-OPT-2026-01`
**Status:** `DEVELOPMENT` — the official clock has **not** started
**Trading mode:** `PAPER` only. There is no live path in this package.

This document is the pre-registration. It is written before the trial begins so that the criteria
cannot be adjusted afterwards to fit whatever the results turn out to be. Anything not written
here is not part of the experiment.

---

## Hypothesis

Two claims are being tested, and they are deliberately separated because they can have different
answers:

**H1 (underlying).** The RSI(2) mean-reversion signal has positive expectancy on the underlying
ETF, traded at the next session's open.

**H2 (options wrapper).** The long-call implementation preserves enough of H1's edge — after
spread, theta, implied-volatility change and contract selection — to be worth trading.

H2 can fail while H1 succeeds. That is in fact the expected outcome given what the pilot showed,
and distinguishing the two is the main reason the trial runs two parallel ledgers.

## Signal

Unchanged from the validated research. **These rules are frozen for the duration.**

| | |
|---|---|
| Entry | RSI(2) < 5 **and** close > 200-day SMA |
| Exit | RSI(2) > 65, **or** close > 5-day SMA, **or** 10 sessions held |
| Direction | Long only |
| Evaluated on | Completed daily closes, adjusted prices |

## Universe

**`core_etf_v1` — SPY, QQQ, IWM, DIA, MDY.** Frozen.

Five liquid, diversified index ETFs, specified in `scanner/README.md` as the pre-committed alert
core. They are chosen by liquidity rules rather than by past performance.

The 63-name `strong` and 120-name `all` presets are **excluded from the formal trial**. They are
today's survivors, selected with hindsight; `scanner/README.md` line 169 says so directly. They
remain available for exploratory research and are labelled retrospective wherever they appear.

ETFs also sidestep single-name earnings risk and corporate actions, which keeps the first trial's
option mechanics simple.

## Timing — the rule that invalidated the pilot

```
session T close        signal computed from completed daily data
                       (never from a partial bar; runs before 16:00 ET are refused)
        |
session T+1 open       order submitted in the execution window
```

Exits follow the same rule: an exit signal at T's close submits at T+1's open. A fill may never
carry a price observed before the signal that caused it existed.

Sessions come from an explicit exchange calendar with holidays and half-days. `date + 1 day` is
never used. Friday signals execute Monday; a Wednesday signal before Thanksgiving executes on
Friday's half day.

## Execution

- Orders route to the broker's **paper** environment. Fills come from the broker.
- A displayed quote is **not** a fill. Unfilled, partially filled, rejected and cancelled orders
  are all first-class outcomes.
- Order type: limit, placed 2% through the ask.
- A signal that cannot be executed within 1 session expires and is recorded as
  `EXPIRED_UNFILLED`.

## Options selection

| gate | value |
|---|---|
| DTE window | 21–60 |
| Delta | 0.55–0.85 |
| Max spread | 10% of mid |
| Min open interest | 100 |
| Max quote age | 900s |
| Adjusted contracts | rejected |

Long calls only. **A debit spread is a different strategy** and would require its own experiment
version — it may not be substituted mid-trial.

## Risk

| limit | value |
|---|---|
| Starting equity | $50,000 |
| Max premium at risk per trade | **$200** |
| Max open positions | **1** |
| Max new trades per session | **1** |
| Max daily strategy loss | **$200** |
| Max per ticker / per sector | 1 / 1 |

```
premium_at_risk = price × contract_multiplier × quantity + transaction_costs
```

If one contract exceeds $200, the opportunity is **rejected and recorded** as
`REJECTED_RISK_LIMIT`. The limit is never raised to accommodate a trade. With SPY calls around
$5–8 this will reject frequently, and those rejections are data — they measure how much of the
strategy is reachable on a small account.

## Transaction costs

Commissions and fees as reported by the broker. No modelled cost substitutes for a reported one.

## Benchmarks

1. Buy-and-hold of each underlying ETF
2. Cash / risk-free baseline
3. **The underlying RSI(2) ledger** — the comparison that answers H2
4. The options ledger

## Duration

One year from `official_start_timestamp`, stamped once at activation and never rewritten.

## Prohibited during the trial

- changing RSI thresholds, hold period, universe, DTE, delta or risk limits
- switching the ML selector from shadow to active
- substituting spreads for long calls
- re-running history to "correct" a recorded fill
- adding names to the universe

Any of these ends this experiment and starts a new version. The config hash is checked on every
run; if it moves, the run fails closed.

## ML selector

`ML_ENABLED_FOR_EXECUTION = False`. It may predict and log. It may **not** influence selection.
Promoting it is a new strategy version, not a tuning step, and there is no automatic promotion
after any number of trades.

## Definition of success

**Operational success** (can this be judged at all?)

- signals generated after complete data, executed the next session, zero same-session fills
- broker reconciliation clean
- cash, positions and equity reconcile
- rejections and missed opportunities recorded
- no fabricated fills or reused stale marks

**Statistical evidence** (is there an edge?)

Reported, not decided automatically:

- expectancy and median trade, with block-bootstrap confidence intervals
- number of *independent* signal periods, not raw trade count
- comparison against all four benchmarks
- sensitivity to cost and execution assumptions

Verdicts are `RESEARCH CRITERIA MET`, `RESEARCH CRITERIA NOT MET` or `INSUFFICIENT EVIDENCE`.

**There is no automatic GO LIVE.** The power analysis already shows one year is unlikely to
settle H2: at the option wrapper's signal-to-noise ratio (0.022 vs 0.204 for the underlying),
roughly 8,635 trades would be needed for t=2. `INSUFFICIENT EVIDENCE` is the most probable
honest outcome and is not a failure.

Any real-money decision requires separate human review outside this system.
