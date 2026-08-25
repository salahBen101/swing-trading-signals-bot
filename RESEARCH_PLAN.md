# RESEARCH_PLAN — is there a robust intraday MNQ edge?

Pre-registered 2026-08-22, against engine baseline `research-engine-v1` (commit `fa72f53`).

**This plan is written before any result is inspected.** The hypothesis list in §5 is fixed
at the moment of writing. Anything added later is recorded as an amendment with a date and
a reason, and counts toward the multiple-testing bar in §7. That is the whole point: a
t-statistic means nothing without the number of things that were tried to get it.

---

## 1. The question

Can a **robust, economically meaningful** intraday edge be found in MNQ/NQ that survives
realistic costs and Tradeify 50K prop-account constraints?

Not "can a profitable backtest be produced". Fourteen families have already produced
profitable-looking backtests here and none survived. The question is whether an edge exists
that is still there when you stop looking for it.

**A clear negative is a successful outcome.** If no robust edge exists in this data, the
correct deliverable is a well-evidenced statement to that effect, not a strategy.

---

## 2. What has already failed

Carried from `docs/` and prior research notes, because it sets the prior:

| body of work | families | verdict |
|---|---|---|
| intraday NQ programme | 13 | all failed out-of-sample |
| 5-minute screen (2017-2026, 2,341 sessions) | 9 | **8 had no gross edge before costs** |
| 1-minute order flow | 1 | real +3.65 tick gross effect, destroyed by a 3.2 tick fee |
| ORB narrow-open (the one survivor of DEV) | 1 | 140% of net profit from 2022 alone; volatility-regime rescue falsified |
| Gao/Han/Li/Zhou intraday momentum (JFE 2018) | 1 | does not replicate on NQ 2017-2022 |
| independent: arXiv:2605.04004, 14 families on MNQ 5-min | 14 | none met deployment criteria |

The single edge ever validated in this repository is RSI(2) long-only swing on equities —
a multi-day holding period, not intraday.

**The most important number in that table is "8 of 9 had no gross edge".** Those did not
fail at the cost hurdle. They failed upstream of it. That shapes the whole design below.

---

## 3. The economics that constrain everything

MNQ: $2.00 per index point, 0.25 tick, $0.50 per tick.

| component | per contract round trip |
|---|---|
| commission | $1.24 |
| slippage, 1 tick per side | $1.00 |
| **total** | **$2.24 = 1.12 index points = 4.48 ticks** |

**Every trade starts 1.12 points behind.** Consequences that are not negotiable:

- A rule must produce a *gross* edge well above 1.12 points to be worth anything net. The
  acceptance bar in §6 requires gross ≥ 4x round-trip cost, i.e. ≥ 4.5 points per trade.
- Turnover is expensive in a way that compounds: 3 trades a day for a year is ~750 round
  trips, or ~840 points of pure cost, against a $50,000 account.
- Prior measurement of cost as a share of available movement: ~20% at a 1-minute hold,
  ~3.9% at 5 minutes, ~1.2% at 60 minutes, under 1% daily. A 2018 DEV run of the shipped
  ORB family measured **97.5%** at a 4.8-minute average hold. Short holds are structurally
  close to unwinnable.

This is why §5 concentrates on hypotheses with a **structural reason to hold 20-120
minutes** rather than on faster signals.

---

## 4. Data

**New for this phase.** The full-Globex DBN archive
(`glbx-mdp3-20170630-20260729.ohlcv-1m.dbn`) contains 4,545,513 one-minute bars covering
**18:00-17:00 ET**, not merely RTH. Every prior family in §2 was tested on the RTH-only
parquet, so the overnight session has never been used here.

That matters: it makes the overnight-inventory, gap, and overnight-sweep hypotheses in §5
testable for the first time, and those are the ones with the clearest microstructural
rationale rather than the clearest indicator.

| dataset | coverage | use |
|---|---|---|
| ETH 1-minute, front-month, volume-rolled | 2017-06-29 → 2026-07-29 | overnight features, gap classification |
| RTH 5-minute (resampled) | same | primary trade timeframe |
| RTH 15-minute (resampled) | same | slower variants |

Front month is chosen per session by traded volume; outright contracts only (calendar
spreads price near 200 and would be catastrophic if averaged in). Roll dates are recorded
so no indicator window spans two instruments.

### Splits, locked

| split | period | sessions | role |
|---|---|---|---|
| DEV | < 2023-01-01 | ~1,420 | every idea, every parameter choice, every discarded experiment |
| VALIDATION | 2023-01-01 → 2024-12-31 | ~516 | sanity check before anything is believed |
| **HOLDOUT** | ≥ 2025-01-01 | ~405 | **untouched** |

DEV deliberately contains the 2018 Q4 selloff, the 2020 COVID crash and the 2022 bear, so
ideas are designed against difficulty rather than tuned on a rising tape.

**DEV has been mined before** (§2). A positive DEV result is therefore weak evidence in
this repository, and is treated as such.

---

## 5. Pre-registered hypotheses

One economic hypothesis at a time. Each states *why the market would behave this way*
before it states a rule. A hypothesis whose rationale is "these indicators crossed" is not
admissible.

### A. Opening dynamics

| id | hypothesis | economic rationale |
|---|---|---|
| A1 | **Opening drive** — a session that opens and moves directly away from the open on strong volume continues in that direction | Genuine imbalance at the open must be absorbed; a one-sided auction that does not immediately fail signals size that is not finished |
| A2 | **Opening range expansion** — sessions whose first 30 minutes are unusually *wide* relative to recent history trend further | Wide early range indicates disagreement being resolved directionally, versus balanced rotation |
| A3 | **Failed opening-range breakout** — a break of the opening range that closes back inside reverses to the opposite extreme | Trapped breakout traders must exit, supplying fuel in the opposite direction |
| A4 | **Opening reversal** — a session that opens far from the prior close and reverses in the first 30 minutes continues reversing | Opening gaps overshoot; the reversal is inventory being corrected |
| A5 | **Overnight inventory correction** — when the overnight session moves strongly one way, the RTH open reverts part of it | Overnight is thin and dealer inventory accumulates one-sided; RTH liquidity lets it be offloaded. *Only testable with the new ETH data.* |

### B. VWAP and session structure

| id | hypothesis | economic rationale |
|---|---|---|
| B1 | **VWAP continuation** — in a session holding one side of VWAP, pullbacks to VWAP resume | VWAP is the reference price for institutional execution; buyers defending it are real |
| B2 | **VWAP rejection** — price reaching VWAP from the wrong side and failing continues away | Failure to reclaim the session's average price marks one side as unwilling |
| B3 | **Stretched deviation reversion** — price ≥ N ATRs from VWAP reverts toward it | Execution algorithms benchmarked to VWAP mean-revert price toward it |
| B4 | **VWAP reclaim after failed auction** — price breaks a session extreme, fails, and reclaims VWAP | A failed auction plus a reclaim is a two-stage confirmation that the extreme was rejected |

### C. Prior-session structure

| id | hypothesis | economic rationale |
|---|---|---|
| C1 | **Previous-day high/low breakout** — clearing a prior-session extreme continues | Resting stops and breakout orders cluster at obvious prior levels |
| C2 | **Failed prior-extreme breakout** — clearing the level then closing back inside reverses | The stop cluster was the liquidity; once taken, there is no follow-through |
| C3 | **Previous close interaction** — behaviour around the prior settlement | Settlement is the reference for overnight P&L and option strikes |
| C4 | **Overnight high/low sweep and rejection** — RTH takes out the overnight extreme and immediately rejects | Overnight extremes are thin-liquidity prints; a sweep in RTH liquidity often marks exhaustion. *Only testable with the new ETH data.* |

### D. Trend and pullback

| id | hypothesis | economic rationale |
|---|---|---|
| D1 | **Strong-session pullback** — in a session with high directional efficiency, pullbacks resume | Persistent one-way order flow does not finish in one push |
| D2 | **Breakout → pullback → continuation** — the retest after a level break | Requires confirmation, trading fewer and better signals |
| D3 | **Volatility-adjusted trend continuation** — trend entries sized and filtered by realized volatility | Trend persistence is regime-dependent; the filter is the hypothesis |

### E. Regime and time conditioning

Applied to whichever of A-D show any gross signal, **not tested standalone**. Conditioning
is a second search dimension and inflates the multiple-testing count, so it is applied only
where there is something to condition.

| id | condition |
|---|---|
| E1 | high vs low trailing realized volatility |
| E2 | trend vs range day classification (efficiency ratio) |
| E3 | gap size and direction at the open |
| E4 | scheduled macro-event days (FOMC/CPI/NFP) vs ordinary |
| E5 | time of day |

**Total pre-registered primary hypotheses: 16 (A1-A5, B1-B4, C1-C4, D1-D3).**

---

## 6. Procedure per hypothesis

```
HYPOTHESIS → SIMPLE IMPLEMENTATION → DEV SCREEN (gross)
    → if no gross edge: REJECT and record. Costs are not applied; there is nothing to cost.
    → if gross edge: DEV NET → VALIDATION → ROBUSTNESS → PROP SIMULATION → ACCEPT/REJECT
```

**Stage 1 — gross screen (DEV).** Does the entry predict anything at all, before costs?
This is deliberately first, because 8 of 9 prior families died here and costing a rule with
no signal wastes effort and invites tinkering.

**Stage 2 — net (DEV).** Full commissions and 1-tick-per-side slippage.

**Stage 3 — validation.** Unchanged rules and parameters on 2023-2024. No refitting.

**Stage 4 — robustness.** All of:

- *Parameter neighbourhood*: neighbouring values must behave similarly, with smooth decay
  rather than a spike at the chosen value. A rule that needs a precise number is a rule
  fitted to noise.
- *Year by year*: no single year may supply the majority of net profit. This is the test
  that killed the previous best candidate.
- *Direction*: long and short examined separately; a long-only "edge" over a sample where
  the index tripled is beta wearing a costume.
- *Cost stress*: 2 ticks per side. An edge that dies here is not deployable.
- *Trade concentration*: results must not depend on a handful of extreme trades. Reported
  as the net P&L excluding the best and worst 5%.
- *Turnover*: cost as a share of gross must stay well below 50%.

**Stage 5 — prop simulation.** Tradeify 50K Evaluation and Sim-Funded, Monte Carlo over
resampled session blocks: pass rate, days to pass, max drawdown distribution, breach rate,
expected profit, losing-sequence length.

**Stage 6 — holdout.** Only for a rule that has already passed everything above, looked at
**once**. If a holdout number ever informs a change, the holdout is spent and every
subsequent result from it is worthless — that fact would be recorded in
`REJECTED_STRATEGIES.md` rather than hidden.

---

## 7. Acceptance criteria

A strategy graduates to paper trading only if **all** hold:

| criterion | threshold |
|---|---|
| gross edge | ≥ 4.5 points per trade (4x round-trip cost) |
| net expectancy | > 0 after commissions and 1-tick slippage |
| statistical bar | \|t\| ≥ the multiple-testing threshold for the cumulative search count |
| sample | ≥ 200 trades on DEV, ≥ 100 on validation |
| year concentration | no year > 60% of net profit; ≥ 60% of years positive |
| direction | both long and short non-negative, or a stated structural reason for one-sidedness |
| parameter stability | ≥ 70% of the neighbourhood within ±1 standard error of the chosen value |
| cost stress | still positive at 2 ticks per side |
| concentration | still positive with best/worst 5% of trades removed |
| turnover | cost < 50% of gross |
| prop survival | Tradeify 50K Evaluation pass rate ≥ 50%, breach rate ≤ 20% |
| holdout | positive, consistent with validation, inspected once |

**Multiple-testing bar.** Cumulative search count for this project = 14 prior families + 16
pre-registered here = 30 before any conditioning or amendment. A Bonferroni-style two-sided
correction at α = 0.05 puts the bar at **|t| ≥ 3.2**, and it rises as the count grows. The
running count is maintained in `STRATEGY_REGISTRY.md`.

---

## 8. Anti-overfitting rules, binding

1. Never optimise against holdout.
2. Never inspect holdout repeatedly while modifying a strategy.
3. One profitable year is not an edge.
4. A rule requiring precise parameter values is rejected.
5. Neighbouring parameters must behave similarly.
6. Commissions and realistic slippage always included.
7. Excessive turnover penalised explicitly.
8. Simple rules preferred; a rule needing many conditions is suspect.
9. Sample size requirements enforced, not waived.
10. An edge that dies under slightly worse costs is rejected.
11. **Every failed hypothesis recorded** in `REJECTED_STRATEGIES.md`. Negative results are
    the main product of this phase, not its waste.
12. No machine learning until the simple hypotheses are exhausted *and* a proper
    walk-forward framework exists. Not in this phase.

---

## 9. Deliverables

| file | contents |
|---|---|
| `RESEARCH_PLAN.md` | this document, plus dated amendments |
| `STRATEGY_REGISTRY.md` | every family tested, full metric set, running search count |
| `REJECTED_STRATEGIES.md` | every rejection with the specific reason it failed |

The full metric set recorded for every family: rationale, exact rules, data period, train /
validation / untouched out-of-sample periods, trade count, gross P&L, commissions,
slippage, net P&L, expectancy, profit factor, maximum drawdown, average R, median holding
time, performance by year, performance by regime, long vs short, parameter sensitivity, and
prop-account survival statistics.

---

## 10. What would make this phase a success

In order of likelihood:

1. **A clear, well-evidenced negative** — the overwhelmingly most likely outcome given §2,
   and a genuinely useful one. It ends an expensive search on evidence rather than fatigue.
2. **A weak effect that fails the deployment bar** but is honestly characterised, narrowing
   where any future search should point.
3. **A robust edge.** Possible, but the prior is against it, and nothing in this plan is
   permitted to manufacture one.

Failing to find an edge is not a failure of the research. Reporting one that is not there
would be.

---

## 11. Outcome (2026-08-22)

The screen ran on the full-Globex dataset built for this phase (3.19M ETH 1-minute bars →
178,620 RTH 5-minute bars, 2017-2026). Results in full: `STRATEGY_REGISTRY.md`; reasons per
family: `REJECTED_STRATEGIES.md`.

**No robust edge was found. All sixteen pre-registered hypotheses were rejected. The holdout
was not touched.**

- Only one family (A4, opening reversal) cleared the DEV gross cost hurdle, at t=+1.72 —
  below even the naive 1.96 bar. It then failed year-concentration (92% from 2022),
  direction (short-only), trade-concentration, and validation (sign flip to −1.80). It was
  2022 shorts.
- No family had both a DEV gross above the hurdle and a positive validation. Seven of sixteen
  had no positive gross at all.
- The overnight session — new to this project and the source of the best-motivated
  hypotheses (A5, C4) — yielded nothing. Those flipped sign on validation like the rest.

This reproduces, on a larger and newer dataset, the prior finding in §2. Running total across
the whole repository: **30 intraday NQ/MNQ families, none survived out-of-sample.**

Per §10, this is outcome (1): a clear, well-evidenced negative. It is the most likely result
given the prior and it is reported as the finding. No strategy graduates to paper trading. No
machine learning is attempted — rule 12 gates it on simple hypotheses being exhausted *and* a
walk-forward framework existing, and while the simple families are now well-exhausted, the
correct conclusion from a clean 30-for-30 negative is not "add model complexity" but "this
horizon and instrument do not carry a retail-expressible edge."

### If research continues

The evidence points away from more intraday rule families. Directions with a different prior,
in rough order of promise:

1. **Longer horizons.** The one edge ever validated in this repository (RSI(2) equities,
   t=8.70) holds for days, not minutes. Cost as a share of the move falls below 1% at a daily
   hold. Overnight and multi-day NQ effects are unexamined here.
2. **A genuinely different instrument or cross-instrument structure** (lead-lag, relative
   value) rather than a new indicator on the same NQ 5-minute bars.
3. **Accepting the negative** and redirecting the engineering — which is sound and
   well-tested — toward a strategy that has independent evidence, rather than continuing to
   search where 30 families have already failed.

---

## 12. Amendment 1 (2026-08-22): dynamic exits, volume profile, order flow

Prompted by a proposal to add bid/ask and volume-profile *confirmations* and to make exits
dynamic — cut losers early, trail winners (dynamic TP/SL). Recorded as an amendment because
§1 fixes the hypothesis list; this adds to the search count (now 33: 30 + these 3).

### Data constraint on bid/ask — untestable, not merely untested

The archive is OHLCV. There is **no quote/BBO data** over the research period. The only
order-flow data (`nq_orderflow_15s.parquet`, aggressor-side ticks) spans 2025-07 → 2026-02,
which is **entirely inside the HOLDOUT**, and is a trade-imbalance *proxy*, not true bid/ask.
A bid/ask confirmation therefore cannot be tested on DEV or validation at all, and testing it
would require spending the holdout — forbidden. Prior work (commit `c31f19e`) already tested
tick order flow and found no edge. **Bid/ask is not evaluated; the data does not exist where
the discipline allows it to be used.**

### The controlling theory

For a zero-edge (random-walk) entry, no exit rule produces positive net expectancy: a
trailing stop changes the *shape* of the win/loss distribution but not its mean, and costs
make the mean negative. So a dynamic exit can only add value where the entry carries
directional *persistence*. Every dynamic-exit result is therefore read against a
**null control**: the identical exit applied to random entries.

### Result — rejected

| test | finding |
|---|---|
| null control (DEV) | random entries + dynamic exit: mean net **−0.90 pt**, 95th pctile +1.41, best-of-40 +2.83 |
| null control (validation) | mean net −1.07 pt, 95th pctile +2.21 |
| momentum entries + dynamic exit (DEV) | A1, C1, D1, D2, D3 all **inside the null band** — indistinguishable from random entries with a good exit |
| A4 + dynamic exit | DEV net +1.87 (barely over null 95th), but **2022 = 142% of net, short-only**; validation **−3.47** |
| volume-profile confirmation | **degrades every entry** (C1 +0.40→−0.01, D1 +0.13→−1.80, A4 +1.87→−2.98) |

The dynamic exit has no intrinsic power (the null centres negative), and no entry's edge
survives being measured against it. Volume-profile confirmation, far from adding value,
removes exactly the trades that made A4 look good — proving there is no acceptance/
continuation structure, only 2022 falling. This is the theory's prediction, confirmed: you
cannot exit your way out of an entry that carries no signal.

Holdout still untouched. Reproduce with `scripts/research_dynamic.py`.
