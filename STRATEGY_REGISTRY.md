# STRATEGY_REGISTRY — every family tested

The complete record of the strategy research phase run against engine baseline
`research-engine-v1`. Rejections and their reasons are in `REJECTED_STRATEGIES.md`; the
design and rules are in `RESEARCH_PLAN.md`.

**Running search count (for the multiple-testing bar): 30.**
14 prior families + 16 pre-registered here. A two-sided Bonferroni correction at α = 0.05
across 30 tests puts the acceptance bar at **|t| ≥ 3.2**. Not one family below reaches it in
the right direction.

---

## Data and method

| item | value |
|---|---|
| instrument | NQ front-month (volume roll), MNQ economics applied ($2/pt, 0.25 tick) |
| source | Databento GLBX.MDP3 1-minute, full Globex 18:00-17:00 ET |
| **new this phase** | the overnight session — every prior family here was RTH-only |
| bars | 3,194,358 ETH 1-minute → 178,620 RTH 5-minute |
| period | 2017-07-20 → 2026-07-29, 2,327 sessions |
| DEV | < 2023-01-01, 1,406 sessions, 107,982 bars |
| VALIDATION | 2023-01-01 → 2024-12-31, 516 sessions, 39,579 bars |
| HOLDOUT | ≥ 2025-01-01 — **untouched; no candidate earned a look** |
| costs | $1.24 commission + 1 tick/side slippage = 1.12 pts round trip |
| fills | next-bar open; gaps at the open; stop-before-target when a bar spans both; one position at a time; flat at session end |
| exit (all) | 1.0 ATR stop, 2.0-2.5 ATR target, hold cap, no entry in the last 3 bars |

Every number below is out-of-costs on the stated split. The screening harness is guarded by
a poison test that reproduces byte-identical trades when all future bars are corrupted, run
against all 16 hypotheses (`tests/tradebot/test_research_harness.py`).

---

## Stage-1 gross screen (DEV) — does the entry predict anything before costs?

Sorted by gross points per trade. The cost hurdle (RESEARCH_PLAN §7) is gross ≥ 4.48 pts.

| id | family | n | gross pt | gross tk | net pt | t | win% | PF | avg R | hold | max DD $ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A4** opening_reversal | A | 361 | **+5.18** | +20.7 | +4.06 | +1.72 | 39.1 | 1.26 | +0.09 | 15m | 1,293 |
| C4 overnight_sweep | C | 1,386 | +2.13 | +8.5 | +1.01 | +0.96 | 35.9 | 1.07 | −0.01 | 15m | 1,794 |
| D1 strong_session_pullback | D | 515 | +1.56 | +6.2 | +0.44 | +0.33 | 38.1 | 1.04 | +0.00 | 20m | 1,283 |
| A1 opening_drive | A | 686 | +1.34 | +5.3 | +0.22 | +0.15 | 37.2 | 1.01 | −0.01 | 15m | 2,576 |
| D3 vol_adjusted_trend | D | 1,119 | +1.10 | +4.4 | −0.02 | −0.02 | 31.9 | 1.00 | −0.02 | 25m | 3,232 |
| C1 prior_extreme_break | C | 1,249 | +0.96 | +3.9 | −0.16 | −0.15 | 35.6 | 0.99 | −0.05 | 15m | 4,535 |
| D2 breakout_pullback | D | 1,907 | +0.52 | +2.1 | −0.60 | −0.79 | 35.3 | 0.96 | −0.08 | 20m | 4,198 |
| B1 vwap_continuation | B | 4,109 | +0.41 | +1.6 | −0.71 | −1.39 | 35.5 | 0.94 | −0.08 | 20m | 6,583 |
| B2 vwap_rejection | B | 3,105 | +0.34 | +1.4 | −0.78 | −1.31 | 35.0 | 0.94 | −0.10 | 20m | 5,537 |
| B4 vwap_reclaim | B | 1,304 | −0.00 | −0.0 | −1.12 | −1.18 | 36.2 | 0.92 | −0.05 | 20m | 5,885 |
| A5 overnight_inventory | A | 274 | −0.21 | −0.8 | −1.33 | −0.45 | 35.4 | 0.93 | −0.01 | 12m | 1,978 |
| A3 failed_orb | A | 1,625 | −0.90 | −3.6 | −2.02 | −2.17 | 34.0 | 0.87 | −0.09 | 20m | 6,676 |
| C2 failed_prior_break | C | 957 | −1.25 | −5.0 | −2.37 | −2.06 | 32.7 | 0.84 | −0.13 | 15m | 5,100 |
| B3 stretched_reversion | B | 4,945 | −1.65 | −6.6 | −2.77 | −6.75 | 31.5 | 0.78 | −0.20 | 20m | 27,448 |
| C3 prior_close_interaction | C | 908 | −3.24 | −13.0 | −4.36 | −3.74 | 32.7 | 0.73 | −0.15 | 15m | 8,247 |
| A2 range_expansion | A | 80 | −7.31 | −29.2 | −8.43 | −1.54 | 27.5 | 0.66 | −0.28 | 15m | 2,349 |

**Only A4 clears the gross cost hurdle**, at t=+1.72 — already below the naive 1.96 bar and
far below the 3.2 multiple-testing bar. Nine of sixteen have any positive gross at all; the
median family has no gross signal. This reproduces the prior "8 of 9 had no gross edge"
result on a wholly new, larger dataset that includes the overnight session.

---

## Stage-3 validation (2023-2024) — same rules, no refit

Only families with a positive DEV gross are meaningful to carry forward, but all sixteen are
shown so the sign flips are visible.

| id | DEV gross | VAL gross | VAL net | VAL n | VAL t | verdict |
|---|---:|---:|---:|---:|---:|---|
| A4 opening_reversal | +5.18 | **−1.80** | −2.92 | 126 | −0.65 | **sign flip** |
| C4 overnight_sweep | +2.13 | **−2.36** | −3.48 | 534 | −1.95 | **sign flip** |
| D1 strong_session_pullback | +1.56 | **−0.74** | −1.86 | 202 | −0.79 | **sign flip** |
| A1 opening_drive | +1.34 | +1.58 | +0.46 | 257 | +0.16 | positive but below hurdle, t≈0 |
| D3 vol_adjusted_trend | +1.10 | +0.07 | −1.05 | 312 | −0.35 | gone |
| C1 prior_extreme_break | +0.96 | +1.64 | +0.52 | 456 | +0.25 | positive but below hurdle, t≈0 |
| D2 breakout_pullback | +0.52 | +0.11 | −1.01 | 711 | −0.72 | gone |
| B1 vwap_continuation | +0.41 | **−0.56** | −1.68 | 1,583 | −1.80 | sign flip |
| B2 vwap_rejection | +0.34 | **−0.34** | −1.46 | 1,164 | −1.32 | sign flip |
| A2 range_expansion | −7.31 | +4.70 | +3.58 | 34 | +0.32 | flip the other way; n=34 noise |
| C2 failed_prior_break | −1.25 | +1.49 | +0.37 | 350 | +0.16 | flip; below hurdle, t≈0 |
| (negative on both) | | | | | | consistent failures — see rejections |

**No family has both a DEV gross above the cost hurdle and a positive validation.** Zero.
The one DEV hurdle-clearer (A4) is the largest sign flip on the board. The families that
stay positive on validation (A1, C1) sit at a fifth of the hurdle with t ≈ 0.2 — the
signature of noise, not edge.

---

## The one candidate that reached Stage 4: A4 opening reversal

A4 was the only family to clear the DEV gross hurdle, so it alone ran the robustness
battery. It failed three of the disqualifying tests — the same pattern that ended the
previous best candidate in this repository.

| robustness test | result | verdict |
|---|---|---|
| **year concentration** | 2022 alone = **+92%** of net profit; 3/6 years positive | **FAIL** (bar: ≤60%, ≥60% positive) |
| **direction** | long +1.27 pt vs short **+8.22** pt gross | **FAIL** — the edge is one-sided shorts |
| **trade concentration** | trimming best/worst 5% cuts net +4.06 → +1.52 pt (−63%) | **FAIL** — a few extreme trades carry it |
| cost stress (2 tk) | +4.06 → +3.56 pt | pass (irrelevant given the above) |
| validation | DEV +5.18 → VAL **−1.80** | **FAIL** — sign flip |

A4's profit is 2022 shorts. 2022 was a persistent downtrend; a rule that fades the opening
gap made money by being short in a down year, over a sample where the index otherwise rose.
That is beta in disguise, and it does not survive contact with 2023-2024. A4 does not
proceed to holdout.

---

## Prop-account survival

Not computed for any family. The Tradeify 50K simulation is gated on a candidate first
passing the DEV → validation → robustness pipeline (RESEARCH_PLAN §6, Stage 5). No family
reached it. Running a survival simulation on a rule with no validated edge would produce a
number with no meaning — the pass rate of a coin.

---

## Conclusion

Sixteen pre-registered intraday hypotheses, tested one economic claim at a time on the
largest and newest dataset available to this project — the first to include the overnight
session — produced **no robust edge**. The single family that cleared the gross cost hurdle
on DEV was 2022 shorts and inverted on validation.

Combined with prior work, the running tally is **30 intraday NQ/MNQ strategy families, none
survived out-of-sample.** The holdout remains unspent because nothing earned a look at it.

This is the expected outcome given the prior in RESEARCH_PLAN §2, and it is reported as a
result rather than iterated away. The detailed reasons per family are in
`REJECTED_STRATEGIES.md`.
