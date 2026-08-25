# REJECTED_STRATEGIES — every failed hypothesis, and why

Negative results are the main product of this research phase, not its waste
(RESEARCH_PLAN §8, rule 11). Every hypothesis tested is here with the specific reason it
was rejected, so the search is not silently repeated later.

All sixteen pre-registered families were rejected. None reached the holdout. Full metrics
are in `STRATEGY_REGISTRY.md`.

**Rejection reason codes**

| code | meaning |
|---|---|
| NO-GROSS | no gross edge before costs — the entry predicts nothing |
| BELOW-HURDLE | positive gross, but under the 4.48-pt cost hurdle, so net is negative or negligible |
| SIGN-FLIP | positive on DEV, negative on validation — the DEV result was noise or regime |
| ONE-YEAR | profit concentrated in a single year (beta/regime in disguise) |
| ONE-SIDED | edge exists only long or only short, without a structural reason |
| CONCENTRATED | result depends on a handful of extreme trades |

---

## A. Opening dynamics

### A1 — opening drive · BELOW-HURDLE
*A session that drives away from the open on volume continues.* DEV gross +1.34 pt
(t=+0.15), a quarter of the cost hurdle; net +0.22 pt. Validation held the sign (+1.58 gross)
but at t=+0.16 — a positive coin, not an edge. The drive contains no exploitable
information once the 1.12-pt round trip is charged.

### A2 — opening range expansion · NO-GROSS
*Wide opening ranges trend further.* DEV gross −7.31 pt over only 80 trades, the worst on
the board. The "recovery" to +4.70 on validation (n=34) is a sign flip on a sample too small
to mean anything. Wide-open sessions do not continue; if anything the wide open is the move.

### A3 — failed opening-range breakout · NO-GROSS
*A failed ORB reverses to the opposite extreme.* Negative gross on both splits (DEV −0.90
t=−2.17, VAL −2.08). Fading the failed break loses before costs. The trapped-trader story is
appealing and wrong: the failure is not reliably followed by a move the other way.

### A4 — opening reversal · ONE-YEAR + ONE-SIDED + SIGN-FLIP
*A gap that reverses in the first 30 minutes keeps reversing.* **The only family to clear the
gross cost hurdle** (DEV +5.18 pt), and it failed everything after. 2022 alone supplied +92%
of net profit; the edge was +8.2 pt short versus +1.3 pt long; trimming the best/worst 5% of
trades cut net by 63%; and validation inverted the sign to −1.80. This is 2022's downtrend
paying short gap-fades, i.e. beta wearing a costume — the exact failure mode that ended the
previous best candidate in this project.

### A5 — overnight inventory correction · NO-GROSS
*A strong one-way overnight session is partly given back in RTH.* First testable here thanks
to the new ETH data. DEV gross −0.21, validation −2.10. The microstructural story (thin
overnight, one-sided dealer inventory) is plausible but the fade does not pay: whatever
inventory effect exists is smaller than the noise and well inside costs.

---

## B. VWAP and session structure

### B1 — VWAP continuation · SIGN-FLIP
*Pullbacks to VWAP resume in a one-sided session.* DEV gross +0.41 (below hurdle), flips to
−0.56 on validation. 4,109 DEV trades — a large sample confirming the effect is genuinely
absent, not merely unmeasured.

### B2 — VWAP rejection · SIGN-FLIP
*A failed reclaim of VWAP continues away.* DEV +0.34, VAL −0.34. Mirror of B1 and equally
empty. Both VWAP-directional stories fail on 3,000+ trade samples.

### B3 — stretched deviation reversion · NO-GROSS
*Price ≥ 2 ATR from VWAP reverts.* The most-traded family (4,945 DEV) and one of the most
decisively negative: DEV gross −1.65 at t=−6.75, VAL −1.47 at t=−3.29. Fading a stretch from
VWAP loses money reliably before costs — a strongly *negative* effect. Deviation from VWAP
is momentum, not a rubber band, on this instrument at this timeframe.

### B4 — VWAP reclaim after failed auction · BELOW-HURDLE / SIGN-FLIP
*A failed push to a session extreme plus a VWAP reclaim continues.* DEV gross essentially
zero (−0.00), VAL −2.39. The two-stage confirmation buys nothing.

---

## C. Prior-session structure

### C1 — previous-day high/low breakout · BELOW-HURDLE
*Clearing a prior extreme continues.* DEV +0.96 gross (t=−0.15 net), VAL +1.64 gross at
t=+0.25. Sign held across splits, which is more than most managed, but at a fifth of the cost
hurdle and t≈0.2 it is noise. The stop cluster at prior levels, if it exists, is arbitraged
flat.

### C2 — failed prior-extreme breakout · SIGN-FLIP
*Clearing a prior extreme then closing back inside reverses.* DEV −1.25, VAL +1.49 — a sign
flip, in the direction that makes it untradeable as a pre-registered rule. n=957/350.

### C3 — previous-close interaction · NO-GROSS
*Rejection of the prior settlement continues.* Strongly negative on both splits (DEV −3.24
t=−3.74, VAL −2.82). Trading around settlement loses before costs; the level attracts flow
but does not predict direction out of it.

### C4 — overnight high/low sweep and rejection · SIGN-FLIP
*RTH sweeps an overnight extreme and rejects.* The best-motivated of the new overnight
hypotheses and the second-best DEV gross (+2.13), and it still flipped hard to −2.36 on
validation. The overnight extreme is swept and rejected often enough to look like a signal on
one sample and not the next. Thin-liquidity overnight prints are real; a tradeable rejection
off them is not.

---

## D. Trend and pullback

### D1 — strong-session pullback · SIGN-FLIP
*Pullbacks resume in a high-efficiency session.* DEV +1.56 gross (below hurdle), VAL −0.74.
The regime filter (efficiency ≥ 0.4) does not make the pullback pay out of sample.

### D2 — breakout → pullback → continuation · BELOW-HURDLE
*The retest after a session-extreme break continues.* DEV +0.52, VAL +0.11 — positive both
times but negligible and net-negative after costs. Requiring the retest trades fewer signals
without making them better.

### D3 — volatility-adjusted trend continuation · BELOW-HURDLE
*Trend continuation, filtered to higher-volatility regimes.* DEV +1.10 gross decays to +0.07
on validation. The volatility regime filter — the actual hypothesis — adds nothing; the edge
it was supposed to isolate is not there.

---

## What the failures have in common

- **The gross screen is where most die.** Seven of sixteen have no positive gross at all, and
  another three are under half a point. This is not a cost problem; the entries carry no
  directional information. Costs then finish the rest.
- **The overnight data changed nothing.** A5, C4 and the gap-based A4 were untestable before
  this phase and were the ones with the cleanest microstructural stories. All three failed.
  The overnight session is not a reservoir of unexploited intraday edge.
- **Mean-reversion against VWAP is negative, not flat.** B3 in particular is reliably
  money-losing before costs (t=−6.75). On NQ 5-minute bars, stretch from the session average
  is a continuation signal, not a reversion one — the opposite of the retail intuition.
- **The one apparent winner was a regime.** A4's entire result was 2022 shorts, which is the
  same lesson the ORB narrow-open filter taught before: a single trending year can
  manufacture a backtest that clears every hurdle except the year-by-year one.

Thirty intraday NQ/MNQ families across all work in this repository; none survived
out-of-sample. The consistent, well-replicated finding is that liquid index-future intraday
returns at the 5-to-20-minute horizon are, for the rule families a retail participant can
express, indistinguishable from a random walk once the 1.12-point round-trip cost is paid.

---

## Amendment 1 — dynamic exits, volume profile, order flow (2026-08-22)

Proposal: add bid/ask and volume-profile confirmations, and dynamic exits (cut losers early,
trail winners). All rejected. Full context in `RESEARCH_PLAN.md` §12; reproduce with
`scripts/research_dynamic.py`.

### Bid/ask order-flow confirmation · UNTESTABLE
No quote/BBO data exists over the research period. The only order-flow data
(`nq_orderflow_15s.parquet`) is a trade-imbalance proxy confined to 2025-07 → 2026-02 —
entirely inside the holdout. It cannot be tested on DEV/validation, and spending the holdout
is forbidden. Prior tick-order-flow research (commit `c31f19e`) already found no edge. Not
evaluated: the data does not exist where it could legitimately be used.

### Dynamic exits (trail winners / cut losers) · NO-EDGE-TO-EXIT
The null control is the verdict: the identical dynamic exit applied to *random* entries
averages −0.90 pt net on DEV and −1.07 pt on validation, with a 95th percentile of +1.4 to
+2.2 pt purely from luck. Applied to the momentum entries, A1/C1/D1/D2/D3 all land inside
that null band — the exit made them indistinguishable from random entries. This is the
theoretical result that a trailing stop cannot create expectancy where the entry has no
directional persistence; the entries do not, so it does not. A4 nominally beat the null 95th
(+1.87) but was more concentrated than ever (2022 = 142% of net, short-only) and inverted to
−3.47 on validation.

### Volume-profile confirmation · DEGRADES EVERY ENTRY
Requiring price to have left the developing value area (an acceptance/continuation filter)
made every tested entry worse: C1 +0.40→−0.01, D1 +0.13→−1.80, A4 +1.87→−2.98. A real
continuation edge would be *sharpened* by this filter. That it is dulled — and that it strips
out precisely the A4 trades that looked good — confirms there is no acceptance structure to
confirm, only the 2022 downtrend.

**Lesson.** Exit engineering and confirmation filters cannot rescue an entry with no gross
signal. The null control should be the first test of any future exit idea: if it lifts random
entries, it is a mirage.
