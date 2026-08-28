# EXPLORATORY PILOT — INVALID FOR FORMAL PERFORMANCE ANALYSIS

**Archived 2026-08-28.** Frozen. Nothing here may be cited as strategy performance, quoted as a
track record, or used as evidence in a capital-allocation decision.

This directory holds the first paper-trading attempt, run between 2026-08-14 and 2026-08-20. It
was a useful engineering pilot — it proved the scanner, quote retrieval and daily automation work
end to end — and it is preserved for that reason. It is **not** a valid performance experiment,
for the specific reasons listed below. Each one independently disqualifies it.

## Why the numbers here are not evidence

**1. Same-session lookahead in execution.** The account detected signals from a completed daily
close and then bought options using quotes from *that same session*. A signal that only exists
after the close cannot be filled at a price observed before it existed. Every entry price in this
archive is therefore unobtainable. This is the single most serious defect and it invalidates the
P&L on its own.

**2. Fills were assumed, never obtained.** Buys were booked at the displayed ask and sells at the
displayed bid, as though a quote were a guaranteed fill. There was no broker, no order, no
acknowledgement, no rejection path, and no partial fills. A displayed quote is an invitation, not
an execution.

**3. Survivorship-biased universe.** The run used the `strong` preset — 63 names selected for
being liquid and important *today*. The repository's own `scanner/README.md` (line 169) states
plainly that this is "today's survivor/liquidity universe, not a point-in-time universe", and its
recommended protocol (line 181) specifies the five-ETF core instead. The pilot used the universe
the documentation warns against.

**4. Mutable JSON ledger.** Account state lived in a single rewritable JSON file with no
migrations, no append-only history, no transactions and no audit trail. State was rewritten in
place on every run, so earlier values are unrecoverable.

**5. Position sizing changed mid-flight.** Starting equity moved from $20,000 to $50,000 and the
allocation from 22%/5 slots to 12%/10 slots *during* the run, with earlier trades left at their
original size. The resulting equity curve reflects two different sizing regimes spliced together.

**6. No risk engine.** There was no per-trade premium cap, no daily loss limit, no concentration
limit and no centralized rejection path.

**7. Sample far too small to mean anything.** 3 closed trades. The power analysis in the project
notes puts the requirement at ~96 share-equivalent trades for a t-statistic of 2.0, and roughly
8,635 for the option wrapper. Three trades at a 100% win rate is noise.

## Contents

| file | what it is |
|---|---|
| `paper_account.json` | final state of the $50k pilot account (8 positions, 3 closed) |
| `paper_options_state.json` | the earlier $20k-era tracker this was migrated from |
| `PAPER_ACCOUNT.html` | dashboard as rendered on the last pilot run |
| `paper_trades.json` | 12-month Black-Scholes backfill — modelled prices, never traded |
| `PAPER_TRADING.md` | report generated from that backfill |
| `portfolio_sim.json` | historical share-based portfolio simulation summary |

`paper_trades.json` and `PAPER_TRADING.md` deserve their own warning: those option values came
from a Black-Scholes model with no bid/ask spread and constant implied volatility. They showed
roughly +34% per trade. That figure is a modelling artifact, not a result.

## What the pilot did establish

Genuinely useful, and the reason this is archived rather than deleted:

- the RSI(2) scanner produces signals correctly against live data
- option chains can be retrieved and specific contracts re-quoted by expiry and strike
- the daily GitHub Actions schedule runs unattended and reproduces local results exactly
- the spread cost is immediately visible and material — roughly 2% of premium on entry

The formal experiment supersedes this entirely and starts from a separate account, a separate
experiment ID and a clean ledger.
