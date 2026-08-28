# Research Ideas — NOT ACTIVE

Observations noted during the infrastructure refactor. **None of these are implemented, enabled,
or reflected in the formal experiment.** They are recorded here precisely so that they do not get
quietly folded into a running trial.

Each would require its own experiment version, its own pre-registration, and its own out-of-sample
evidence before it could be taken seriously.

## 1. Exit threshold appears to be a no-op

Measured across four independent windows: exit at RSI(2) > 65, 70, 75 and 80 produce **identical**
final equity. The `close > 5-day SMA` rule fires on the same session in nearly every case, so
raising the RSI threshold only relabels the exit reason.

RSI > 60 did win 3 of 4 windows (+2.5 CAGR points, better Sharpe, smaller drawdown) by exiting
roughly a session earlier and freeing a capital slot sooner. That is a plausible mechanism rather
than pure curve-fit — but 3/4 is not 4/4, and the original validation specifically warned that
this strategy's threshold sensitivity degrades smoothly and must not be optimised further.

**Not changed.** Would need a fresh holdout.

## 2. Asymmetric exits helped every breakout entry tested

In the separate 5-minute futures work, a Chandelier trailing exit beat a fixed 2R target for
*every* entry family tested (+1.3 to +3 ticks gross each). Fixed targets amputate the right tail
that trend systems live on. Irrelevant to RSI(2) as specified — it is a mean-reversion strategy
with a fast structural exit — but worth remembering before assuming a fixed target is neutral.

## 3. The capital constraint is the dominant term, not the signal

Signals cluster: measured concurrency was mean 3.9, max 19, with 68 of 255 days needing more than
5 simultaneous positions. A small account misses signals precisely on broad selloff days, which is
when the setup fires most and works best.

At the formal trial's 1-position limit this is severe. The rejection log will measure it directly,
and that measurement is arguably more valuable than the P&L.

## 4. Options may be the wrong instrument for this signal

Signal-to-noise per trade: **0.204 underlying, 0.022 options.** Leverage multiplies edge and noise
equally while the spread subtracts a fixed cost, so the ratio collapses roughly tenfold.

If H1 holds and H2 fails, the honest conclusion is to trade the underlying, not to hunt for a
better contract-selection rule.

## 5. Tier labels do not mean what they appear to mean

`tier_of()` returns `retrospective` / `unconfirmed` / `no edge`. **No name carries a `strong`
tier.** The `strong` *preset* is a 63-name list and is unrelated to the tier vocabulary. This is
a naming collision that has already caused one misreading during this project.

## 6. CRM returns no data

yfinance reports "possibly delisted" for CRM. It silently shrinks the `strong` universe from 63 to
62 with no warning. Irrelevant to the formal five-ETF trial; needs fixing before any equity
universe is used again.
