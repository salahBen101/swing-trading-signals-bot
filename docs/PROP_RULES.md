# Tradeify 50K prop-account rules

Last verified against official sources: **2026-08-22**  
Profile scope: newly purchased/current-dashboard 50K futures accounts  
Runtime profiles: `config/prop_firms/tradeify_growth_50k.yaml`,
`tradeify_select_50k.yaml`, and `tradeify_lightning_50k.yaml`

## CHANGE ALERT — 2026-08-25 — UNAPPROVED / NOT A VERIFICATION

The daily official-source check at `2026-08-25T18:39:32.039836+00:00` again found the
four pages reported on August 24 and additionally found that Tradeify's shared
[Daily Loss Limit](https://help.tradeify.co/en/articles/10468321-rules-daily-loss-limit)
page no longer matches the reviewed baseline. The distinct changed-page set is now five
official pages. Per-profile results were Growth 2/9 changed, Lightning 2/10 changed, and
Select 3/10 changed.

This is fingerprint evidence only. It does not approve the page's current wording, infer
a cohort rule, or replace the human-reviewed 2026-08-22 baseline. The checked-in profiles
and source hashes were not changed. New append-only records, alerts, and empty unapproved
proposals were written under `logs/`; all profiles remain stale and Stage 2+ remains
blocked pending human review and a newly versioned, tested, explicitly approved baseline.

## CHANGE ALERT — 2026-08-24 — UNAPPROVED / NOT A VERIFICATION

The automated official-source check at `2026-08-24T23:35:50.533847+00:00` detected new
content hashes on four official Tradeify pages:

| Affected profile | Official page whose reviewed content changed |
|---|---|
| Growth 50K | [Growth Evaluation Accounts](https://help.tradeify.co/en/articles/10495915-growth-evaluation-accounts) |
| Lightning 50K | [Lightning Funded Accounts](https://help.tradeify.co/en/articles/10495938-lightning-funded-accounts) |
| Select 50K | [Select Evaluation Accounts](https://help.tradeify.co/en/articles/12853921-select-evaluation-accounts) |
| Select 50K | [Select Flex and Select Daily Payout Policies](https://help.tradeify.co/en/articles/12853966-select-flex-and-select-daily-payout-policies) |

This check establishes only that the reviewed text changed. It does **not** establish what
rule changed, approve a new interpretation, replace the human-reviewed 2026-08-22
baseline, or authorize any profile edit. The three machine-readable profiles and
`source_snapshots.yaml` intentionally remain unchanged and are now stale for Stage 2+.
The rule matrix below remains the dated 2026-08-22 baseline for labelled Stage 0/1
research only.

Before any affected profile can be reconsidered, a human must review the new official
text, resolve conflicts, create a new dated profile and source baseline, test the proposed
change, and explicitly approve it. No live execution was enabled; all Stage 2+ execution
remains fail-closed.

This document records the firm's outer limits. They are not trading targets. The project's
personal defaults—$200 maximum planned risk per trade, one trade per Tradeify session,
$200 maximum daily strategy loss, and one open position—are deliberately much stricter.

Legacy accounts can retain different targets and position limits after reset. The current
profiles must not be selected for a legacy account merely because its nominal size is 50K.
The account's actual Tradeify dashboard values are authoritative.

## Official sources checked

All pages below were checked on 2026-08-22.

| Topic | Official source |
|---|---|
| Growth evaluation | [Growth Evaluation Accounts](https://help.tradeify.co/en/articles/10495915-growth-evaluation-accounts) |
| Select evaluation | [Select Evaluation Accounts](https://help.tradeify.co/en/articles/12853921-select-evaluation-accounts) |
| Lightning funded rules | [Lightning Funded Accounts](https://help.tradeify.co/en/articles/10495938-lightning-funded-accounts) |
| Growth payouts | [Growth Funded: Account Payout Policy](https://help.tradeify.co/en/articles/11083796-growth-funded-account-payout-policy) |
| Select funded/payout policies | [Select Flex and Select Daily Payout Policies](https://help.tradeify.co/en/articles/12853966-select-flex-and-select-daily-payout-policies) |
| Lightning payouts | [Lightning Funded: Account Payout Policy](https://help.tradeify.co/en/articles/10495932-lightning-funded-account-payout-policy) |
| Trailing drawdown | [Rules: Trailing Max Drawdowns](https://help.tradeify.co/en/articles/10495897-rules-trailing-max-drawdowns) |
| Daily loss | [Rules: Daily Loss Limit](https://help.tradeify.co/en/articles/10468321-rules-daily-loss-limit) |
| Consistency | [Rules: Consistency Rule](https://help.tradeify.co/en/articles/10468320-rules-consistency-rule) |
| Trading hours | [Rules: Permitted Times to Trade](https://help.tradeify.co/en/articles/10495876-rules-permitted-times-to-trade) |
| Bot, microscalping, activity, DCA | [Guidelines for Traders](https://help.tradeify.co/en/articles/10468318-guidelines-for-traders) |
| Hedging/product groups | [Rules: Hedging & Correlated Products](https://help.tradeify.co/en/articles/10495868-rules-hedging-correlated-products) |
| Tradovate API availability | [Rules: Supported Trading Products & Assets](https://help.tradeify.co/en/articles/10468222-rules-supported-trading-products-assets) |
| News trading | [Rules: News Trading](https://help.tradeify.co/en/articles/10495874-rules-news-trading) |
| Contractual conduct | [Tradeify funded trader agreement](https://tradeify.co/funded-trader-agreement) |

## Current 50K rule matrix

| Program and stage | Profit target / minimum days | Maximum drawdown | Daily loss limit | Maximum position | Consistency |
|---|---|---|---|---|---|
| Growth Evaluation | $3,000; can pass in 1 trading day | $2,000 EOD trailing | $1,250 soft pause | 4 minis / 40 micros | None in evaluation |
| Growth Sim Funded | No evaluation target; payout balance at least $53,000 | $2,000 EOD trailing; locks at $50,100 | $1,250; next-session increase to $2,000 after EOD balance reaches $53,000 | 4 / 40 | 35% for payout |
| Select Evaluation | $3,000; at least 3 trading days | $2,000 EOD trailing | None | 4 / 40 | 40% to pass |
| Select Flex Sim Funded | Payout milestones, no account profit target | $2,000 EOD trailing; locks at $50,100 | None | starts 2 / 20; scales to 3 / 30 at EOD $51,500 and 4 / 40 at EOD $52,000 | None |
| Select Daily Sim Funded | Buffer/payout milestones, no account profit target | $2,000 EOD trailing; locks at $50,100 | $1,000 soft pause | same Select scaling | None |
| Lightning Sim Funded | No evaluation; fresh payout profit goal $3,000 first, $2,000 later | $2,000 EOD trailing; locks at $50,100 | $1,250; next-session increase to $2,000 after EOD balance reaches $53,000 | 4 / 40 | 20%, then 25%, then 30% for payout cycles |

Ten micros equal one mini for the position-limit calculation. Minis and micros may be
mixed, but gross combined exposure must remain within the limit. Opposing positions are
not netted into permission.

## Drawdown methodology

For all current profiles in this file:

- The initial 50K failure floor is `$50,000 - $2,000 = $48,000`.
- Before any funded lock, the floor is `highest completed EOD balance - $2,000`.
- The high-water mark updates only at end of day and never moves down.
- The current floor is enforced in real time against net liquidation. Reaching or falling
  below it is a hard, permanent account failure.
- Evaluation drawdowns do not lock.
- A Sim Funded 50K floor locks permanently at $50,100 when the EOD balance reaches
  $52,100. Select Flex also documents an immediate lock when a payout is requested, if it
  has not already locked.

“EOD trailing” therefore does **not** mean intraday losses can cross the existing floor.
Only movement of the floor waits until EOD.

The Daily Loss Limit (where present) is a soft pause until the next 6:00 PM ET session; it
does not rescue an account that first reaches the hard drawdown floor, and slippage can
overshoot it. It must never be used as the bot's stop.

## Consistency calculations

The official calculation is:

```text
largest profitable EOD day / cumulative profit for the applicable cycle <= limit
```

Losing days reduce the denominator. The official page states that commissions are not
included in its profit figure. Exceeding consistency delays a pass or payout; it does not
fail the account.

- Select Evaluation: 40%, evaluation cycle only.
- Growth Sim Funded: 35%, each payout cycle.
- Lightning: 20% for payout one, 25% for payout two, 30% for payout three and later;
  both consistency and fresh-profit tracking reset after payout.
- Select funded: no consistency rule.

## Payout eligibility

Amounts below are gross request limits; Tradeify documents a 90% trader / 10% firm split.
The simulator must not treat meeting a mathematical threshold as a guaranteed approval.

### Growth Sim Funded 50K

- Maintain balance of at least $53,000 until approval.
- Satisfy 35% consistency.
- Accumulate five qualifying days per payout cycle, each stated as profit **greater than**
  $150; the day count resets after payout.
- Minimum request: $500.
- Maximum requests: $1,500 / $2,000 / $2,500 / $3,000 for payouts 1 / 2 / 3 / 4+.

### Select Flex Sim Funded 50K

- Five winning days per cycle at $150 minimum daily profit.
- No minimum balance and no consistency rule.
- Request at most 50% of total current profit, capped at $3,000.
- After payout one, each new payout cycle must first return to positive net profit.

### Select Daily Sim Funded 50K

- Daily eligibility after EOD reconciliation.
- The balance must exceed $52,100 and remain above that buffer after withdrawal.
- Minimum request $250; cap $1,000.
- For later requests, the cycle must be positive and the request is also capped at twice
  the profit earned since the preceding payout.

### Lightning Sim Funded 50K

- No minimum trading-day count.
- Earn $3,000 fresh profit for payout one and $2,000 fresh profit after every payout.
- Meet the cycle's 20% / 25% / 30% consistency threshold.
- Minimum request $1,000.
- Maximum requests: $2,000 for payouts 1–3, then $2,500 for payout 4+.
- Leftover account balance does not satisfy the next cycle's fresh-profit goal.

### Funded-account hold-duration test

Every funded payout also requires both:

- more than 50% of trades held longer than 10 seconds; and
- more than 50% of profit generated by trades held longer than 10 seconds.

Failure blocks payout rather than failing the account.

## Trading times

- Tradeify session: 6:00 PM ET through 5:00 PM ET the next calendar day.
- Every position must be flat by **4:45 PM ET**.
- Holiday shortened-session deadline: **12:59 PM ET**.
- No position through the maintenance break, across sessions, or over a weekend.
- Holiday hours are announced externally. If the holiday schedule has not been verified,
  the production-stage rule is to reject entries.

This project is stricter: the initial strategy configuration trades RTH only and flattens
before 4:00 PM ET.

## Prohibited and restricted conduct

- No opposing positions on the same instrument or correlated instruments in the same
  product group, whether in one account or across accounts controlled by the trader. MNQ
  and MES are both equity-index products.
- No team trading, third-party account management, or copying another person's strategy.
- No exploitation of display discrepancies, delayed/external feeds, platform errors, or
  system delays.
- No manipulative/disruptive behaviour or HFT bots.
- News trading is allowed at the trader's risk.
- Tradeify permits structured DCA/scaling, but this project still bans averaging down,
  martingale, and post-loss size increases.
- At least one trade per calendar week is expected for account activity. The bot must
  never manufacture a non-setup trade to satisfy this.

## Automation and execution-route restrictions

Personal bots are allowed only if the operator can prove sole ownership, no one else has
access to or uses the strategy, the bot is not shared with another trader or prop firm,
and it is not HFT. Tradeify may request documentation and a live video of the operator
enabling the code on the operator's PC.

This makes multi-firm *architecture* acceptable but simultaneous deployment of the same
strategy/build to Tradeify and another firm prohibited. A future production gate must pin
and attest the deployed build hash to Tradeify exclusively.

Tradeify states that the Tradovate API is unavailable for Evaluation and Sim Funded
accounts; it is available only for Live funded accounts. The repository's Tradovate demo
adapter is therefore a paper/testing seam, **not an authorized Tradeify Stage 3 execution
route**. Stage 3 remains disabled until an approved platform-native route is separately
verified.

## Conflicts and fail-closed decisions

The following official-source inconsistencies are recorded rather than silently guessed:

- A general pricing overview labels Growth with consistency, while the dedicated Growth
  Evaluation page says none and the consistency/payout pages assign 35% to Growth Sim
  Funded. Profiles use the stage-specific dedicated pages.
- The dedicated July 2026 trading-time page says 4:45 PM ET; an older/common FAQ says
  4:59 PM. Profiles enforce 4:45 PM.
- DLL documentation does not precisely settle unrealized-P&L treatment or whether the 6%
  increased DLL persists after later payouts. Broker/dashboard state must be authoritative
  and written confirmation is required before Stage 3.
- Select Daily wording is inconsistent about whether the 2x continuity formula constrains
  payout one. It is modelled from payout two; the first-cycle $2,100 buffer already makes
  the distinction immaterial to its $1,000 cap.
- “Before the new dashboard launch” is not assigned a precise timestamp everywhere.
  Actual account dashboard parameters must be captured; legacy profiles cannot be
  inferred.

Result: the unchanged profiles are retained only as the dated 2026-08-22 baseline for
labelled backtest/replay research. The 2026-08-24 alert makes them stale for Stage 2+ until
human review produces and approves a new version. Automated paper, Evaluation, Sim Funded,
and live execution remain fail-closed.
