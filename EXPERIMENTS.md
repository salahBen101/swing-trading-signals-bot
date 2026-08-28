# EXPERIMENTS — research ledger

Record every strategy or risk experiment before interpreting its holdout result. Include
data version/hash, split, parameters searched, costs/slippage, hypothesis, result, and a
clear decision. Rejected hypotheses stay in this file so they are not rediscovered and
retested without accounting for the search count.

## Carried-forward evidence (recorded 2026-08-22)

Prior repository research screened thirteen intraday NQ strategy families and more than
one hundred configurations from 2017 through 2026. None survived out of sample. Eight of
nine families examined on five-minute bars lacked a gross edge before costs. The narrow
open 30-minute ORB result was dominated by 2022 and failed its volatility-regime rescue
test. See `PROJECT_SPEC.md` section 12 for the retained summary.

Decision: treat every current strategy as an unproven hypothesis. Do not deploy based on
this historical search.

## 2026-08-25 — official-rule change alert repeated and expanded

- Profiles checked: Tradeify Growth 50K, Select 50K, and Lightning 50K.
- Method: the same normalized official-page fingerprint comparison against the locked
  human-reviewed 2026-08-22 baselines; no profile mutation was permitted.
- Result at `2026-08-25T18:39:32.039836+00:00`: Growth 2/9 changed, Lightning 2/10
  changed, and Select 3/10 changed.
- The four August 24 mismatches remain. The shared
  [Daily Loss Limit](https://help.tradeify.co/en/articles/10468321-rules-daily-loss-limit)
  page is an additional mismatch affecting all three profiles.
- Evidence: new append-only JSONL records plus three timestamped alert documents and
  three empty, unapproved proposals under `logs/prop_rule_alerts/`.
- Holdout status: untouched; this was an operational rule check, not strategy research.
- Decision: keep `safe_to_reuse_reviewed_rules=false`, retain the 2026-08-22 profile
  contents/hashes, and require human review plus a new tested and approved version.

## 2026-08-24 — official-rule change alert (not a new verification)

- Profiles checked: Tradeify Growth 50K, Select 50K, and Lightning 50K.
- Method: fetched every official URL declared by each profile and compared normalized
  content with the human-reviewed 2026-08-22 SHA-256 baselines.
- Result: Growth had 1 of 9 pages changed, Lightning 1 of 10, and Select 2 of 10 at
  `2026-08-24T23:35:50.533847+00:00`.
- Changed official pages:
  - [Growth Evaluation Accounts](https://help.tradeify.co/en/articles/10495915-growth-evaluation-accounts)
  - [Lightning Funded Accounts](https://help.tradeify.co/en/articles/10495938-lightning-funded-accounts)
  - [Select Evaluation Accounts](https://help.tradeify.co/en/articles/12853921-select-evaluation-accounts)
  - [Select Flex and Select Daily Payout Policies](https://help.tradeify.co/en/articles/12853966-select-flex-and-select-daily-payout-policies)
- Evidence: append-only check records in `logs/prop_rule_verification.jsonl`; three alert
  documents and three empty, unapproved review proposals in `logs/prop_rule_alerts/`.
- Interpretation: a content hash proves only that reviewed text changed. It does not say
  which rule changed, approve a new interpretation, or justify mutating a profile.
- Decision: `safe_to_reuse_reviewed_rules=false` for all three profiles. Keep the dated
  profiles and baselines unchanged/stale, block Stage 2+, and require human review plus a
  newly versioned and tested profile. No live execution was enabled.

## 2026-08-22 — official-rule verification (operational evidence, not a strategy test)

- Profiles: Tradeify Growth 50K, Select 50K, and Lightning 50K.
- Evidence: every URL declared by each profile was fetched from Tradeify's official help
  content and compared with the reviewed SHA-256 baseline in
  `config/prop_firms/source_snapshots.yaml`.
- Result: Growth 9/9 unchanged; Select 10/10 unchanged; Lightning 10/10 unchanged at
  `2026-08-22T20:00:00+00:00`.
- Audit record: append-only results in `logs/prop_rule_verification.jsonl`.
- Interpretation: the reviewed source text was unchanged. This does not resolve the
  documented permitted-time conflict, grant API access, establish a trading edge, or
  authorize Stage 3/4.
- Decision: profiles remain usable for Stage 0 research only; deployment stays blocked.

## Template

### YYYY-MM-DD — experiment id and hypothesis

- Data and content hash:
- Train / validation / holdout boundaries:
- Configurations evaluated before this result:
- Costs, fees, and slippage:
- Parameters:
- Result:
- Robustness / Monte Carlo result:
- Holdout status (untouched / spent):
- Decision:
