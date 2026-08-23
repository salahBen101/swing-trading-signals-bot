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
