from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from tradebot.prop_firms import AccountPhase, PropProfileError, load_prop_profile


ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "config" / "prop_firms"


def load(name: str):
    return load_prop_profile(PROFILES / name)


def test_every_checked_in_profile_is_current_official_and_fail_closed_for_api() -> None:
    files = sorted(PROFILES.glob("tradeify_*_50k.yaml"))
    assert [path.name for path in files] == [
        "tradeify_growth_50k.yaml",
        "tradeify_lightning_50k.yaml",
        "tradeify_select_50k.yaml",
    ]
    for path in files:
        profile = load_prop_profile(path)
        assert profile.verified_on.isoformat() == "2026-08-22"
        assert profile.reverify_after_hours == 24
        assert profile.account_size_usd == 50_000
        assert all(source.official for source in profile.sources)
        assert all(source.url.startswith("https://help.tradeify.co/") for source in profile.sources)
        assert any("news-trading" in source.url for source in profile.sources)
        assert profile.compliance.automation_allowed
        assert not profile.compliance.tradovate_api_allowed
        assert not profile.compliance.bot_cross_firm_deployment_allowed
        assert profile.compliance.proof_video_may_be_required
        assert profile.ambiguity_notes


def test_growth_current_rules_and_drawdown_floor() -> None:
    profile = load("tradeify_growth_50k.yaml")
    evaluation = profile.rules_for("evaluation")
    assert evaluation.phase is AccountPhase.EVALUATION
    assert evaluation.profit_target_usd == 3_000
    assert evaluation.minimum_trading_days == 1
    assert evaluation.daily_loss_limit_usd == 1_250
    assert evaluation.contracts.max_micros == 40
    assert evaluation.consistency.fraction_for_cycle() is None
    assert evaluation.drawdown.floor(50_000, 50_000) == 48_000
    assert evaluation.drawdown.floor(50_000, 51_000) == 49_000
    assert evaluation.drawdown.is_breached(49_000, 49_000)

    funded = profile.rules_for("sim_funded")
    assert funded.drawdown.lock_trigger_balance_usd == 52_100
    assert funded.drawdown.floor(50_000, 52_100, locked=True) == 50_100
    assert funded.daily_loss_limit_for(52_999) == 1_250
    assert funded.daily_loss_limit_for(53_000) == 2_000
    assert funded.consistency.fraction_for_cycle() == pytest.approx(0.35)
    assert funded.payout.minimum_balance_usd == 53_000
    assert funded.payout.winning_day_strictly_greater
    assert [funded.payout.payout_cap(i) for i in range(1, 6)] == [
        1_500, 2_000, 2_500, 3_000, 3_000
    ]


def test_select_evaluation_and_sticky_funded_scaling() -> None:
    profile = load("tradeify_select_50k.yaml")
    evaluation = profile.rules_for("evaluation")
    assert evaluation.profit_target_usd == 3_000
    assert evaluation.minimum_trading_days == 3
    assert evaluation.daily_loss_limit_usd is None
    assert evaluation.consistency.fraction_for_cycle() == pytest.approx(0.40)

    flex = profile.rules_for("sim_funded_flex")
    assert flex.daily_loss_limit_usd is None
    assert flex.contracts.allowed_micros(50_000) == 20
    assert flex.contracts.allowed_micros(51_499.99) == 20
    assert flex.contracts.allowed_micros(51_500) == 30
    assert flex.contracts.allowed_micros(52_000) == 40
    assert flex.payout.winning_days_required == 5
    assert flex.payout.max_fraction_total_profit == pytest.approx(0.50)

    daily = profile.rules_for("sim_funded_daily")
    assert daily.daily_loss_limit_usd == 1_000
    assert daily.payout.minimum_balance_usd == 52_100
    assert daily.payout.minimum_balance_must_remain_after_payout
    assert daily.payout.minimum_payout_usd == 250
    assert daily.payout.cycle_profit_multiplier == 2.0


def test_lightning_progressive_consistency_and_payout_goals() -> None:
    rules = load("tradeify_lightning_50k.yaml").rules_for("sim_funded")
    assert [rules.consistency.fraction_for_cycle(i) for i in range(1, 5)] == [
        0.20, 0.25, 0.30, 0.30
    ]
    assert rules.consistency.resets_after_payout
    assert rules.payout.first_profit_goal_usd == 3_000
    assert rules.payout.subsequent_profit_goal_usd == 2_000
    assert [rules.payout.payout_cap(i) for i in range(1, 6)] == [
        2_000, 2_000, 2_000, 2_500, 2_500
    ]


def test_dedicated_flatten_deadline_is_conservative() -> None:
    window = load("tradeify_growth_50k.yaml").trading_window
    assert not window.position_must_be_flat(datetime(2026, 8, 22, 20, 44, tzinfo=timezone.utc))
    assert window.position_must_be_flat(datetime(2026, 8, 22, 20, 45, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="timezone-aware"):
        window.position_must_be_flat(datetime(2026, 8, 22, 16, 45))


def test_daily_verification_freshness_is_conservative() -> None:
    profile = load("tradeify_growth_50k.yaml")
    assert profile.rules_are_fresh(datetime(2026, 8, 23, 0, 0, tzinfo=timezone.utc))
    assert not profile.rules_are_fresh(datetime(2026, 8, 23, 0, 0, 1, tzinfo=timezone.utc))


def test_unknown_profile_key_is_rejected_with_path(tmp_path: Path) -> None:
    raw = yaml.safe_load((PROFILES / "tradeify_growth_50k.yaml").read_text(encoding="utf-8"))
    raw["phases"]["evaluation"]["contracts"]["max_microz"] = 99
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(PropProfileError, match=r"phases\.evaluation\.contracts.*max_microz"):
        load_prop_profile(bad)


def test_source_check_date_must_match_profile(tmp_path: Path) -> None:
    raw = yaml.safe_load((PROFILES / "tradeify_growth_50k.yaml").read_text(encoding="utf-8"))
    raw["sources"][0]["checked_on"] = "2026-08-21"
    bad = tmp_path / "stale-source.yaml"
    bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(PropProfileError, match="checked_on must match"):
        load_prop_profile(bad)
