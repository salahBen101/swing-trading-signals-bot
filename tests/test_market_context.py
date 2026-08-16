"""Tests for the strict, fail-closed squeeze market-context gate."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from scanner.market_context import (
    MarketContext,
    evaluate_context,
    load_context,
    merge_override,
    save_context,
)


NOW = datetime(2026, 8, 7, 20, 30, tzinfo=timezone.utc)


def qualified_context() -> MarketContext:
    """A complete fixture whose data dates are inside the deliberately tight policies."""

    return MarketContext(
        ticker="AAPL",
        retrieved_at=NOW,
        short_interest_pct_float=0.12,
        days_to_cover=2.1,
        short_interest_as_of=date(2026, 8, 1),
        short_interest_source="licensed short-interest vendor",
        short_interest_verified=True,
        borrow_available_shares=200_000,
        borrow_fee_rate_pct=1.3,
        borrow_as_of=NOW - timedelta(minutes=10),
        borrow_source="broker securities-lending feed",
        borrow_verified=True,
        next_earnings_date=date(2026, 8, 24),
        earnings_as_of=NOW - timedelta(minutes=10),
        earnings_source="company investor relations calendar",
        earnings_verified=True,
        options_as_of=NOW - timedelta(minutes=10),
        option_call_open_interest=120_000,
        option_put_open_interest=90_000,
        options_source="licensed options feed",
        options_verified=True,
    )


def test_complete_fresh_verified_context_qualifies() -> None:
    gate = evaluate_context(qualified_context(), now=NOW)

    assert gate.eligible
    assert gate.label == "QUALIFIED"
    assert gate.reasons == ()


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"borrow_verified": False}, "verified borrow availability/fee snapshot missing"),
        ({"borrow_as_of": NOW - timedelta(hours=25)}, "borrow snapshot is stale"),
        ({"borrow_as_of": NOW + timedelta(minutes=1)}, "borrow snapshot is stale"),
        ({"earnings_as_of": NOW - timedelta(hours=25)}, "earnings-calendar snapshot is stale"),
        ({"next_earnings_date": date(2026, 8, 12)}, "earnings within 10 calendar days"),
        ({"options_as_of": NOW - timedelta(hours=25)}, "option-chain snapshot is stale"),
        ({"options_as_of": NOW + timedelta(minutes=1)}, "option-chain snapshot is stale"),
        ({"short_interest_pct_float": 1.2}, "short-interest percentage or days-to-cover missing"),
        ({"retrieved_at": NOW + timedelta(minutes=1)}, "market-context snapshot timestamp is in the future"),
    ],
)
def test_missing_stale_future_or_invalid_component_suppresses(change: dict[str, object], reason: str) -> None:
    gate = evaluate_context(replace(qualified_context(), **change), now=NOW)

    assert not gate.eligible
    assert reason in gate.reasons


def test_json_roundtrip_preserves_context_and_false_string_is_false(tmp_path) -> None:
    context = qualified_context()
    path = save_context(context, tmp_path)

    loaded = load_context("AAPL", tmp_path)
    assert path.exists()
    assert loaded == context

    raw = context.to_dict()
    raw["borrow_verified"] = "false"
    assert not MarketContext.from_dict(raw).borrow_verified


def test_partial_override_cannot_reclassify_yahoo_data_as_verified() -> None:
    with pytest.raises(ValueError, match="options_verified=true requires same-source fields"):
        merge_override(qualified_context(), {"ticker": "AAPL", "options_verified": True})


def test_complete_independent_override_can_qualify_a_blank_snapshot() -> None:
    source = qualified_context()
    override = {
        "ticker": "AAPL",
        "short_interest_pct_float": source.short_interest_pct_float,
        "days_to_cover": source.days_to_cover,
        "short_interest_as_of": source.short_interest_as_of.isoformat(),
        "short_interest_source": source.short_interest_source,
        "short_interest_verified": True,
        "borrow_available_shares": source.borrow_available_shares,
        "borrow_fee_rate_pct": source.borrow_fee_rate_pct,
        "borrow_as_of": source.borrow_as_of.isoformat(),
        "borrow_source": source.borrow_source,
        "borrow_verified": True,
        "next_earnings_date": source.next_earnings_date.isoformat(),
        "earnings_as_of": source.earnings_as_of.isoformat(),
        "earnings_source": source.earnings_source,
        "earnings_verified": True,
        "options_as_of": source.options_as_of.isoformat(),
        "option_call_open_interest": source.option_call_open_interest,
        "option_put_open_interest": source.option_put_open_interest,
        "options_source": source.options_source,
        "options_verified": True,
    }
    blank = MarketContext(ticker="AAPL", retrieved_at=NOW)

    merged = merge_override(blank, override)

    assert evaluate_context(merged, now=NOW).eligible
