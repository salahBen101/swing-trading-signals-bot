"""Focused regression tests for the standalone RSI(2) scanner."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scanner import options_fit
from scanner import rsi2_scanner as scanner
from scanner.squeeze_strategy import SqueezeSignal


class _FakeTicker:
    def __init__(self, expiry: str, calls: pd.DataFrame) -> None:
        self.options = [expiry]
        self._calls = calls

    def option_chain(self, expiry: str) -> SimpleNamespace:
        assert expiry in self.options
        return SimpleNamespace(calls=self._calls)


def test_package_import_keeps_sector_lookup_available(capsys) -> None:
    """``python -m scanner.rsi2_scanner`` must be able to render normal readings."""
    reading = scanner.Reading(
        ticker="AAPL",
        price=100.0,
        prev_close=101.0,
        rsi2=20.0,
        sma200=90.0,
        sma5=99.0,
        above_trend=True,
        signal=False,
        flip_price=None,
        stale=False,
    )

    scanner.render([reading], entry_rsi=5.0, show_options=False, send_notifications=False)

    assert "tech" in capsys.readouterr().out


def test_fetch_candidates_skips_nan_quotes_and_keeps_valid_rows(monkeypatch) -> None:
    expiry = (datetime.now(options_fit.MARKET_TZ).date() + timedelta(days=10)).isoformat()
    calls = pd.DataFrame(
        [
            {
                "strike": 100.0,
                "bid": np.nan,
                "ask": pd.NA,
                "impliedVolatility": 0.25,
                "openInterest": 100,
                "volume": 10,
            },
            {
                "strike": 100.0,
                "bid": np.nan,
                "ask": 2.0,
                "impliedVolatility": np.nan,
                "openInterest": np.nan,
                "volume": pd.NA,
            },
            {
                "strike": "not-a-strike",
                "bid": 1.0,
                "ask": 2.0,
                "impliedVolatility": 0.25,
                "openInterest": 100,
                "volume": 10,
            },
        ]
    )
    ticker = _FakeTicker(expiry, calls)
    monkeypatch.setattr(options_fit.yf, "Ticker", lambda symbol: ticker)

    candidates = options_fit.fetch_candidates("TEST", 100.0)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.bid == 0.0
    assert candidate.ask == 2.0
    assert candidate.iv == 0.30
    assert candidate.open_interest == 0
    assert candidate.volume == 0


def test_scan_ticker_uses_supplied_as_of_date_for_staleness(monkeypatch) -> None:
    dates = pd.date_range(end="2026-01-02", periods=205, freq="B")
    raw = pd.DataFrame({"Close": np.linspace(100.0, 120.0, len(dates))}, index=dates)
    monkeypatch.setattr(scanner.yf, "download", lambda *_args, **_kwargs: raw)

    reading = scanner.scan_ticker("TEST", as_of=date(2026, 1, 8))

    assert reading.error == ""
    assert reading.stale


def test_force_options_error_is_reported_without_aborting(monkeypatch, capsys) -> None:
    reading = scanner.Reading(
        ticker="SPY",
        price=100.0,
        prev_close=101.0,
        rsi2=20.0,
        sma200=90.0,
        sma5=99.0,
        above_trend=True,
        signal=False,
        flip_price=None,
        stale=False,
    )

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("upstream is unavailable")

    monkeypatch.setattr(scanner, "scan_ticker", lambda *_args, **_kwargs: reading)
    monkeypatch.setattr(scanner, "render", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scanner, "fetch_candidates", unavailable)
    monkeypatch.setattr(
        "sys.argv",
        ["rsi2_scanner.py", "--tickers", "SPY", "--force-options", "SPY", "--no-options"],
    )

    scanner.main()

    assert "options chain unavailable for SPY: RuntimeError" in capsys.readouterr().out


def test_default_alerts_are_core_only_and_fail_closed_before_the_close(monkeypatch, capsys) -> None:
    sent: list[list[str]] = []
    monkeypatch.setattr(
        scanner,
        "notify",
        lambda signals, provisional: sent.append([item.ticker for item in signals]),
    )
    non_core = scanner.Reading(
        ticker="CVS", price=100.0, prev_close=101.0, rsi2=1.0, sma200=90.0, sma5=99.0,
        above_trend=True, signal=True, flip_price=None, stale=False,
    )
    core = scanner.Reading(
        ticker="SPY", price=100.0, prev_close=101.0, rsi2=1.0, sma200=90.0, sma5=99.0,
        above_trend=True, signal=True, flip_price=None, stale=False,
    )
    after_close = datetime(2026, 8, 7, 16, 1, tzinfo=scanner.MARKET_TZ)
    scanner.render([non_core, core], 5.0, send_notifications=True, scan_started_at=after_close)
    assert sent == [["SPY"]]

    sent.clear()
    before_close = datetime(2026, 8, 7, 15, 55, tzinfo=scanner.MARKET_TZ)
    scanner.render([core], 5.0, send_notifications=True, scan_started_at=before_close)
    assert sent == []
    assert "Alerts suppressed: the scan began before the close" in capsys.readouterr().out


def test_squeeze_panel_is_informational_and_never_sends_a_notification(monkeypatch, capsys) -> None:
    sent: list[list[str]] = []
    monkeypatch.setattr(
        scanner,
        "notify",
        lambda signals, provisional: sent.append([item.ticker for item in signals]),
    )
    squeeze = SqueezeSignal(
        status="confirmed",
        price=105.0,
        squeeze_days=0,
        recent_squeeze_days=4,
        bandwidth_percentile=0.08,
        volume_ratio=1.6,
        breakout_level=103.0,
        above_trend=True,
        close_above_breakout=True,
    )
    reading = scanner.Reading(
        ticker="AAPL", price=105.0, prev_close=104.0, rsi2=45.0, sma200=95.0, sma5=104.0,
        above_trend=True, signal=False, flip_price=None, stale=False, squeeze=squeeze,
    )

    scanner.render(
        [reading], 5.0, send_notifications=True,
        scan_started_at=datetime(2026, 8, 7, 16, 1, tzinfo=scanner.MARKET_TZ),
    )

    output = capsys.readouterr().out
    assert "VOLATILITY-COMPRESSION WATCH" in output
    assert "SUPPRESSED" in output
    assert "order, or alert" in output
    assert sent == []


def test_confirmed_squeeze_reads_the_context_cache_and_fails_closed(tmp_path, capsys) -> None:
    squeeze = SqueezeSignal(
        status="confirmed",
        price=105.0,
        squeeze_days=0,
        recent_squeeze_days=4,
        bandwidth_percentile=0.08,
        volume_ratio=1.6,
        breakout_level=103.0,
        above_trend=True,
        close_above_breakout=True,
    )
    reading = scanner.Reading(
        ticker="AAPL", price=105.0, prev_close=104.0, rsi2=45.0, sma200=95.0, sma5=104.0,
        above_trend=True, signal=False, flip_price=None, stale=False, squeeze=squeeze,
    )
    after_close = datetime(2026, 8, 7, 16, 1, tzinfo=scanner.MARKET_TZ)

    scanner.apply_squeeze_context_gate([reading], context_dir=tmp_path, now=after_close)
    assert reading.squeeze_context is not None
    assert not reading.squeeze_context.eligible
    assert reading.squeeze_context.reasons == ("no market-context snapshot",)

    scanner.render([reading], 5.0, send_notifications=False, scan_started_at=after_close)

    output = capsys.readouterr().out
    assert "SUPPRESSED" in output
    assert "no market-context snapshot" in output


def test_confirmed_squeeze_is_suppressed_while_daily_price_is_provisional(capsys) -> None:
    squeeze = SqueezeSignal(
        status="confirmed",
        price=105.0,
        squeeze_days=0,
        recent_squeeze_days=4,
        bandwidth_percentile=0.08,
        volume_ratio=1.6,
        breakout_level=103.0,
        above_trend=True,
        close_above_breakout=True,
    )
    reading = scanner.Reading(
        ticker="AAPL", price=105.0, prev_close=104.0, rsi2=45.0, sma200=95.0, sma5=104.0,
        above_trend=True, signal=False, flip_price=None, stale=False, squeeze=squeeze,
        squeeze_context=scanner.ContextGate(True, ()),
    )

    scanner.render(
        [reading], 5.0, send_notifications=False,
        scan_started_at=datetime(2026, 8, 7, 15, 55, tzinfo=scanner.MARKET_TZ),
    )

    output = capsys.readouterr().out
    assert "SUPPRESSED" in output
    assert "daily price bar is provisional until the close" in output
