"""M1: value types, contract registry, configuration.

The contract-spec test is the most important one in this file. A wrong multiplier scales
every P&L, position size and risk limit in the project while leaving output that still
looks entirely plausible, so the published CME numbers are pinned here.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tradebot.config import Config, ConfigError, load_config
from tradebot.core.clock import MARKET_TZ, SimulatedClock, SystemClock, ensure_aware
from tradebot.core.models import Bar, Fill, Order, OrderIntent, Position, Trade
from tradebot.core.types import ExitReason, OrderStatus, OrderType, Side, TradingMode
from tradebot.instruments.registry import (
    UnknownInstrument,
    get_instrument,
    known_instruments,
    register_instrument,
)

ET = MARKET_TZ


def ts(hour: int = 10, minute: int = 0) -> datetime:
    return datetime(2026, 3, 10, hour, minute, tzinfo=ET)


# ---------------------------------------------------------------------------- instruments


def test_mnq_matches_the_published_cme_contract_specification():
    mnq = get_instrument("MNQ")
    assert mnq.multiplier == 2.0
    assert mnq.tick_size == 0.25
    assert mnq.tick_value == 0.50
    assert mnq.contract_months == "HMUZ"


def test_nq_and_mnq_differ_only_by_the_multiplier():
    nq, mnq = get_instrument("NQ"), get_instrument("MNQ")
    assert nq.tick_size == mnq.tick_size
    assert nq.multiplier == 10 * mnq.multiplier


def test_one_point_on_mnq_is_two_dollars_per_contract():
    mnq = get_instrument("MNQ")
    assert mnq.points_to_usd(1.0) == 2.0
    assert mnq.points_to_usd(10.0, quantity=3) == 60.0
    assert mnq.usd_to_points(60.0, quantity=3) == 10.0


def test_lookup_is_case_insensitive_and_unknown_symbols_raise():
    assert get_instrument("mnq").symbol == "MNQ"
    with pytest.raises(UnknownInstrument):
        get_instrument("BTC")
    assert "MNQ" in known_instruments()


def test_registering_over_a_known_instrument_is_refused():
    spec = get_instrument("MNQ")
    with pytest.raises(ValueError, match="already registered"):
        register_instrument(spec)


def test_prices_snap_to_the_tick_grid():
    mnq = get_instrument("MNQ")
    assert mnq.round_to_tick(18234.1873) == 18234.25
    assert mnq.round_to_tick(18234.1) == 18234.0
    assert mnq.round_to_tick(18234.125) in (18234.0, 18234.25)  # banker's rounding at .5


def test_directional_rounding_never_moves_a_stop_closer_to_the_entry():
    mnq = get_instrument("MNQ")
    # A long's stop sits below entry, so it must round DOWN (further away).
    assert mnq.round_away(18234.1873, direction=-1) == 18234.0
    # A short's stop sits above entry, so it must round UP.
    assert mnq.round_away(18234.1873, direction=+1) == 18234.25


# ---------------------------------------------------------------------------- core models


def test_a_bar_with_inconsistent_ohlc_cannot_be_constructed():
    with pytest.raises(ValueError, match="high < low"):
        Bar(ts(), open=100, high=99, low=101, close=100, volume=1)
    with pytest.raises(ValueError, match="open outside"):
        Bar(ts(), open=105, high=101, low=99, close=100, volume=1)
    with pytest.raises(ValueError, match="close outside"):
        Bar(ts(), open=100, high=101, low=99, close=105, volume=1)
    with pytest.raises(ValueError, match="negative volume"):
        Bar(ts(), open=100, high=101, low=99, close=100, volume=-1)


def test_a_naive_timestamp_is_rejected_rather_than_assumed():
    with pytest.raises(ValueError, match="timezone-aware"):
        Bar(datetime(2026, 3, 10, 10, 0), open=100, high=101, low=99, close=100, volume=1)


def test_ensure_aware_rejects_naive_and_converts_aware():
    utc_noon = datetime(2026, 3, 10, 16, 0, tzinfo=ZoneInfo("UTC"))
    assert ensure_aware(utc_noon).hour == 12  # 16:00 UTC is 12:00 EDT
    with pytest.raises(ValueError, match="naive datetime"):
        ensure_aware(datetime(2026, 3, 10, 12, 0))


def test_an_intent_must_carry_a_stop_on_the_correct_side():
    OrderIntent(
        timestamp=ts(), instrument="MNQ", side=Side.BUY, strategy="t",
        stop_price=99.0, reference_price=100.0,
    )
    with pytest.raises(ValueError, match="long stop"):
        OrderIntent(
            timestamp=ts(), instrument="MNQ", side=Side.BUY, strategy="t",
            stop_price=101.0, reference_price=100.0,
        )
    with pytest.raises(ValueError, match="short stop"):
        OrderIntent(
            timestamp=ts(), instrument="MNQ", side=Side.SELL, strategy="t",
            stop_price=99.0, reference_price=100.0,
        )


def test_an_intent_is_inert_and_exposes_no_way_to_send_itself():
    intent = OrderIntent(
        timestamp=ts(), instrument="MNQ", side=Side.BUY, strategy="t",
        stop_price=99.0, reference_price=100.0,
    )
    forbidden = {"send", "submit", "execute", "place", "broker", "client"}
    assert forbidden.isdisjoint(dir(intent))


def test_intent_fingerprints_ignore_identity_and_features():
    kwargs = dict(
        timestamp=ts(), instrument="MNQ", side=Side.BUY, strategy="t",
        stop_price=99.0, reference_price=100.0,
    )
    a = OrderIntent(**kwargs, features={"atr": 1.0})
    b = OrderIntent(**kwargs, features={"atr": 2.0})
    assert a.intent_id != b.intent_id
    assert a.fingerprint() == b.fingerprint()


def test_order_binding_fields_change_with_economics_but_not_with_lifecycle():
    order = Order(
        order_id="o1", timestamp=ts(), instrument="MNQ", side=Side.BUY,
        quantity=2, order_type=OrderType.MARKET,
    )
    assert order.with_status(OrderStatus.ACCEPTED, broker_order_id="X").binding_fields() == (
        order.binding_fields()
    )
    bigger = Order(
        order_id="o1", timestamp=ts(), instrument="MNQ", side=Side.BUY,
        quantity=3, order_type=OrderType.MARKET,
    )
    assert bigger.binding_fields() != order.binding_fields()


def test_orders_reject_impossible_shapes():
    with pytest.raises(ValueError, match="quantity must be positive"):
        Order(order_id="o", timestamp=ts(), instrument="MNQ", side=Side.BUY,
              quantity=0, order_type=OrderType.MARKET)
    with pytest.raises(ValueError, match="requires a limit_price"):
        Order(order_id="o", timestamp=ts(), instrument="MNQ", side=Side.BUY,
              quantity=1, order_type=OrderType.LIMIT)
    with pytest.raises(ValueError, match="requires a stop_price"):
        Order(order_id="o", timestamp=ts(), instrument="MNQ", side=Side.BUY,
              quantity=1, order_type=OrderType.STOP)
    with pytest.raises(ValueError, match="unknown order purpose"):
        Order(order_id="o", timestamp=ts(), instrument="MNQ", side=Side.BUY,
              quantity=1, order_type=OrderType.MARKET, purpose="BYPASS_RISK")


def test_fills_reject_nonpositive_quantity_and_price():
    with pytest.raises(ValueError):
        Fill(fill_id="f", order_id="o", timestamp=ts(), instrument="MNQ",
             side=Side.BUY, quantity=0, price=100.0)
    with pytest.raises(ValueError):
        Fill(fill_id="f", order_id="o", timestamp=ts(), instrument="MNQ",
             side=Side.BUY, quantity=1, price=0.0)
    with pytest.raises(ValueError, match="adverse slippage"):
        Fill(fill_id="f", order_id="o", timestamp=ts(), instrument="MNQ",
             side=Side.BUY, quantity=1, price=100.0, slippage_points=-0.25)


def test_position_pnl_arithmetic_uses_the_instrument_multiplier():
    mnq = get_instrument("MNQ")
    long = Position(
        instrument="MNQ", side=Side.BUY, quantity=2, entry_price=18000.0,
        entry_time=ts(), strategy="t", initial_stop=17990.0, stop_price=17990.0,
        risk_per_contract_points=10.0,
    )
    assert long.unrealized_points(18010.0) == 10.0
    assert long.unrealized_usd(18010.0, mnq.multiplier) == 40.0  # 10 pts * 2 ct * $2
    assert long.risk_usd(mnq.multiplier) == 40.0
    assert long.r_multiple_at(18010.0, mnq.multiplier) == 1.0

    short = Position(
        instrument="MNQ", side=Side.SELL, quantity=1, entry_price=18000.0,
        entry_time=ts(), strategy="t", initial_stop=18010.0, stop_price=18010.0,
        risk_per_contract_points=10.0,
    )
    assert short.unrealized_points(17990.0) == 10.0
    assert short.r_multiple_at(17990.0, mnq.multiplier) == 1.0
    assert short.signed_quantity == -1


def test_position_excursions_track_the_favourable_and_adverse_extremes():
    pos = Position(
        instrument="MNQ", side=Side.BUY, quantity=1, entry_price=100.0,
        entry_time=ts(), strategy="t", initial_stop=95.0, stop_price=95.0,
    )
    pos.observe(Bar(ts(10, 1), open=100, high=104, low=97, close=101, volume=1))
    pos.observe(Bar(ts(10, 2), open=101, high=102, low=96, close=99, volume=1))
    assert pos.bars_held == 2
    assert pos.max_favorable_price == 104
    assert pos.max_adverse_price == 96


def test_trade_reports_winner_and_duration():
    trade = Trade(
        trade_id="t1", instrument="MNQ", strategy="s", side=Side.BUY, quantity=1,
        entry_time=ts(10, 0), entry_price=100.0,
        exit_time=ts(10, 30), exit_price=105.0, exit_reason=ExitReason.TAKE_PROFIT,
        gross_pnl_usd=10.0, commission_usd=1.24, net_pnl_usd=8.76,
        r_multiple=1.5, bars_held=30, initial_stop=97.0, slippage_usd=0.50,
    )
    assert trade.is_winner
    assert trade.duration_seconds == 1800
    assert trade.slippage_usd == 0.50


def test_order_status_partitions_into_working_and_terminal():
    assert OrderStatus.FILLED.is_terminal and not OrderStatus.FILLED.is_working
    assert OrderStatus.ACCEPTED.is_working and not OrderStatus.ACCEPTED.is_terminal
    assert OrderStatus.PARTIALLY_FILLED.is_working


def test_side_sign_and_opposite():
    assert int(Side.BUY) == 1 and int(Side.SELL) == -1
    assert Side.BUY.opposite is Side.SELL
    assert Side.BUY.is_long and not Side.SELL.is_long


# ---------------------------------------------------------------------------- clock


def test_simulated_clock_only_moves_when_moved():
    clock = SimulatedClock(ts(9, 30))
    assert clock.now() == ts(9, 30)
    clock.advance(timedelta(minutes=5))
    assert clock.now() == ts(9, 35)
    clock.set(ts(15, 58))
    assert clock.now() == ts(15, 58)
    with pytest.raises(ValueError, match="timezone-aware"):
        clock.set(datetime(2026, 3, 10, 10, 0))


def test_system_clock_is_market_time_aware():
    assert SystemClock().now().tzinfo is not None


# ---------------------------------------------------------------------------- config


def test_the_shipped_config_loads_and_validates():
    cfg = load_config()
    assert cfg.mode is TradingMode.BACKTEST
    assert cfg.deployment.stage == 0
    assert cfg.prop_firm.phase == "evaluation"
    assert cfg.prop_firm.internal_safety_buffer_usd == 400.0
    assert cfg.instrument == "MNQ"
    assert cfg.bar_seconds == 60.0
    assert cfg.risk.per_trade.max_contracts >= cfg.risk.per_trade.min_contracts
    assert cfg.risk.per_trade.max_risk_per_trade_usd == 200.0
    assert cfg.risk.daily.max_daily_loss_usd == 200.0
    assert cfg.risk.daily.max_trades_per_day == 1
    assert cfg.risk.max_open_positions == 1


def test_a_typo_in_a_risk_limit_is_an_error_not_a_silent_default(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("risk:\n  daily:\n    max_daily_loss_used: 500\n", encoding="utf-8")
    with pytest.raises(ConfigError) as err:
        load_config(path)
    assert "max_daily_loss_used" in str(err.value)
    assert "risk.daily" in str(err.value)


@pytest.mark.parametrize(
    "yaml_text, fragment",
    [
        ("risk:\n  daily:\n    max_daily_loss_usd: -100\n", "risk.daily.max_daily_loss_usd"),
        ("risk:\n  per_trade:\n    max_contracts: 0\n", "risk.per_trade.max_contracts"),
        ("risk:\n  max_open_positions: 0\n", "risk.max_open_positions"),
        ("prop_firm:\n  internal_safety_buffer_usd: 0\n", "prop_firm.internal_safety_buffer_usd"),
        ("deployment:\n  stage: 3\n", "requires a separately verified human approval"),
        ("session:\n  rth_start: '17:00'\n", "session.rth_start"),
        ("dashboard:\n  port: 99999\n", "dashboard.port"),
        ("timeframe: 7min\n", "timeframe"),
    ],
)
def test_invalid_values_raise_with_the_offending_key_path(tmp_path, yaml_text, fragment):
    path = tmp_path / "c.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError) as err:
        load_config(path)
    assert fragment in str(err.value)


def test_live_mode_is_refused_at_load_time(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("mode: LIVE\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not implemented"):
        load_config(path)


def test_the_dashboard_cannot_be_bound_to_a_routable_interface(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("dashboard:\n  host: 0.0.0.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="loopback"):
        load_config(path)


def test_environment_overrides_reach_nested_keys(monkeypatch, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("risk:\n  daily:\n    max_daily_loss_usd: 1000\n", encoding="utf-8")
    monkeypatch.setenv("TRADEBOT__RISK__DAILY__MAX_DAILY_LOSS_USD", "250")
    assert load_config(path).risk.daily.max_daily_loss_usd == 250.0


def test_production_pinned_environment_can_only_tighten_risk(monkeypatch, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "risk:\n"
        "  daily:\n"
        "    max_daily_loss_usd: 200\n"
        "prop_firm:\n"
        "  internal_safety_buffer_usd: 400\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADEBOT__RISK__DAILY__MAX_DAILY_LOSS_USD", "150")
    monkeypatch.setenv("TRADEBOT__PROP_FIRM__INTERNAL_SAFETY_BUFFER_USD", "500")

    cfg = load_config(path, production_pinned=True)

    assert cfg.risk.daily.max_daily_loss_usd == 150.0
    assert cfg.prop_firm.internal_safety_buffer_usd == 500.0


@pytest.mark.parametrize(
    ("key", "value", "fragment"),
    [
        (
            "TRADEBOT__RISK__PER_TRADE__MAX_RISK_PER_TRADE_USD",
            "201",
            "weakens the pinned limit",
        ),
        (
            "TRADEBOT__PROP_FIRM__INTERNAL_SAFETY_BUFFER_USD",
            "399",
            "weakens the pinned safeguard",
        ),
        (
            "TRADEBOT__BROKER__ADAPTER",
            "tradovate",
            "not an approved risk-tightening key",
        ),
        (
            "TRADEBOT__DEPLOYMENT__STAGE",
            "2",
            "not an approved risk-tightening key",
        ),
    ],
)
def test_production_pinned_environment_refuses_weaker_or_identity_changes(
    monkeypatch, tmp_path, key, value, fragment
):
    path = tmp_path / "c.yaml"
    path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv(key, value)

    with pytest.raises(ConfigError, match=fragment):
        load_config(path, production_pinned=True)


def test_production_pinned_config_refuses_programmatic_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="does not accept programmatic overrides"):
        load_config(
            path,
            overrides={"risk": {"daily": {"max_daily_loss_usd": 100.0}}},
            production_pinned=True,
        )


def test_defaults_alone_produce_a_valid_configuration():
    cfg = Config()
    cfg.validate()
    assert cfg.risk.per_trade.max_risk_per_trade_usd == 200.0
    assert cfg.risk.daily.max_daily_loss_usd == 200.0
    assert cfg.risk.daily.max_trades_per_day == 1
    assert cfg.risk.max_open_positions == 1
