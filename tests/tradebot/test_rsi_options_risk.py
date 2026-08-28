"""Risk engine and quote gates. Both fail closed.

The $200 boundary is tested from both sides because a limit that is only tested where it passes
is not tested. The quote gates are tested for every way a feed can be wrong, because the pilot's
failure mode was reusing a stale mark rather than refusing to price the position.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tradebot.rsi_options.experiment import RiskLimits
from tradebot.rsi_options.quotes import (
    OptionQuote,
    validate_contract_terms,
    validate_quote,
)
from tradebot.rsi_options.risk import PortfolioSnapshot, ProposedTrade, RiskEngine
from tradebot.rsi_options.states import RejectReason

NOW = datetime(2026, 8, 20, 14, 30, tzinfo=timezone.utc)


@pytest.fixture
def engine() -> RiskEngine:
    return RiskEngine(RiskLimits())


@pytest.fixture
def empty_book() -> PortfolioSnapshot:
    return PortfolioSnapshot(cash=50_000.0, equity=50_000.0)


def trade(price: float, qty: int = 1, ticker: str = "SPY", sector: str = "Index",
          costs: float = 0.0) -> ProposedTrade:
    return ProposedTrade(ticker=ticker, sector=sector, option_symbol=f"{ticker}260918C",
                         price_per_contract=price, contract_multiplier=100,
                         quantity=qty, transaction_costs=costs)


class TestPremiumCapBoundary:
    def test_199_dollars_is_allowed(self, engine, empty_book):
        d = engine.evaluate(trade(1.99), empty_book, NOW)
        assert d.approved
        assert d.premium_at_risk == pytest.approx(199.0)

    def test_exactly_200_is_allowed(self, engine, empty_book):
        d = engine.evaluate(trade(2.00), empty_book, NOW)
        assert d.approved
        assert d.premium_at_risk == pytest.approx(200.0)

    def test_201_dollars_is_rejected(self, engine, empty_book):
        d = engine.evaluate(trade(2.01), empty_book, NOW)
        assert not d.approved
        assert d.rejection.reason is RejectReason.RISK_PREMIUM_EXCEEDED
        assert d.premium_at_risk == pytest.approx(201.0)

    def test_costs_count_toward_the_cap(self, engine, empty_book):
        """$199 of premium plus $2 of commission is $201 at risk, not $199."""
        d = engine.evaluate(trade(1.99, costs=2.0), empty_book, NOW)
        assert not d.approved
        assert d.rejection.reason is RejectReason.RISK_PREMIUM_EXCEEDED

    def test_expensive_contract_yields_zero_affordable_quantity(self, engine):
        """A $3,755 contract cannot be traded within a $200 cap at ANY size."""
        assert engine.max_affordable_quantity(37.55, 100) == 0

    def test_zero_quantity_is_rejected_not_silently_resized(self, engine, empty_book):
        d = engine.evaluate(trade(37.55, qty=0), empty_book, NOW)
        assert not d.approved
        assert d.rejection.reason is RejectReason.RISK_PREMIUM_EXCEEDED

    def test_limit_is_never_raised_to_fit_a_trade(self, engine, empty_book):
        before = engine.limits.max_premium_risk_per_trade
        engine.evaluate(trade(50.0), empty_book, NOW)
        assert engine.limits.max_premium_risk_per_trade == before


class TestOnePositionEnforced:
    def test_second_position_rejected(self, engine):
        book = PortfolioSnapshot(
            cash=49_800.0, equity=50_000.0,
            open_positions=({"ticker": "QQQ", "sector": "Index", "premium_at_risk": 180.0},))
        d = engine.evaluate(trade(1.50, ticker="SPY"), book, NOW)
        assert not d.approved
        assert d.rejection.reason is RejectReason.RISK_MAX_OPEN_POSITIONS

    def test_second_trade_same_session_rejected(self, engine):
        book = PortfolioSnapshot(cash=50_000.0, equity=50_000.0, trades_this_session=1)
        d = engine.evaluate(trade(1.50), book, NOW)
        assert not d.approved
        assert d.rejection.reason is RejectReason.RISK_MAX_TRADES_PER_SESSION


class TestDailyLossLimit:
    def test_reaching_the_limit_blocks_new_risk(self, engine):
        book = PortfolioSnapshot(cash=49_800.0, equity=49_800.0, realized_pnl_today=-200.0)
        d = engine.evaluate(trade(1.50), book, NOW)
        assert not d.approved
        assert d.rejection.reason is RejectReason.RISK_DAILY_LOSS_REACHED

    def test_below_the_limit_still_allowed(self, engine):
        book = PortfolioSnapshot(cash=49_850.0, equity=49_850.0, realized_pnl_today=-150.0)
        assert engine.evaluate(trade(1.50), book, NOW).approved


class TestConcentrationAndCash:
    def test_insufficient_cash_rejected(self, engine):
        book = PortfolioSnapshot(cash=100.0, equity=100.0)
        d = engine.evaluate(trade(1.50), book, NOW)
        assert not d.approved
        assert d.rejection.reason is RejectReason.INSUFFICIENT_CASH

    def test_unknown_multiplier_is_no_trade(self, engine, empty_book):
        t = ProposedTrade(ticker="SPY", sector="Index", option_symbol="X",
                          price_per_contract=1.0, contract_multiplier=0, quantity=1)
        d = engine.evaluate(t, empty_book, NOW)
        assert not d.approved
        assert d.rejection.reason is RejectReason.CONTRACT_MULTIPLIER_UNKNOWN


# --------------------------------------------------------------------------- quotes
def quote(**kw) -> OptionQuote:
    base = dict(symbol="SPY260918C00500000", underlying="SPY", expiry="2026-09-18",
                strike=500.0, bid=2.00, ask=2.10, bid_size=50, ask_size=50,
                quote_timestamp=NOW - timedelta(seconds=30), data_received_timestamp=NOW,
                open_interest=5000, delta=0.65, contract_multiplier=100)
    base.update(kw)
    return OptionQuote(**base)


GATES = dict(max_age_seconds=900, max_spread_pct=10.0, min_open_interest=100)


class TestQuoteGates:
    def test_good_quote_passes(self):
        assert validate_quote(quote(), now=NOW, **GATES) is None

    def test_missing_quote_rejected(self):
        r = validate_quote(None, now=NOW, **GATES)
        assert r.reason is RejectReason.QUOTE_MISSING

    def test_stale_quote_rejected(self):
        q = quote(quote_timestamp=NOW - timedelta(hours=3))
        r = validate_quote(q, now=NOW, **GATES)
        assert r.reason is RejectReason.QUOTE_STALE

    def test_no_timestamp_rejected(self):
        r = validate_quote(quote(quote_timestamp=None), now=NOW, **GATES)
        assert r.reason is RejectReason.QUOTE_INVALID_TIMESTAMP

    def test_future_timestamp_rejected(self):
        q = quote(quote_timestamp=NOW + timedelta(minutes=5))
        r = validate_quote(q, now=NOW, **GATES)
        assert r.reason is RejectReason.QUOTE_INVALID_TIMESTAMP

    def test_zero_bid_rejected(self):
        r = validate_quote(quote(bid=0.0), now=NOW, **GATES)
        assert r.reason is RejectReason.QUOTE_ZERO_BID

    def test_zero_ask_rejected(self):
        r = validate_quote(quote(ask=0.0), now=NOW, **GATES)
        assert r.reason is RejectReason.QUOTE_ZERO_ASK

    def test_crossed_book_rejected(self):
        r = validate_quote(quote(bid=2.50, ask=2.00), now=NOW, **GATES)
        assert r.reason is RejectReason.QUOTE_CROSSED

    def test_wide_spread_rejected(self):
        r = validate_quote(quote(bid=1.00, ask=2.00), now=NOW, **GATES)
        assert r.reason is RejectReason.QUOTE_SPREAD_TOO_WIDE

    def test_low_open_interest_rejected(self):
        r = validate_quote(quote(open_interest=10), now=NOW, **GATES)
        assert r.reason is RejectReason.CONTRACT_LOW_OPEN_INTEREST

    def test_missing_size_rejected_when_required(self):
        r = validate_quote(quote(bid_size=None), now=NOW, require_size=True, **GATES)
        assert r.reason is RejectReason.QUOTE_NO_SIZE

    def test_adjusted_contract_rejected(self):
        r = validate_quote(quote(is_adjusted=True), now=NOW, **GATES)
        assert r.reason is RejectReason.CONTRACT_ADJUSTED

    def test_unknown_multiplier_rejected(self):
        r = validate_quote(quote(contract_multiplier=None), now=NOW, **GATES)
        assert r.reason is RejectReason.CONTRACT_MULTIPLIER_UNKNOWN


class TestContractTerms:
    TERMS = dict(min_dte=21, max_dte=60, min_delta=0.55, max_delta=0.85)

    def test_valid_contract_passes(self):
        assert validate_contract_terms(quote(), as_of=NOW, **self.TERMS) is None

    def test_expired_contract_rejected(self):
        q = quote(expiry="2026-08-01")
        r = validate_contract_terms(q, as_of=NOW, **self.TERMS)
        assert r.reason is RejectReason.CONTRACT_EXPIRED

    def test_too_short_dted_rejected(self):
        q = quote(expiry="2026-08-28")
        r = validate_contract_terms(q, as_of=NOW, **self.TERMS)
        assert r.reason is RejectReason.CONTRACT_DTE_OUT_OF_RANGE

    def test_delta_out_of_range_rejected(self):
        r = validate_contract_terms(quote(delta=0.20), as_of=NOW, **self.TERMS)
        assert r.reason is RejectReason.CONTRACT_DELTA_OUT_OF_RANGE

    def test_missing_delta_is_no_trade(self):
        r = validate_contract_terms(quote(delta=None), as_of=NOW, **self.TERMS)
        assert r.reason is RejectReason.CONTRACT_METADATA_MISSING


class TestStateMachine:
    def test_legal_path(self):
        from tradebot.rsi_options.states import SignalState, assert_transition

        assert_transition(SignalState.SIGNAL_DETECTED, SignalState.PENDING_ENTRY)
        assert_transition(SignalState.PENDING_ENTRY, SignalState.ORDER_SUBMITTED)
        assert_transition(SignalState.ORDER_SUBMITTED, SignalState.FILLED)

    def test_cannot_skip_straight_to_filled(self):
        from tradebot.rsi_options.states import (
            IllegalTransition,
            SignalState,
            assert_transition,
        )

        with pytest.raises(IllegalTransition):
            assert_transition(SignalState.SIGNAL_DETECTED, SignalState.FILLED)

    def test_terminal_states_are_terminal(self):
        from tradebot.rsi_options.states import (
            IllegalTransition,
            SignalState,
            assert_transition,
        )

        with pytest.raises(IllegalTransition):
            assert_transition(SignalState.FILLED, SignalState.PENDING_ENTRY)


class TestExperimentSafety:
    def test_default_status_is_development(self):
        from tradebot.rsi_options.experiment import (
            ExperimentState,
            ExperimentStatus,
            default_config,
        )

        st = ExperimentState(config=default_config(("SPY",)))
        assert st.status is ExperimentStatus.DEVELOPMENT
        assert st.official_start_timestamp is None

    def test_cannot_activate_from_development(self):
        from tradebot.rsi_options.experiment import ExperimentState, default_config

        st = ExperimentState(config=default_config(("SPY",)))
        with pytest.raises(RuntimeError, match="READY_FOR_PAPER"):
            st.activate()

    def test_start_timestamp_cannot_be_rewritten(self):
        from tradebot.rsi_options.experiment import (
            ExperimentState,
            ExperimentStatus,
            default_config,
        )

        st = ExperimentState(config=default_config(("SPY",)),
                             status=ExperimentStatus.READY_FOR_PAPER)
        st.activate()
        first = st.official_start_timestamp
        with pytest.raises(RuntimeError):
            st.activate()
        assert st.official_start_timestamp == first

    def test_frozen_config_change_is_detected(self):
        from dataclasses import replace

        from tradebot.rsi_options.experiment import (
            ConfigFrozenError,
            ExperimentState,
            ExperimentStatus,
            RiskLimits,
            default_config,
        )

        st = ExperimentState(config=default_config(("SPY",)),
                             status=ExperimentStatus.READY_FOR_PAPER)
        st.activate()
        st.config = replace(st.config, risk=RiskLimits(max_premium_risk_per_trade=1000.0))
        with pytest.raises(ConfigFrozenError):
            st.check_not_tampered()

    def test_ml_execution_disabled_in_formal_config(self):
        from tradebot.rsi_options.experiment import default_config

        cfg = default_config(("SPY",))
        assert cfg.ml_enabled_for_execution is False
        assert cfg.validate() == []

    def test_live_mode_refused(self, monkeypatch):
        from tradebot.rsi_options.experiment import (
            UnsupportedTradingMode,
            require_paper_mode,
        )

        monkeypatch.setenv("TRADING_MODE", "LIVE")
        with pytest.raises(UnsupportedTradingMode):
            require_paper_mode()

    def test_paper_mode_accepted(self, monkeypatch):
        from tradebot.rsi_options.experiment import require_paper_mode

        monkeypatch.setenv("TRADING_MODE", "PAPER")
        require_paper_mode()
