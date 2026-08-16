"""Fail-closed market-context snapshots for the squeeze watch.

Daily OHLCV can describe a volatility compression, but it cannot establish a
short-squeeze thesis.  This module keeps the non-price inputs separate, dated,
and auditable.  A setup is *data-qualified* only when every required snapshot
is present and fresh; missing data suppresses the setup instead of being
silently treated as favourable.

Yahoo is used only as a convenience source for current earnings and option
chain snapshots.  It does not provide a reliable, dated borrow feed or a
point-in-time short-interest history.  Those fields must be supplied through
the documented override file from a licensed/broker source.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


MARKET_CONTEXT_VERSION = 1
DEFAULT_CONTEXT_DIR = Path("data_cache/market_context")


def _parse_bool(value: Any, field: str) -> bool:
    """Accept JSON booleans (and unambiguous CLI-style spellings) only."""

    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"invalid {field}; use true or false")


@dataclass(frozen=True, slots=True)
class MarketContext:
    """A dated snapshot of non-price inputs for one equity ticker."""

    ticker: str
    retrieved_at: datetime
    short_interest_pct_float: float | None = None
    days_to_cover: float | None = None
    short_interest_as_of: date | None = None
    short_interest_source: str = ""
    short_interest_verified: bool = False
    borrow_available_shares: int | None = None
    borrow_fee_rate_pct: float | None = None
    borrow_as_of: datetime | None = None
    borrow_source: str = ""
    borrow_verified: bool = False
    next_earnings_date: date | None = None
    earnings_as_of: datetime | None = None
    earnings_source: str = ""
    earnings_verified: bool = False
    options_as_of: datetime | None = None
    option_call_open_interest: int | None = None
    option_put_open_interest: int | None = None
    options_source: str = ""
    options_verified: bool = False
    provider_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["version"] = MARKET_CONTEXT_VERSION
        for key in ("retrieved_at", "borrow_as_of", "earnings_as_of", "options_as_of"):
            value = data[key]
            data[key] = value.isoformat() if value else None
        for key in ("short_interest_as_of", "next_earnings_date"):
            value = data[key]
            data[key] = value.isoformat() if value else None
        data["provider_errors"] = list(self.provider_errors)
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MarketContext":
        def parse_datetime(value: Any) -> datetime | None:
            if value in (None, ""):
                return None
            parsed = pd.Timestamp(value)
            if parsed.tzinfo is None:
                parsed = parsed.tz_localize("UTC")
            return parsed.to_pydatetime()

        def parse_date(value: Any) -> date | None:
            if value in (None, ""):
                return None
            return pd.Timestamp(value).date()

        ticker = str(raw.get("ticker") or "").strip().upper()
        if not ticker:
            raise ValueError("market-context snapshot has no ticker")
        retrieved_at = parse_datetime(raw.get("retrieved_at"))
        if retrieved_at is None:
            raise ValueError("market-context snapshot has no retrieved_at timestamp")

        float_fields = ("short_interest_pct_float", "days_to_cover", "borrow_fee_rate_pct")
        values: dict[str, Any] = {}
        for name in float_fields:
            value = raw.get(name)
            if value in (None, ""):
                values[name] = None
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid {name}") from exc
            values[name] = parsed if isfinite(parsed) else None

        shares = raw.get("borrow_available_shares")
        if shares in (None, ""):
            values["borrow_available_shares"] = None
        else:
            try:
                values["borrow_available_shares"] = int(shares)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid borrow_available_shares") from exc

        for name in ("option_call_open_interest", "option_put_open_interest"):
            value = raw.get(name)
            if value in (None, ""):
                values[name] = None
            else:
                try:
                    values[name] = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid {name}") from exc

        return cls(
            ticker=ticker,
            retrieved_at=retrieved_at,
            short_interest_as_of=parse_date(raw.get("short_interest_as_of")),
            borrow_as_of=parse_datetime(raw.get("borrow_as_of")),
            next_earnings_date=parse_date(raw.get("next_earnings_date")),
            earnings_as_of=parse_datetime(raw.get("earnings_as_of")),
            options_as_of=parse_datetime(raw.get("options_as_of")),
            short_interest_source=str(raw.get("short_interest_source") or "").strip(),
            short_interest_verified=_parse_bool(raw.get("short_interest_verified"), "short_interest_verified"),
            borrow_source=str(raw.get("borrow_source") or "").strip(),
            borrow_verified=_parse_bool(raw.get("borrow_verified"), "borrow_verified"),
            earnings_source=str(raw.get("earnings_source") or "").strip(),
            earnings_verified=_parse_bool(raw.get("earnings_verified"), "earnings_verified"),
            options_source=str(raw.get("options_source") or "").strip(),
            options_verified=_parse_bool(raw.get("options_verified"), "options_verified"),
            provider_errors=tuple(str(item) for item in raw.get("provider_errors", ()) if item),
            **values,
        )


@dataclass(frozen=True, slots=True)
class ContextGateConfig:
    """Freshness/risk rules, not unvalidated alpha thresholds."""

    earnings_blackout_days: int = 10
    max_short_interest_age_days: int = 21
    max_borrow_age_hours: int = 24
    max_earnings_age_hours: int = 24
    max_options_age_hours: int = 24


@dataclass(frozen=True, slots=True)
class ContextGate:
    eligible: bool
    reasons: tuple[str, ...]

    @property
    def label(self) -> str:
        return "QUALIFIED" if self.eligible else "SUPPRESSED"


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _finite_nonnegative(value: float | int | None) -> bool:
    return value is not None and isfinite(float(value)) and float(value) >= 0.0


def _fresh_timestamp(value: datetime, now: datetime, maximum_age: timedelta) -> bool:
    """Fresh timestamps cannot be in the future or older than the stated policy."""

    age = now - _as_utc(value)
    return timedelta(0) <= age <= maximum_age


def evaluate_context(
    context: MarketContext | None,
    *,
    now: datetime | None = None,
    config: ContextGateConfig = ContextGateConfig(),
) -> ContextGate:
    """Return a fail-closed result for a squeeze *data* qualification."""

    if context is None:
        return ContextGate(False, ("no market-context snapshot",))
    instant = _as_utc(now or datetime.now(timezone.utc))
    today = instant.date()
    reasons: list[str] = []

    if _as_utc(context.retrieved_at) > instant:
        reasons.append("market-context snapshot timestamp is in the future")

    if (
        not _finite_nonnegative(context.short_interest_pct_float)
        or context.short_interest_pct_float > 1.0
        or not _finite_nonnegative(context.days_to_cover)
    ):
        reasons.append("short-interest percentage or days-to-cover missing")
    if context.short_interest_as_of is None or not context.short_interest_source or not context.short_interest_verified:
        reasons.append("verified, dated short-interest source missing")
    elif not 0 <= (today - context.short_interest_as_of).days <= config.max_short_interest_age_days:
        reasons.append("short-interest snapshot is stale")

    if (
        not _finite_nonnegative(context.borrow_available_shares)
        or not _finite_nonnegative(context.borrow_fee_rate_pct)
        or context.borrow_as_of is None
        or not context.borrow_source
        or not context.borrow_verified
    ):
        reasons.append("verified borrow availability/fee snapshot missing")
    elif not _fresh_timestamp(
        context.borrow_as_of,
        instant,
        timedelta(hours=config.max_borrow_age_hours),
    ):
        reasons.append("borrow snapshot is stale")

    if (
        context.next_earnings_date is None
        or context.earnings_as_of is None
        or not context.earnings_source
        or not context.earnings_verified
    ):
        reasons.append("verified next earnings date missing")
    else:
        if not _fresh_timestamp(
            context.earnings_as_of,
            instant,
            timedelta(hours=config.max_earnings_age_hours),
        ):
            reasons.append("earnings-calendar snapshot is stale")
        days_until = (context.next_earnings_date - today).days
        if days_until < 0:
            reasons.append("earnings date is stale")
        elif days_until <= config.earnings_blackout_days:
            reasons.append(f"earnings within {config.earnings_blackout_days} calendar days")

    call_oi, put_oi = context.option_call_open_interest, context.option_put_open_interest
    if (
        context.options_as_of is None
        or not context.options_source
        or not _finite_nonnegative(call_oi)
        or not _finite_nonnegative(put_oi)
        or call_oi + put_oi <= 0
        or not context.options_verified
    ):
        reasons.append("verified option-chain open-interest snapshot missing")
    elif not _fresh_timestamp(
        context.options_as_of,
        instant,
        timedelta(hours=config.max_options_age_hours),
    ):
        reasons.append("option-chain snapshot is stale")

    return ContextGate(not reasons, tuple(reasons))


def context_path(ticker: str, directory: str | Path = DEFAULT_CONTEXT_DIR) -> Path:
    return Path(directory) / f"{ticker.upper().replace('-', '')}.json"


def load_context(ticker: str, directory: str | Path = DEFAULT_CONTEXT_DIR) -> MarketContext | None:
    path = context_path(ticker, directory)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        context = MarketContext.from_dict(raw)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid context file {path.name}: {type(exc).__name__}") from exc
    if context.ticker != ticker.upper():
        raise ValueError(f"context file {path.name} belongs to {context.ticker}, not {ticker.upper()}")
    return context


def save_context(context: MarketContext, directory: str | Path = DEFAULT_CONTEXT_DIR) -> Path:
    path = context_path(context.ticker, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(context.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def merge_override(context: MarketContext, override: dict[str, Any]) -> MarketContext:
    """Merge a licensed/broker snapshot without accepting unknown fields."""

    allowed = {field.name for field in fields(MarketContext)} - {"ticker", "retrieved_at", "provider_errors"}
    unknown = sorted(set(override).difference(allowed | {"ticker", "version"}))
    if unknown:
        raise ValueError(f"unknown market-context override fields: {', '.join(unknown)}")
    if "ticker" in override and str(override["ticker"]).upper() != context.ticker:
        raise ValueError("override ticker does not match snapshot ticker")

    # A Yahoo convenience value must never become 'verified' merely because a user flips a
    # boolean.  A qualifying override has to bring its own dated values and source for the
    # component it vouches for.  This is intentionally strict: a partial override is useful
    # for display, but it cannot turn into a signal.
    verification_requirements = {
        "short_interest_verified": (
            "short_interest_pct_float",
            "days_to_cover",
            "short_interest_as_of",
            "short_interest_source",
        ),
        "borrow_verified": (
            "borrow_available_shares",
            "borrow_fee_rate_pct",
            "borrow_as_of",
            "borrow_source",
        ),
        "earnings_verified": (
            "next_earnings_date",
            "earnings_as_of",
            "earnings_source",
        ),
        "options_verified": (
            "options_as_of",
            "option_call_open_interest",
            "option_put_open_interest",
            "options_source",
        ),
    }
    for verified_field, required_fields in verification_requirements.items():
        if verified_field in override and _parse_bool(override[verified_field], verified_field):
            missing = [field for field in required_fields if override.get(field) in (None, "")]
            if missing:
                raise ValueError(
                    f"{verified_field}=true requires same-source fields: {', '.join(missing)}"
                )
    merged = context.to_dict()
    for key, value in override.items():
        if key in allowed and value not in (None, ""):
            merged[key] = value
    return MarketContext.from_dict(merged)


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _calendar_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, pd.Index)):
        candidates = [_calendar_date(item) for item in value]
        candidates = [item for item in candidates if item is not None]
        return min(candidates) if candidates else None
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError):
        return None


class YahooContextProvider:
    """Fetch only current Yahoo convenience fields; never invents borrow data."""

    def fetch(self, ticker: str, *, now: datetime | None = None) -> MarketContext:
        instant = _as_utc(now or datetime.now(timezone.utc))
        errors: list[str] = []
        short_pct = days_to_cover = None
        earnings_date: date | None = None
        call_oi = put_oi = None
        try:
            instrument = yf.Ticker(ticker)
            try:
                info = instrument.info or {}
                short_pct = _finite_float(info.get("shortPercentOfFloat"))
                days_to_cover = _finite_float(info.get("shortRatio"))
            except Exception as exc:
                errors.append(f"short fields: {type(exc).__name__}")

            try:
                calendar = instrument.calendar
                if isinstance(calendar, dict):
                    earnings_date = _calendar_date(calendar.get("Earnings Date"))
                if earnings_date is None:
                    dates = instrument.earnings_dates
                    if dates is not None and not dates.empty:
                        future = [timestamp for timestamp, row in dates.iterrows() if pd.isna(row.get("Reported EPS"))]
                        earnings_date = _calendar_date(future)
            except Exception as exc:
                errors.append(f"earnings: {type(exc).__name__}")

            try:
                call_total = put_total = 0
                expiries = list(instrument.options or [])[:3]
                for expiry in expiries:
                    chain = instrument.option_chain(expiry)
                    calls = chain.calls["openInterest"] if "openInterest" in chain.calls else pd.Series(dtype=float)
                    puts = chain.puts["openInterest"] if "openInterest" in chain.puts else pd.Series(dtype=float)
                    call_total += int(pd.to_numeric(calls, errors="coerce").fillna(0).sum())
                    put_total += int(pd.to_numeric(puts, errors="coerce").fillna(0).sum())
                if expiries:
                    call_oi, put_oi = call_total, put_total
            except Exception as exc:
                errors.append(f"options: {type(exc).__name__}")
        except Exception as exc:
            errors.append(f"ticker: {type(exc).__name__}")

        return MarketContext(
            ticker=ticker.upper(),
            retrieved_at=instant,
            short_interest_pct_float=short_pct,
            days_to_cover=days_to_cover,
            # Yahoo's info fields do not disclose a trustworthy publication date.
            short_interest_as_of=None,
            short_interest_source="Yahoo Finance convenience fields (undated)",
            next_earnings_date=earnings_date,
            earnings_as_of=instant if earnings_date is not None else None,
            earnings_source="Yahoo Finance calendar" if earnings_date is not None else "",
            options_as_of=instant if call_oi is not None and put_oi is not None else None,
            option_call_open_interest=call_oi,
            option_put_open_interest=put_oi,
            options_source="Yahoo Finance option chain" if call_oi is not None else "",
            provider_errors=tuple(errors),
        )
