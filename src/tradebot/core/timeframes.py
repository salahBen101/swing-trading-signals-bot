"""Timeframe names and their durations.

One table, so the config loader, the bar store, the feed and the data-integrity guard all
agree on what "5min" means. The integrity thresholds scale with bar duration, so a
disagreement here would quietly mis-calibrate them.
"""

from __future__ import annotations

TIMEFRAME_SECONDS: dict[str, float] = {
    "1s": 1, "5s": 5, "10s": 10, "15s": 15, "30s": 30,
    "1min": 60, "2min": 120, "3min": 180, "5min": 300,
    "10min": 600, "15min": 900, "30min": 1800, "60min": 3600, "1h": 3600,
}

# Aliases people actually type. Kept separate from the canonical table so
# `known_timeframes()` lists one name per duration rather than every spelling.
_ALIASES = {
    "1m": "1min", "2m": "2min", "3m": "3min", "5m": "5min",
    "10m": "10min", "15m": "15min", "30m": "30min", "60m": "60min",
    "1t": "1min", "h": "1h", "hour": "1h", "min": "1min",
}


class UnknownTimeframe(ValueError):
    pass


def canonical(timeframe: str) -> str:
    key = str(timeframe).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in TIMEFRAME_SECONDS:
        raise UnknownTimeframe(
            f"timeframe {timeframe!r} is not supported. Known: {known_timeframes()}"
        )
    return key


def timeframe_seconds(timeframe: str) -> float:
    return float(TIMEFRAME_SECONDS[canonical(timeframe)])


def pandas_rule(timeframe: str) -> str:
    """The pandas offset alias for `resample`."""
    key = canonical(timeframe)
    return key.replace("min", "min").replace("1h", "60min")


def known_timeframes() -> list[str]:
    return sorted(TIMEFRAME_SECONDS, key=lambda k: TIMEFRAME_SECONDS[k])
