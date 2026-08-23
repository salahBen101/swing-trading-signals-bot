"""Market data feeds.

**The one invariant:** a feed yields *closed* bars and nothing else. An in-progress bar has
a high, a low and a close that are all still going to move, and a strategy that sees one is
reading the future in the most direct way possible — it acts on a close before the close
exists. Every intraday system that "worked in backtest and not in live" is a candidate for
having got this wrong in one direction or the other.

`BarAggregator` therefore has two clearly separate accessors: `update()`/`flush()`, which
return only completed bars, and `current_partial()`, which is diagnostic and is never
called by the runner. The separation is what makes the invariant testable.
"""

from __future__ import annotations

import time as _time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

import pandas as pd

from ..core.clock import MARKET_TZ, Clock, SimulatedClock, SystemClock, ensure_aware
from ..core.models import Bar
from .store import frame_to_bars


@dataclass(frozen=True, slots=True)
class FeedHealth:
    last_bar_time: datetime | None
    last_arrival_time: datetime | None
    seconds_since_last_bar: float
    staleness_multiple: float
    is_stale: bool
    is_hard_stale: bool


@runtime_checkable
class MarketDataFeed(Protocol):
    instrument: str
    timeframe_seconds: float

    def __iter__(self) -> Iterator[Bar]: ...

    def health(self, now: datetime) -> FeedHealth: ...


class _FeedBase:
    """Shared staleness bookkeeping.

    Staleness is measured from the *arrival* of the last bar, not from its timestamp: a
    replay feed that hands over a 2019 bar is not stale, and a live feed that has gone
    silent for ten minutes is, even though its last bar's timestamp looks recent relative
    to itself.
    """

    def __init__(
        self,
        instrument: str,
        timeframe_seconds: float,
        *,
        stale_multiple: float = 3.0,
        hard_stale_multiple: float = 10.0,
        clock: Clock | None = None,
    ) -> None:
        if timeframe_seconds <= 0:
            raise ValueError("timeframe_seconds must be positive")
        self.instrument = instrument
        self.timeframe_seconds = float(timeframe_seconds)
        self.stale_multiple = stale_multiple
        self.hard_stale_multiple = hard_stale_multiple
        self._clock = clock or SystemClock()
        self.last_bar_time: datetime | None = None
        self.last_arrival_time: datetime | None = None
        self.bars_emitted = 0

    def _record(self, bar: Bar) -> None:
        self.last_bar_time = bar.timestamp
        self.last_arrival_time = self._clock.now()
        self.bars_emitted += 1

    def health(self, now: datetime | None = None) -> FeedHealth:
        now = now or self._clock.now()
        if self.last_arrival_time is None:
            # Nothing has arrived yet. That is "not started", not "stale" — reporting a
            # fresh process as DOWN would trip the monitor before the first bar ever lands.
            return FeedHealth(None, None, 0.0, 0.0, False, False)
        gap = (ensure_aware(now) - self.last_arrival_time).total_seconds()
        multiple = gap / self.timeframe_seconds
        return FeedHealth(
            last_bar_time=self.last_bar_time,
            last_arrival_time=self.last_arrival_time,
            seconds_since_last_bar=gap,
            staleness_multiple=multiple,
            is_stale=multiple >= self.stale_multiple,
            is_hard_stale=multiple >= self.hard_stale_multiple,
        )


class ReplayFeed(_FeedBase):
    """Drives the runner from a historical frame.

    Used for backtests and for paper trading against recorded data. `speed` controls
    pacing: `None` runs as fast as possible (the backtest case), a number sleeps so that
    one bar interval takes `interval / speed` real seconds (useful for watching the
    dashboard behave).

    When given a `SimulatedClock`, the feed pins it to each bar's *close* time before
    yielding. Everything downstream — session rollover, the flatten-before-close rule,
    staleness — then sees the instant that the bar actually belongs to, rather than
    wall-clock time leaking into a simulated decision.
    """

    def __init__(
        self,
        bars: pd.DataFrame | Iterable[Bar],
        instrument: str,
        timeframe_seconds: float,
        *,
        speed: float | None = None,
        clock: Clock | None = None,
        **kwargs,
    ) -> None:
        super().__init__(instrument, timeframe_seconds, clock=clock, **kwargs)
        self._bars = frame_to_bars(bars) if isinstance(bars, pd.DataFrame) else list(bars)
        self.speed = speed

    def __len__(self) -> int:
        return len(self._bars)

    def __iter__(self) -> Iterator[Bar]:
        interval = timedelta(seconds=self.timeframe_seconds)
        for bar in self._bars:
            if isinstance(self._clock, SimulatedClock):
                # The bar's timestamp is its OPEN; it is only knowable at its close.
                self._clock.set(bar.timestamp + interval)
            self._record(bar)
            yield bar
            if self.speed:
                _time.sleep(self.timeframe_seconds / self.speed)


class BarAggregator:
    """Builds closed bars from a stream of price updates.

    A bucket is emitted only once its interval has elapsed, which happens in one of two
    ways: an update arrives that belongs to a later bucket, or `flush(now)` is called with
    a time past the bucket's end. The second case matters for thin instruments and for the
    tail of a session, where waiting for the next trade would delay the close indefinitely.
    """

    def __init__(self, interval_seconds: float, *, tz=MARKET_TZ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval = float(interval_seconds)
        self.tz = tz
        self._bucket_start: datetime | None = None
        self._o = self._h = self._l = self._c = 0.0
        self._v = 0.0

    def _floor(self, ts: datetime) -> datetime:
        ts = ensure_aware(ts, tz=self.tz)
        epoch_seconds = ts.timestamp()
        floored = epoch_seconds - (epoch_seconds % self.interval)
        return datetime.fromtimestamp(floored, tz=self.tz)

    def _close_current(self) -> Bar | None:
        if self._bucket_start is None:
            return None
        bar = Bar(
            timestamp=self._bucket_start,
            open=self._o, high=self._h, low=self._l, close=self._c, volume=self._v,
        )
        self._bucket_start = None
        return bar

    def update(self, timestamp: datetime, price: float, volume: float = 0.0) -> Bar | None:
        """Fold one price update in. Returns a bar only when one *completes*."""
        bucket = self._floor(timestamp)
        completed: Bar | None = None

        if self._bucket_start is not None and bucket > self._bucket_start:
            completed = self._close_current()

        if self._bucket_start is None:
            self._bucket_start = bucket
            self._o = self._h = self._l = self._c = float(price)
            self._v = float(volume)
        else:
            self._h = max(self._h, price)
            self._l = min(self._l, price)
            self._c = float(price)
            self._v += float(volume)
        return completed

    def flush(self, now: datetime) -> Bar | None:
        """Close the in-progress bucket if its interval has elapsed. Otherwise nothing."""
        if self._bucket_start is None:
            return None
        if ensure_aware(now, tz=self.tz) < self._bucket_start + timedelta(seconds=self.interval):
            return None
        return self._close_current()

    def current_partial(self) -> Bar | None:
        """Diagnostic only.

        Never call this from the runner. It exists so a dashboard can show an in-progress
        bar, and so a test can assert that the value it returns is exactly what `update()`
        refused to hand over.
        """
        if self._bucket_start is None:
            return None
        return Bar(
            timestamp=self._bucket_start,
            open=self._o, high=self._h, low=self._l, close=self._c, volume=self._v,
        )


class LiveFeed(_FeedBase):
    """Wraps a raw update source in a `BarAggregator`.

    The source is any iterable of `(timestamp, price, volume)`; a broker's market-data
    socket adapts to that shape without this module knowing anything about the broker.
    v1 ships no such source (see docs/LIMITATIONS.md) — this is the seam it will attach to,
    and it is exercised by tests with a synthetic source.
    """

    def __init__(
        self,
        source: Iterable[tuple[datetime, float, float]],
        instrument: str,
        timeframe_seconds: float,
        **kwargs,
    ) -> None:
        super().__init__(instrument, timeframe_seconds, **kwargs)
        self._source = source
        self._aggregator = BarAggregator(timeframe_seconds)

    def __iter__(self) -> Iterator[Bar]:
        for ts, price, volume in self._source:
            bar = self._aggregator.update(ts, price, volume)
            if bar is not None:
                self._record(bar)
                yield bar
        final = self._aggregator.flush(self._clock.now())
        if final is not None:
            self._record(final)
            yield final
