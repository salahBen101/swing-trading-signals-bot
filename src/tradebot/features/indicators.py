"""Causal indicator library.

**Every function here must satisfy prefix-equality:** computing on `bars[:k]` must give
exactly the first `k` values of computing on the whole frame. That property is what makes
a backtest's indicator values equal to the ones a live run would have had at the same
instant, and `tests/tradebot/test_features.py` asserts it for every public function here.

Practically, that bans three things:

* `shift(-n)` and any negative offset
* `center=True` on a rolling window
* any statistic computed over the whole series (a global mean, a full-sample quantile,
  `scipy` filters that run forwards and backwards)

Rolling and expanding windows are fine. `ewm` is fine. Anything session-anchored is fine
provided it groups by session and accumulates forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing, the one ATR/RSI/ADX are defined against.

    Equivalent to an EMA with alpha = 1/period rather than 2/(period+1). Using an ordinary
    EMA here makes every Wilder-derived indicator disagree with every charting package,
    which turns "my backtest and my platform disagree" into a multi-day investigation.
    """
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return wilder_ema(true_range(high, low, close), period)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_ema(gain, period)
    avg_loss = wilder_ema(loss, period)
    # A window with no down bars has an undefined RS; RSI is 100 there by convention.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def bollinger(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, window)
    sd = series.rolling(window, min_periods=window).std(ddof=0)
    return mid - num_std * sd, mid, mid + num_std * sd


def donchian(
    high: pd.Series, low: pd.Series, window: int = 20
) -> tuple[pd.Series, pd.Series]:
    """Rolling channel over the *previous* `window` bars.

    The current bar is excluded. A Donchian breakout compares the current price to the
    prior range; if the current bar's own high is inside the channel it defines, price can
    never exceed it and the signal never fires.
    """
    upper = high.shift(1).rolling(window, min_periods=window).max()
    lower = low.shift(1).rolling(window, min_periods=window).min()
    return lower, upper


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder's ADX with +DI and -DI. Returns `(adx, plus_di, minus_di)`."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    tr_smooth = wilder_ema(true_range(high, low, close), period)
    plus_di = 100.0 * wilder_ema(plus_dm, period) / tr_smooth.replace(0.0, np.nan)
    minus_di = 100.0 * wilder_ema(minus_dm, period) / tr_smooth.replace(0.0, np.nan)

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return wilder_ema(dx, period), plus_di, minus_di


def efficiency_ratio(series: pd.Series, window: int = 40) -> pd.Series:
    """Kaufman's efficiency ratio: net displacement over total path length, in [0, 1].

    Near 1 the market is travelling in a straight line; near 0 it is grinding sideways.
    Used as a directionality gate — a trend rule in a chop regime is the classic way to
    donate to the market.
    """
    net = (series - series.shift(window)).abs()
    path = series.diff().abs().rolling(window, min_periods=window).sum()
    return net / path.replace(0.0, np.nan)


def realized_volatility(series: pd.Series, window: int = 50) -> pd.Series:
    """Standard deviation of log returns over `window` bars, unannualised."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window, min_periods=window).std(ddof=0)


def rolling_percentile(series: pd.Series, window: int = 252, min_periods: int = 30) -> pd.Series:
    """Where the current value sits within its own trailing distribution, in [0, 1].

    Deliberately *not* a whole-sample quantile. Ranking today's ATR against the full
    history including the future is one of the most common and least visible forms of
    look-ahead in volatility filters.
    """
    return series.rolling(window, min_periods=min_periods).rank(pct=True)


def autocorrelation(series: pd.Series, window: int = 20, lag: int = 1) -> pd.Series:
    returns = series.diff()
    return returns.rolling(window, min_periods=window).corr(returns.shift(lag))


def volume_ratio(volume: pd.Series, window: int = 50) -> pd.Series:
    """Current volume relative to its trailing average. The current bar is excluded from
    the baseline so a genuine surge is not diluted by itself."""
    baseline = volume.shift(1).rolling(window, min_periods=window).mean()
    return volume / baseline.replace(0.0, np.nan)


# ------------------------------------------------------------------ session-anchored


def session_key(index: pd.DatetimeIndex) -> pd.Series:
    """The trading date each bar belongs to. RTH-only data makes this the calendar date."""
    return pd.Series(index.normalize(), index=index, name="session")


def session_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Volume-weighted average price, reset at every session boundary.

    Typical price (H+L+C)/3 is the industry convention. The reset matters: a VWAP carried
    across sessions is anchored to yesterday's business and no longer describes where
    today's participants are positioned.
    """
    typical = (high + low + close) / 3.0
    session = session_key(close.index)
    pv = (typical * volume).groupby(session).cumsum()
    vol = volume.groupby(session).cumsum()
    return pv / vol.replace(0.0, np.nan)


def session_vwap_bands(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, num_std: float = 1.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """VWAP with volume-weighted standard-deviation bands, `(lower, vwap, upper)`."""
    typical = (high + low + close) / 3.0
    session = session_key(close.index)
    vwap = session_vwap(high, low, close, volume)

    vol_cum = volume.groupby(session).cumsum().replace(0.0, np.nan)
    pv2 = (typical.pow(2) * volume).groupby(session).cumsum()
    variance = (pv2 / vol_cum) - vwap.pow(2)
    sd = np.sqrt(variance.clip(lower=0.0))
    return vwap - num_std * sd, vwap, vwap + num_std * sd


def session_cumulative(series: pd.Series, how: str = "max") -> pd.Series:
    """Running session high/low/first — the session's own extremes so far, never ahead."""
    grouped = series.groupby(session_key(series.index))
    if how == "max":
        return grouped.cummax()
    if how == "min":
        return grouped.cummin()
    if how == "first":
        return grouped.transform("first")
    raise ValueError(f"unsupported how={how!r}")


def opening_range(
    high: pd.Series, low: pd.Series, minutes: int, session_start: str = "09:30"
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """The first `minutes` of each session, `(low, high, is_complete)`.

    The range is `NaN` until the window closes, and `is_complete` is False during it. A
    breakout rule that reads a still-forming opening range is comparing price to a level
    that is defined by that same price — it always looks like a breakout and never is.
    """
    index = high.index
    session = session_key(index)
    start = pd.to_datetime(session_start).time()

    start_ts = pd.Series(
        pd.DatetimeIndex(session).tz_localize(None) + pd.Timedelta(
            hours=start.hour, minutes=start.minute
        ),
        index=index,
    )
    elapsed = (index.tz_localize(None) - pd.DatetimeIndex(start_ts)).total_seconds() / 60.0
    in_window = pd.Series((elapsed >= 0) & (elapsed < minutes), index=index)

    or_high = high.where(in_window).groupby(session).cummax()
    or_low = low.where(in_window).groupby(session).cummin()

    # Freeze the values at the moment the window closes, then carry them forward. Without
    # the forward fill the level is only readable on the boundary bar itself.
    complete = pd.Series(elapsed >= minutes, index=index)
    or_high = or_high.groupby(session).ffill().where(complete)
    or_low = or_low.groupby(session).ffill().where(complete)
    return or_low, or_high, complete


def bars_since_session_start(index: pd.DatetimeIndex) -> pd.Series:
    """0 on the session's first bar, 1 on the next, and so on."""
    session = session_key(index)
    return pd.Series(1, index=index).groupby(session).cumsum() - 1


def prior_session_value(series: pd.Series, how: str = "last") -> pd.Series:
    """Yesterday's close / high / low, broadcast across today.

    Built from a per-session aggregate shifted by one session, so today's own bars never
    contribute to today's "prior session" level.
    """
    session = session_key(series.index)
    per_session = series.groupby(session).agg(how)
    shifted = per_session.shift(1)
    return session.map(shifted)
