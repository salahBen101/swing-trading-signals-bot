"""Session-developing volume profile, for use as an entry confirmation.

A volume profile is volume-at-price rather than volume-over-time. From it come the point of
control (POC, the price with the most traded volume) and the value area (the tightest band
holding 70% of volume). The confirmation idea: price *leaving* the value area is acceptance
of a new range — a continuation context — whereas price *inside* it is balance, where a
directional bet has no edge.

Built causally and session-developing: the profile at bar *i* uses only bars up to and
including *i*, so a confirmation read on bar *i* could have been made in real time. The
profile resets each session.

Volume is distributed across each bar's range on a tick-binned grid. From 5-minute OHLCV
that is an approximation of the true intrabar distribution, but a defensible one — a bar's
volume did trade somewhere between its low and high — and it is the same approximation for
every bar, so it cannot favour one hypothesis over another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Coarser than the 0.25 tick so the grid stays small over a full session; 1 index point
# (4 ticks) is fine enough to place the POC and value-area edges without a huge array.
_PRICE_BIN = 1.0
_VALUE_AREA_FRACTION = 0.70


def add_value_area(bars: pd.DataFrame) -> pd.DataFrame:
    """Attach developing POC and value-area edges to each bar.

    New columns: `vp_poc`, `vp_val` (value-area low), `vp_vah` (value-area high). Each is
    the profile as it stood at that bar's close, using that session's bars so far.
    """
    out = bars.copy()
    highs = out["high"].to_numpy(dtype="float64")
    lows = out["low"].to_numpy(dtype="float64")
    volumes = out["volume"].to_numpy(dtype="float64")
    sessions = out["session"].to_numpy()

    poc = np.full(len(out), np.nan)
    val = np.full(len(out), np.nan)
    vah = np.full(len(out), np.nan)

    start = 0
    for end in _session_bounds(sessions):
        _fill_session(highs[start:end], lows[start:end], volumes[start:end],
                      poc[start:end], val[start:end], vah[start:end])
        start = end

    out["vp_poc"] = poc
    out["vp_val"] = val
    out["vp_vah"] = vah
    # Where the close sits relative to the developing value area: +1 above, -1 below, 0 in.
    close = out["close"].to_numpy(dtype="float64")
    out["vp_position"] = np.where(
        close > vah, 1, np.where(close < val, -1, 0)
    ).astype("int8")
    return out


def _session_bounds(sessions: np.ndarray) -> list[int]:
    changes = np.flatnonzero(sessions[1:] != sessions[:-1]) + 1
    return [*changes.tolist(), len(sessions)]


def _fill_session(
    highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray,
    poc: np.ndarray, val: np.ndarray, vah: np.ndarray,
) -> None:
    if len(highs) == 0:
        return
    lo = float(np.floor(lows.min() / _PRICE_BIN) * _PRICE_BIN)
    hi = float(np.ceil(highs.max() / _PRICE_BIN) * _PRICE_BIN)
    n_bins = max(1, int(round((hi - lo) / _PRICE_BIN)) + 1)
    grid = lo + np.arange(n_bins) * _PRICE_BIN
    profile = np.zeros(n_bins)

    for k in range(len(highs)):
        bar_lo, bar_hi, vol = lows[k], highs[k], volumes[k]
        first = int((bar_lo - lo) / _PRICE_BIN)
        last = int((bar_hi - lo) / _PRICE_BIN)
        span = max(1, last - first + 1)
        profile[first:last + 1] += vol / span

        total = profile.sum()
        if total <= 0:
            continue
        peak = int(profile.argmax())
        poc[k] = grid[peak]

        # Grow a window outward from the POC until it holds the value-area fraction.
        lower = upper = peak
        covered = profile[peak]
        target = _VALUE_AREA_FRACTION * total
        while covered < target and (lower > 0 or upper < n_bins - 1):
            take_up = profile[upper + 1] if upper < n_bins - 1 else -1.0
            take_dn = profile[lower - 1] if lower > 0 else -1.0
            if take_up >= take_dn:
                upper += 1
                covered += max(0.0, take_up)
            else:
                lower -= 1
                covered += max(0.0, take_dn)
        val[k] = grid[lower]
        vah[k] = grid[upper]


def value_area_confirms(bars: pd.DataFrame, direction: np.ndarray) -> np.ndarray:
    """Keep only entries where price has left the value area in the trade's direction.

    A long is confirmed when the close is above the developing value-area high; a short when
    below the value-area low. This is the "acceptance of a new range" filter: it removes
    trades taken while price is still balanced around the point of control, which is where a
    directional entry has the least reason to work.
    """
    if "vp_position" not in bars.columns:
        raise ValueError("call add_value_area(bars) before value_area_confirms")
    position = bars["vp_position"].to_numpy()
    confirmed = direction.copy()
    # Long kept only if price is above value area; short only if below.
    kill = ((direction > 0) & (position <= 0)) | ((direction < 0) & (position >= 0))
    confirmed[kill] = 0
    return confirmed
