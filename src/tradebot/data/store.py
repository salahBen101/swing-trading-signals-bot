"""Historical bar storage.

Parquet files addressed by `(instrument, timeframe)`, with a JSON manifest recording what
is in each one. The manifest exists so a backtest report can state exactly which dataset
produced it — row count, date range, and a content hash — and so a later run can detect
that the underlying data changed out from under a saved result.

Import is idempotent: re-importing the same rows is a no-op, and overlapping rows are
resolved in favour of the incoming data after both sides pass integrity validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..core.clock import MARKET_TZ
from ..core.models import Bar
from ..core.timeframes import timeframe_seconds
from .schema import REQUIRED_COLUMNS, DataIntegrityError, validate_bars

MANIFEST_NAME = "tradebot_manifest.json"


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    instrument: str
    timeframe: str
    path: str
    rows: int
    sessions: int
    start: str
    end: str
    content_hash: str
    imported_at: str

    def summary(self) -> str:
        return (
            f"{self.instrument} {self.timeframe}: {self.rows:,} bars, "
            f"{self.sessions:,} sessions, {self.start} -> {self.end} "
            f"[{self.content_hash[:12]}]"
        )


def _content_hash(df: pd.DataFrame) -> str:
    """Stable digest of the bar contents.

    Hashing the index and the OHLCV values (rather than the file bytes) means the digest
    is invariant to parquet compression settings and row-group layout, so re-writing an
    unchanged dataset does not appear to be a change.
    """
    hasher = hashlib.sha256()
    hasher.update(df.index.astype("int64").to_numpy().tobytes())
    for col in REQUIRED_COLUMNS:
        hasher.update(df[col].astype("float64").to_numpy().tobytes())
    return hasher.hexdigest()


def normalize_bars(df: pd.DataFrame, *, tz: str = "US/Eastern") -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame into the store's canonical shape.

    Canonical means: a tz-aware `DatetimeIndex` named `timestamp`, sorted, deduplicated,
    with exactly the five OHLCV columns as float64/float64/.../float64. Anything the
    caller brought along (symbol, vendor ids) is dropped here rather than carried into the
    engine, where an extra column becomes an accidental dependency.
    """
    out = df.copy()

    lower = {c.lower(): c for c in out.columns}
    for want in REQUIRED_COLUMNS:
        if want not in out.columns:
            if want in lower:
                out = out.rename(columns={lower[want]: want})
            else:
                raise DataIntegrityError(f"input frame has no {want!r} column")

    if not isinstance(out.index, pd.DatetimeIndex):
        ts_col = next((c for c in ("timestamp", "ts_event", "time", "date") if c in out.columns), None)
        if ts_col is None:
            raise DataIntegrityError(
                "input frame needs a DatetimeIndex or a timestamp/ts_event/time/date column"
            )
        out.index = pd.to_datetime(out[ts_col], utc=True)
        out = out.drop(columns=[ts_col])

    out.index = (
        out.index.tz_localize("UTC") if out.index.tz is None else out.index
    ).tz_convert(tz)
    out.index.name = "timestamp"

    out = out.loc[:, list(REQUIRED_COLUMNS)].astype("float64")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


class BarStore:
    """Parquet-backed store. One file per (instrument, timeframe)."""

    def __init__(self, root: str | Path = "data_cache") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / MANIFEST_NAME

    # -- addressing ---------------------------------------------------------------------
    def path_for(self, instrument: str, timeframe: str) -> Path:
        return self.root / f"{instrument.lower()}_{timeframe.lower()}.parquet"

    def _key(self, instrument: str, timeframe: str) -> str:
        return f"{instrument.upper()}|{timeframe.lower()}"

    # -- manifest -----------------------------------------------------------------------
    def manifest(self) -> dict[str, DatasetInfo]:
        if not self._manifest_path.exists():
            return {}
        raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return {k: DatasetInfo(**v) for k, v in raw.items()}

    def _write_manifest(self, entries: dict[str, DatasetInfo]) -> None:
        payload = {k: asdict(v) for k, v in entries.items()}
        self._manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def info(self, instrument: str, timeframe: str) -> DatasetInfo | None:
        return self.manifest().get(self._key(instrument, timeframe))

    # -- write --------------------------------------------------------------------------
    def import_bars(
        self,
        instrument: str,
        timeframe: str,
        bars: pd.DataFrame,
        *,
        replace: bool = False,
        validate: bool = True,
    ) -> DatasetInfo:
        """Merge `bars` into the dataset. Idempotent for identical input."""
        incoming = normalize_bars(bars)
        secs = timeframe_seconds(timeframe)
        if validate:
            validate_bars(incoming, timeframe_seconds=secs)

        path = self.path_for(instrument, timeframe)
        if path.exists() and not replace:
            existing = pd.read_parquet(path)
            existing.index = pd.DatetimeIndex(existing.index).tz_convert(MARKET_TZ)
            # Incoming wins on collision: a re-import is normally a correction.
            combined = pd.concat([existing, incoming])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = incoming

        if validate:
            validate_bars(combined, timeframe_seconds=secs)

        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path)

        info = DatasetInfo(
            instrument=instrument.upper(),
            timeframe=timeframe.lower(),
            path=str(path.relative_to(self.root)),
            rows=len(combined),
            sessions=int(combined.index.normalize().nunique()),
            start=combined.index[0].isoformat(),
            end=combined.index[-1].isoformat(),
            content_hash=_content_hash(combined),
            imported_at=datetime.now(tz=MARKET_TZ).isoformat(timespec="seconds"),
        )
        entries = self.manifest()
        entries[self._key(instrument, timeframe)] = info
        self._write_manifest(entries)
        return info

    def import_parquet(self, instrument: str, timeframe: str, source: str | Path, **kw) -> DatasetInfo:
        return self.import_bars(instrument, timeframe, pd.read_parquet(source), **kw)

    def import_csv(self, instrument: str, timeframe: str, source: str | Path, **kw) -> DatasetInfo:
        return self.import_bars(instrument, timeframe, pd.read_csv(source), **kw)

    # -- read ---------------------------------------------------------------------------
    def load(
        self,
        instrument: str,
        timeframe: str,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        validate: bool = True,
    ) -> pd.DataFrame:
        path = self.path_for(instrument, timeframe)
        if not path.exists():
            raise FileNotFoundError(
                f"no dataset for {instrument} {timeframe} at {path}. "
                f"Import one first (tradebot import-data --help)."
            )
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(df.index)
        df.index = (
            df.index.tz_localize("UTC") if df.index.tz is None else df.index
        ).tz_convert(MARKET_TZ)
        df.index.name = "timestamp"

        if validate:
            validate_bars(df, timeframe_seconds=timeframe_seconds(timeframe))

        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz=MARKET_TZ)]
        if end is not None:
            df = df[df.index < pd.Timestamp(end, tz=MARKET_TZ)]
        return df

    def has(self, instrument: str, timeframe: str) -> bool:
        return self.path_for(instrument, timeframe).exists()

    def datasets(self) -> list[DatasetInfo]:
        return sorted(self.manifest().values(), key=lambda d: (d.instrument, d.timeframe))


def resample(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate to a coarser timeframe.

    Empty buckets are dropped rather than forward-filled. Inventing a flat bar where
    nothing traded fabricates a zero-range, zero-volume observation, which then feeds
    volatility and volume-surge indicators as if it were a real quiet period.
    """
    agg = bars.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"])
    agg.index.name = "timestamp"
    return agg


def frame_to_bars(df: pd.DataFrame) -> list[Bar]:
    """Materialise a frame as `Bar` objects. Validation happens in `Bar.__post_init__`."""
    return [
        Bar(
            timestamp=ts.to_pydatetime(),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        )
        for ts, row in zip(df.index, df.itertuples(index=False), strict=True)
    ]
