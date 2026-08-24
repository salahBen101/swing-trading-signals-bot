"""Full-Globex (ETH) front-month NQ bars.

Every strategy family previously tested in this repository used an RTH-only dataset, which
made three of the most economically motivated hypotheses untestable: overnight inventory
correction, gap behaviour, and overnight-extreme sweeps. The Databento archive actually
contains the whole 18:00-17:00 ET session, so this module builds the dataset that unlocks
them.

Three things in the raw file will silently corrupt a study if mishandled, so each is
handled explicitly:

1. **Calendar spreads.** Rows like `NQU7-NQZ7` price near 200 rather than 20,000. The file
   is 103 spread symbols to 40 outrights. Averaging one into the price series would be
   catastrophic, so only `NQ` + month code + year digits survives.
2. **Overlapping contract months.** A continuous front-month series is built by taking,
   per session, whichever outright traded the most volume — the standard volume roll,
   which follows where liquidity actually is.
3. **The session boundary is not midnight.** A CME equity-index trading day runs from
   18:00 ET the previous evening to 17:00 ET. Assigning bars to calendar dates would split
   every overnight session in half and make "the overnight move" meaningless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import pandas as pd

from ..core.clock import MARKET_TZ

# NQ outright contracts only: NQ + month code + one or two year digits.
OUTRIGHT_RE = re.compile(r"^NQ[HMUZ]\d{1,2}$")

# CME Globex equity index, US/Eastern. The session opens at 18:00 the previous evening and
# settles at 17:00; 17:00-18:00 is the daily maintenance halt.
SESSION_OPEN = time(18, 0)
SESSION_CLOSE = time(17, 0)
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)


@dataclass(frozen=True, slots=True)
class EthBuildReport:
    raw_rows: int
    spread_rows_dropped: int
    outright_rows: int
    bars: int
    sessions: int
    contracts_used: int
    roll_dates: int
    first: pd.Timestamp
    last: pd.Timestamp

    def summary(self) -> str:
        return "\n".join([
            f"raw 1m rows          {self.raw_rows:,}",
            f"spread rows dropped  {self.spread_rows_dropped:,}",
            f"outright rows kept   {self.outright_rows:,}",
            f"front-month bars     {self.bars:,}",
            f"sessions             {self.sessions:,}",
            f"contracts / rolls    {self.contracts_used} / {self.roll_dates}",
            f"range                {self.first} -> {self.last}",
        ])


def session_date(index: pd.DatetimeIndex) -> pd.Series:
    """Map each bar to the trading date it belongs to.

    Bars from 18:00 ET onward belong to the *next* calendar day's session, which is what
    makes "the overnight move into today's open" a coherent quantity.
    """
    local = index.tz_convert(MARKET_TZ)
    dates = pd.Series(local.normalize(), index=index)
    evening = local.time >= SESSION_OPEN
    dates[evening] = dates[evening] + pd.Timedelta(days=1)
    return dates


def build_eth_bars(dbn_path: str | Path) -> tuple[pd.DataFrame, EthBuildReport]:
    """Continuous front-month 1-minute ETH bars, tz-aware, with session labelling."""
    import databento as db

    frame = db.DBNStore.from_file(str(dbn_path)).to_df()
    if not isinstance(frame, pd.DataFrame):
        frame = pd.concat(list(frame))
    raw_rows = len(frame)

    symbols = frame["symbol"].astype("string")
    is_outright = symbols.str.fullmatch(OUTRIGHT_RE, na=False)
    frame = frame.loc[is_outright, ["open", "high", "low", "close", "volume", "symbol"]]
    outright_rows = len(frame)

    frame.index = pd.DatetimeIndex(frame.index).tz_convert(MARKET_TZ)
    frame = frame.sort_index(kind="stable")

    sessions = session_date(frame.index)
    front, contracts, rolls = _select_front_month(frame, sessions)

    front = front.drop(columns=["symbol"]).astype("float64")
    front.index.name = "timestamp"
    # Recompute after the roll filter so labels line up with the surviving rows.
    front["session"] = session_date(front.index).to_numpy()
    local_time = front.index.tz_convert(MARKET_TZ).time
    front["is_rth"] = [(RTH_OPEN <= t < RTH_CLOSE) for t in local_time]

    report = EthBuildReport(
        raw_rows=raw_rows,
        spread_rows_dropped=raw_rows - outright_rows,
        outright_rows=outright_rows,
        bars=len(front),
        sessions=int(front["session"].nunique()),
        contracts_used=len(contracts),
        roll_dates=rolls,
        first=front.index[0],
        last=front.index[-1],
    )
    return front, report


def _select_front_month(
    frame: pd.DataFrame, sessions: pd.Series
) -> tuple[pd.DataFrame, list[str], int]:
    """Front month = whichever outright traded the most volume that session."""
    volume = frame.groupby([sessions, frame["symbol"]], observed=True)["volume"].sum()
    front_by_session = volume.groupby(level=0).idxmax().map(lambda pair: pair[1])

    mapped = sessions.map(front_by_session)
    front = frame[frame["symbol"].to_numpy() == mapped.to_numpy()]

    changes = front_by_session.ne(front_by_session.shift(1))
    changes.iloc[0] = False
    contracts = list(dict.fromkeys(front_by_session.tolist()))
    return front, contracts, int(changes.sum())


def load_or_build(
    dbn_path: str | Path,
    cache_path: str | Path,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, EthBuildReport | None]:
    """Parquet cache in front of a multi-minute parse."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        bars = pd.read_parquet(cache_path)
        bars.index = pd.DatetimeIndex(bars.index).tz_convert(MARKET_TZ)
        return bars, None

    bars, report = build_eth_bars(dbn_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(cache_path)
    return bars, report
