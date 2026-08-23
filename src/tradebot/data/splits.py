"""Train / validation / holdout boundaries, defined once and imported everywhere.

The boundaries are constants rather than parameters on purpose. Every strategy idea, every
parameter choice and every discarded experiment happens on DEV. VALIDATION is a sanity
check before anything is believed. HOLDOUT is looked at once, at the end, for ideas that
already survived both. If a HOLDOUT number ever informs a design decision, the holdout is
spent and its verdict no longer means anything — say so in the report rather than pretending
otherwise.

DEV deliberately holds the hardest regimes in the sample — the 2018 Q4 selloff, the 2020
COVID crash and the 2022 bear market — so a strategy is designed against difficulty rather
than tuned on a rising tape and then surprised by one.

These boundaries match the ones locked for the earlier 1-minute research on this same
dataset, so results are comparable across both bodies of work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from ..core.clock import MARKET_TZ

VALIDATION_START = pd.Timestamp("2023-01-01", tz=MARKET_TZ)
HOLDOUT_START = pd.Timestamp("2025-01-01", tz=MARKET_TZ)


class Split(str, Enum):
    DEV = "dev"
    VALIDATION = "validation"
    HOLDOUT = "holdout"
    TRAIN = "train"  # DEV + VALIDATION, for a final refit before touching HOLDOUT
    ALL = "all"


def slice_split(bars: pd.DataFrame, split: Split | str) -> pd.DataFrame:
    # `str(Split.DEV)` is "Split.DEV" from Python 3.12 on, so route through the member
    # check rather than stringifying an enum that is already the right type.
    split = split if isinstance(split, Split) else Split(str(split).strip().lower())
    idx = bars.index
    if split is Split.DEV:
        return bars[idx < VALIDATION_START]
    if split is Split.VALIDATION:
        return bars[(idx >= VALIDATION_START) & (idx < HOLDOUT_START)]
    if split is Split.HOLDOUT:
        return bars[idx >= HOLDOUT_START]
    if split is Split.TRAIN:
        return bars[idx < HOLDOUT_START]
    return bars


def dev(bars: pd.DataFrame) -> pd.DataFrame:
    return slice_split(bars, Split.DEV)


def validation(bars: pd.DataFrame) -> pd.DataFrame:
    return slice_split(bars, Split.VALIDATION)


def holdout(bars: pd.DataFrame) -> pd.DataFrame:
    return slice_split(bars, Split.HOLDOUT)


def train(bars: pd.DataFrame) -> pd.DataFrame:
    return slice_split(bars, Split.TRAIN)


@dataclass(frozen=True, slots=True)
class SplitSummary:
    name: str
    rows: int
    sessions: int
    start: str
    end: str

    def line(self) -> str:
        if self.rows == 0:
            return f"  {self.name:<11} empty"
        return (
            f"  {self.name:<11} {self.start} -> {self.end}  "
            f"{self.rows:>9,} bars  {self.sessions:>5,} sessions"
        )


def summarize(bars: pd.DataFrame) -> list[SplitSummary]:
    out = []
    for split in (Split.DEV, Split.VALIDATION, Split.HOLDOUT):
        part = slice_split(bars, split)
        if part.empty:
            out.append(SplitSummary(split.value.upper(), 0, 0, "", ""))
            continue
        out.append(
            SplitSummary(
                name=split.value.upper(),
                rows=len(part),
                sessions=int(part.index.normalize().nunique()),
                start=str(part.index[0].date()),
                end=str(part.index[-1].date()),
            )
        )
    return out


def describe(bars: pd.DataFrame) -> str:
    lines = [s.line() for s in summarize(bars)]
    lines[-1] += "   <- untouched until final"
    return "\n".join(lines)
