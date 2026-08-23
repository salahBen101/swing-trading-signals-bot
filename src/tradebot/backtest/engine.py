"""The backtest driver.

Deliberately thin. It builds the *same* strategy, risk engine, cost model, execution engine
and simulated broker that a paper run uses, then feeds them historical bars through the
*same* `ExecutionEngine.on_bar`. There is no second implementation of the fill logic, the
stop handling or the risk limits, so a backtest and a paper run cannot silently disagree —
which is the single most valuable property a backtester can have and the one most often
missing.

Two settings differ from live by default, both stated in the report rather than hidden:

* `enforce_drawdown_floor=False`. A real prop account is closed when it breaches the
  trailing floor, and enforcing that truncates a nine-year sample at the first bad month,
  measuring the account rule instead of the strategy. The breach is still detected and
  reported, so "this would have blown the evaluation on 2020-03-12" is not lost.
* Latency and failure injection are off. They are available for robustness runs; leaving
  them on by default would make results non-reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import pandas as pd

from ..broker.costs import CostModel
from ..broker.guarded import GuardedBroker
from ..broker.simulated import SimulatedBroker
from ..config import Config
from ..core.clock import SimulatedClock
from ..core.models import Bar, Rejection, Trade
from ..data.splits import Split, slice_split
from ..execution.engine import ExecutionEngine
from ..features.pipeline import FeatureFrame, build_features
from ..instruments.registry import InstrumentSpec, get_instrument
from ..journal.db import Journal
from ..risk.killswitch import KillSwitch
from ..risk.limits import RiskEngine
from ..strategy.base import Strategy


class BacktestTerminalStateError(RuntimeError):
    """The sample ended or crossed a session boundary with unresolved exposure."""


@dataclass(frozen=True, slots=True)
class BacktestResult:
    trades: list[Trade]
    rejections: list[Rejection]
    equity_curve: pd.Series
    strategy_name: str
    strategy_spec: dict
    instrument: str
    timeframe: str
    split: str
    bars: int
    sessions: int
    start: datetime | None
    end: datetime | None
    warmup_bars: int
    starting_equity: float
    cost_description: str
    ended_flat: bool = True
    working_orders_at_end: int = 0
    dataset_hash: str = ""
    floor_breached_at: datetime | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else self.starting_equity


def run_backtest(
    bars: pd.DataFrame,
    strategy: Strategy,
    config: Config,
    *,
    split: Split | str = Split.DEV,
    instrument: InstrumentSpec | None = None,
    journal: Journal | None = None,
    stress_costs: bool = False,
    enforce_drawdown_floor: bool = False,
    features: FeatureFrame | None = None,
    dataset_hash: str = "",
    progress_every: int = 0,
) -> BacktestResult:
    instrument = instrument or get_instrument(config.instrument)
    selected_split = split if isinstance(split, Split) else Split(str(split).strip().lower())
    sliced = slice_split(bars, selected_split)
    if sliced.empty:
        raise ValueError(f"split {selected_split.value!r} contains no bars")

    features = (
        _align_precomputed_features(features, sliced, strategy)
        if features is not None
        else build_features(sliced, strategy.features)
    )
    clock = SimulatedClock(sliced.index[0].to_pydatetime())
    costs = CostModel.from_config(config.costs, instrument, stress=stress_costs)

    # A flag file under the system temp would leak between runs; a backtest gets its own
    # in-memory switch so one run's halt cannot silently disarm the next one's.
    kill_switch = KillSwitch(
        _scratch_flag(config), max_consecutive_errors=config.risk.kill_switch.max_consecutive_errors,
        clock=clock,
    )
    kill_switch.clear()

    risk = RiskEngine(
        config.risk, instrument, config.session, clock=clock, kill_switch=kill_switch,
        starting_equity=config.risk.starting_equity_usd,
        enforce_drawdown_floor=enforce_drawdown_floor,
    )
    broker = SimulatedBroker(
        instrument, costs, config=config.broker.simulated, clock=clock,
        starting_equity=config.risk.starting_equity_usd,
    )
    broker.connect()
    guarded = GuardedBroker(broker, risk)

    engine = ExecutionEngine(
        strategy, risk, guarded, instrument, costs, journal=journal,
        starting_equity=config.risk.starting_equity_usd,
        log_every_bar=config.journal.log_every_bar,
    )

    return _drive_prepared_backtest(
        sliced=sliced,
        features=features,
        strategy=strategy,
        config=config,
        selected_split=selected_split,
        instrument=instrument,
        costs=costs,
        clock=clock,
        risk=risk,
        guarded=guarded,
        engine=engine,
        stress_costs=stress_costs,
        enforce_drawdown_floor=enforce_drawdown_floor,
        dataset_hash=dataset_hash,
        progress_every=progress_every,
    )


def _drive_prepared_backtest(
    *,
    sliced: pd.DataFrame,
    features: FeatureFrame,
    strategy: Strategy,
    config: Config,
    selected_split: Split,
    instrument: InstrumentSpec,
    costs: CostModel,
    clock: SimulatedClock,
    risk,
    guarded: GuardedBroker,
    engine: ExecutionEngine,
    stress_costs: bool,
    enforce_drawdown_floor: bool,
    dataset_hash: str,
    progress_every: int,
    before_bar: Callable[[Bar], None] | None = None,
    after_bar: Callable[[Bar], None] | None = None,
    on_session_end: Callable[[datetime, bool], None] | None = None,
    additional_notes: tuple[str, ...] = (),
    describe_legacy_floor: bool = True,
) -> BacktestResult:
    """Drive one already-constructed guarded execution stack.

    Both the legacy research runner and the prop-account runner use this event loop.  The
    hooks may observe/reconcile state but cannot replace ``ExecutionEngine.on_bar`` or
    submit orders, preserving the one guarded execution path.
    """

    opens = sliced["open"].to_numpy()
    highs = sliced["high"].to_numpy()
    lows = sliced["low"].to_numpy()
    closes = sliced["close"].to_numpy()
    volumes = sliced["volume"].to_numpy()
    index = sliced.index
    frame = features.frame

    equity_points: list[float] = []
    for i in range(len(sliced)):
        timestamp = index[i].to_pydatetime()
        if i and timestamp.date() != index[i - 1].date():
            _require_flat_terminal_state(
                engine, guarded, at=index[i - 1].to_pydatetime(),
                context="session boundary",
            )
            if on_session_end is not None:
                on_session_end(index[i - 1].to_pydatetime(), False)
        clock.set(timestamp)
        bar = Bar(timestamp, opens[i], highs[i], lows[i], closes[i], volumes[i])
        if before_bar is not None:
            before_bar(bar)
        engine.on_bar(
            bar, frame.iloc[i], index=i, frame=frame, bars=sliced,
            warmed_up=i >= features.warmup_bars,
        )
        if after_bar is not None:
            after_bar(bar)

        position = engine.state.position
        unrealized = (
            position.unrealized_usd(closes[i], instrument.multiplier)
            if position is not None else 0.0
        )
        equity_points.append(engine.state.equity + unrealized)

        if progress_every and i and i % progress_every == 0:
            print(f"  {i:,}/{len(sliced):,} bars, {len(engine.state.trades)} trades",
                  flush=True)

    working_at_end = _require_flat_terminal_state(
        engine, guarded, at=index[-1].to_pydatetime(), context="end of data",
    )
    if on_session_end is not None:
        on_session_end(index[-1].to_pydatetime(), True)

    notes = list(additional_notes)
    if selected_split is Split.HOLDOUT:
        notes.append(
            "WARNING: LOCKED HOLDOUT DATA WAS ACCESSED; this spends the final "
            "out-of-sample test and must be recorded in EXPERIMENTS.md"
        )
    elif selected_split is Split.ALL:
        notes.append(
            "WARNING: ALL DATA WAS ACCESSED, INCLUDING ANY LOCKED HOLDOUT ROWS; "
            "this run must never be used for parameter selection"
        )
    if describe_legacy_floor and not enforce_drawdown_floor:
        notes.append(
            "trailing-drawdown floor NOT enforced: the whole sample is measured rather "
            "than the account being closed at the first breach"
        )
    if describe_legacy_floor and risk.floor_breached_at is not None:
        notes.append(
            f"the trailing-drawdown floor WOULD have been breached at "
            f"{risk.floor_breached_at.isoformat()} — a real prop account ends there"
        )
    if stress_costs:
        notes.append(f"stress costs applied: {costs.describe()}")

    return BacktestResult(
        trades=list(engine.state.trades),
        rejections=list(engine.rejections),
        equity_curve=pd.Series(equity_points, index=index, name="equity"),
        strategy_name=strategy.name,
        strategy_spec=strategy.spec.to_dict(),
        instrument=instrument.symbol,
        timeframe=config.timeframe,
        split=selected_split.value,
        bars=len(sliced),
        sessions=int(index.normalize().nunique()),
        start=index[0].to_pydatetime(),
        end=index[-1].to_pydatetime(),
        warmup_bars=features.warmup_bars,
        starting_equity=config.risk.starting_equity_usd,
        cost_description=costs.describe(),
        ended_flat=True,
        working_orders_at_end=working_at_end,
        dataset_hash=dataset_hash,
        floor_breached_at=risk.floor_breached_at,
        notes=tuple(notes),
    )


def _align_precomputed_features(
    features: FeatureFrame, sliced: pd.DataFrame, strategy: Strategy
) -> FeatureFrame:
    """Select the requested split by timestamp without ever pairing rows positionally.

    A caller commonly caches one causal feature frame for the full archive and then runs a
    validation split. Positional indexing into that full frame paired 2024 bars with 2017
    features. Exact timestamp containment and spec equality make that failure loud.
    """
    frame = features.frame
    if features.spec != strategy.features:
        raise ValueError(
            "precomputed feature spec does not match the strategy's declared feature spec"
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("precomputed features need a DatetimeIndex")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("precomputed feature timestamps must be unique and monotonic")

    missing = sliced.index.difference(frame.index)
    if len(missing):
        example = ", ".join(ts.isoformat() for ts in missing[:3])
        raise ValueError(
            f"precomputed features are missing {len(missing)} bar timestamp(s), e.g. {example}"
        )

    locations = frame.index.get_indexer(sliced.index)
    if (locations < 0).any() or (locations[1:] <= locations[:-1]).any():
        raise ValueError("precomputed feature rows do not align monotonically to the split")
    aligned = frame.loc[sliced.index]
    local_warmup = int((locations < features.warmup_bars).sum())
    return FeatureFrame(frame=aligned, spec=features.spec, warmup_bars=local_warmup)


def _require_flat_terminal_state(
    engine: ExecutionEngine, broker: GuardedBroker, *, at: datetime, context: str
) -> int:
    """Fail rather than hide an overnight/open-ended position in a completed report."""
    broker_positions = broker.get_positions()
    working = [order for order in broker.get_orders() if not order.status.is_terminal]
    local_position = engine.state.position
    local_entry = engine.state.entry_order
    if local_position is not None or local_entry is not None or broker_positions or working:
        local_qty = 0 if local_position is None else local_position.signed_quantity
        broker_qty = sum(position.quantity for position in broker_positions)
        raise BacktestTerminalStateError(
            f"backtest reached {context} at {at.isoformat()} with unresolved exposure: "
            f"local position {local_qty}, broker position {broker_qty}, "
            f"{len(working)} working order(s). Provide a complete session ending after "
            "the configured flatten time; the result was not produced."
        )
    return len(working)


def _scratch_flag(config: Config) -> str:
    """A per-process kill-switch path, so a backtest never trips the live flag file."""
    import tempfile
    import uuid
    from pathlib import Path

    return str(Path(tempfile.gettempdir()) / f"tradebot-backtest-{uuid.uuid4().hex}.flag")
