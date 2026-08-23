"""Backtest reports.

A report is not a scoreboard. Everything that would let a reader talk themselves into a
result is printed alongside the result: the exact rule set, the dataset and its hash, the
cost model, the split, the search count and the |t| that count implies, and whether a real
prop account would have been closed partway through.

The header states the project's prior explicitly. Thirteen intraday families have failed
out-of-sample here already; a fourteenth showing a positive number on DEV is the expected
outcome of noise, not news.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from ..backtest.engine import BacktestResult
from .metrics import Bucket, Metrics, by_year, compute_metrics, selection_bar


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def _table(rows: Iterable[Bucket], title: str) -> list[str]:
    rows = list(rows)
    if not rows:
        return []
    out = [f"  {title}", f"    {'':<10} {'n':>6} {'net':>12} {'avg':>10} {'win%':>7} {'avgR':>7}"]
    for bucket in rows:
        out.append(
            f"    {bucket.label:<10} {bucket.trades:>6} {bucket.net_pnl:>12,.2f} "
            f"{bucket.avg_pnl:>10,.2f} {bucket.win_rate:>6.1%} {bucket.avg_r:>7.2f}"
        )
    return out


def text_report(
    result: BacktestResult, metrics: Metrics | None = None, *, configurations_tried: int = 1
) -> str:
    metrics = metrics or compute_metrics(
        result.trades, starting_equity=result.starting_equity,
        configurations_tried=configurations_tried,
        marked_equity_curve=result.equity_curve,
    )
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add(f"BACKTEST  {result.strategy_name}  {result.instrument} {result.timeframe}")
    add("=" * 78)

    add("")
    add("DATASET")
    add(f"  split          {result.split.upper()}")
    add(f"  period         {result.start.date()} -> {result.end.date()}")
    add(f"  bars           {result.bars:,} over {result.sessions:,} sessions")
    add(f"  warmup         {result.warmup_bars:,} bars (no trading before this)")
    add(f"  terminal       {'FLAT' if result.ended_flat else 'UNRESOLVED'} "
        f"({result.working_orders_at_end} working orders)")
    if result.dataset_hash:
        add(f"  content hash   {result.dataset_hash[:16]}")

    add("")
    add("RULES")
    for line in _spec_lines(result.strategy_spec):
        add(f"  {line}")

    add("")
    add("COSTS")
    add(f"  {result.cost_description}")

    add("")
    add("RESULTS")
    if metrics.trades == 0:
        add("  no trades were taken")
    else:
        add(f"  trades              {metrics.trades:,}")
        add(f"  win rate            {metrics.win_rate:.1%} "
            f"({metrics.wins}W / {metrics.losses}L / {metrics.scratches} scratch)")
        add(f"  gross P&L           {_fmt_money(metrics.gross_pnl)}")
        add(f"  commission          {_fmt_money(-metrics.commission)}")
        add(f"  net P&L             {_fmt_money(metrics.net_pnl)}")
        add(f"  expectancy/trade    {_fmt_money(metrics.expectancy)}")
        add(f"  average winner      {_fmt_money(metrics.avg_winner)}")
        add(f"  average loser       {_fmt_money(metrics.avg_loser)}")
        add(f"  largest winner      {_fmt_money(metrics.largest_winner)}")
        add(f"  largest loser       {_fmt_money(metrics.largest_loser)}")
        add(f"  profit factor       {metrics.profit_factor:.2f}")
        add(f"  average R           {metrics.avg_r:+.3f}   (total {metrics.total_r:+.1f}R)")
        add(f"  max drawdown        {_fmt_money(metrics.max_drawdown)} "
            f"({metrics.max_drawdown_pct:.1%})")
        add(f"  Sharpe (per-trade R)"
            f"{'n/a (<30 trades)' if metrics.sharpe is None else f'{metrics.sharpe:.3f}'}")
        add(f"  max consec wins     {metrics.max_consecutive_wins}")
        add(f"  max consec losses   {metrics.max_consecutive_losses}")
        add(f"  avg bars held       {metrics.avg_bars_held:.1f}")
        if metrics.cost_share_of_gross is not None:
            add(f"  cost / |gross|      {metrics.cost_share_of_gross:.1%}")
        add(f"  final equity        {_fmt_money(metrics.final_equity)}")

    if metrics.trades:
        add("")
        add("BREAKDOWN")
        lines.extend(_table(metrics.by_side, "by side"))
        lines.extend(_table(metrics.by_hour, "by entry hour"))
        lines.extend(_table(metrics.by_weekday, "by weekday"))
        lines.extend(_table(metrics.by_exit_reason, "by exit reason"))
        years = by_year(result.trades)
        if len(years) > 1:
            lines.extend(_table(years, "by year"))
            add("")
            add("    A rule whose profit comes from one year is not an edge. In this "
                "project's")
            add("    earlier work an ORB variant drew 140% of its net profit from 2022 "
                "alone and")
            add("    the 'it is really a volatility regime' rescue was falsified.")

    add("")
    add("REJECTED SIGNALS")
    counts: dict[str, int] = {}
    for rejection in result.rejections:
        key = f"{rejection.stage}/{rejection.reason.value}"
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        add("  none")
    for key, count in sorted(counts.items(), key=lambda kv: -kv[1])[:15]:
        add(f"  {count:>8,}  {key}")

    add("")
    add("EVIDENCE")
    if metrics.t_statistic is None:
        add("  t-statistic         n/a (too few trades)")
    else:
        bar = selection_bar(metrics.configurations_tried)
        verdict = "CLEARS" if metrics.clears_selection_bar else "does NOT clear"
        add(f"  t-statistic         {metrics.t_statistic:+.2f}")
        add(f"  configurations      {metrics.configurations_tried}")
        add(f"  selection bar       t >= {bar:.2f}  ->  {verdict} it")
    add("")
    add("  Prior: thirteen intraday NQ strategy families and well over a hundred")
    add("  configurations have been tested in this repository, and none survived")
    add("  out-of-sample. A positive DEV number is the expected behaviour of noise.")
    add("  A result means something only on data that has never informed a choice.")

    if result.notes or metrics.notes:
        add("")
        add("NOTES")
        for note in (*result.notes, *metrics.notes):
            add(f"  - {note}")

    add("=" * 78)
    return "\n".join(lines)


def _spec_lines(spec: dict) -> list[str]:
    out = [f"name           {spec.get('name')}", f"description    {spec.get('description')}"]
    for key in ("entry_conditions", "filters", "invalidation_conditions"):
        values = spec.get(key) or []
        label = key.replace("_", " ")
        if values:
            out.append(f"{label:<14} {values[0]}")
            out.extend(f"{'':<14} {value}" for value in values[1:])
        else:
            out.append(f"{label:<14} (none)")
    out.append(f"stop           {spec.get('stop_loss')}")
    out.append(f"target         {spec.get('profit_target')}")
    hours = spec.get("trading_hours") or {}
    out.append(
        f"hours          {hours.get('earliest_entry')}-{hours.get('latest_entry')} "
        f"(flat by {hours.get('force_flat_at')})"
    )
    out.append(f"max trades     {spec.get('max_trades_per_session')} per session")
    params = spec.get("params") or {}
    if params:
        out.append(f"params         {json.dumps(params, default=str, sort_keys=True)}")
    return out


def json_report(
    result: BacktestResult, metrics: Metrics | None = None, *, configurations_tried: int = 1
) -> str:
    metrics = metrics or compute_metrics(
        result.trades, starting_equity=result.starting_equity,
        configurations_tried=configurations_tried,
        marked_equity_curve=result.equity_curve,
    )
    payload = {
        "strategy": result.strategy_name,
        "spec": result.strategy_spec,
        "instrument": result.instrument,
        "timeframe": result.timeframe,
        "split": result.split,
        "period": {
            "start": result.start.isoformat() if result.start else None,
            "end": result.end.isoformat() if result.end else None,
            "bars": result.bars,
            "sessions": result.sessions,
            "warmup_bars": result.warmup_bars,
            "dataset_hash": result.dataset_hash,
        },
        "terminal": {
            "flat": result.ended_flat,
            "working_orders": result.working_orders_at_end,
        },
        "costs": result.cost_description,
        "metrics": metrics.to_dict(),
        "by_year": [
            {"label": b.label, "trades": b.trades, "net_pnl": b.net_pnl,
             "win_rate": b.win_rate, "avg_r": b.avg_r}
            for b in by_year(result.trades)
        ],
        "rejections": _rejection_counts(result),
        "floor_breached_at": (
            result.floor_breached_at.isoformat() if result.floor_breached_at else None
        ),
        "notes": list(result.notes) + list(metrics.notes),
    }
    return json.dumps(payload, indent=2, default=str)


def _rejection_counts(result: BacktestResult) -> list[dict]:
    counts: dict[tuple[str, str], int] = {}
    for rejection in result.rejections:
        key = (rejection.stage, rejection.reason.value)
        counts[key] = counts.get(key, 0) + 1
    return [
        {"stage": stage, "reason": reason, "count": count}
        for (stage, reason), count in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
