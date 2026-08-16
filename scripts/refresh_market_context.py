"""Refresh convenience fields and merge independently verified broker/vendor overrides.

This writes data snapshots only.  It never creates an order, an alert, or an
investment recommendation.  Yahoo fields are deliberately unverified.  A strict scanner
will suppress a setup until every component is supplied from its own dated, verified source.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scanner.market_context import (  # noqa: E402
    DEFAULT_CONTEXT_DIR,
    YahooContextProvider,
    merge_override,
    save_context,
)
from scanner.universe import resolve  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="*", default=["all"], help="universe preset or explicit symbols")
    parser.add_argument("--context-dir", type=Path, default=DEFAULT_CONTEXT_DIR)
    parser.add_argument(
        "--override-dir",
        type=Path,
        default=DEFAULT_CONTEXT_DIR / "overrides",
        help="Per-ticker licensed/broker JSON snapshots, e.g. AAPL.json",
    )
    args = parser.parse_args()

    provider = YahooContextProvider()
    for ticker in resolve(args.tickers):
        context = provider.fetch(ticker)
        override_path = args.override_dir / f"{ticker.replace('-', '')}.json"
        if override_path.exists():
            try:
                context = merge_override(context, json.loads(override_path.read_text(encoding="utf-8")))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"{ticker}: override rejected ({type(exc).__name__}: {exc})")
                continue
        path = save_context(context, args.context_dir)
        print(f"{ticker}: wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
