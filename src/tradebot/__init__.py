"""MNQ intraday futures trading system.

See PROJECT_SPEC.md for what this must and must not do, and docs/ARCHITECTURE.md for how
the pieces fit together.

The one invariant worth restating at the top of the package: a strategy returns an inert
`OrderIntent` and holds no broker handle. Orders reach a broker only via
`risk.RiskEngine` -> `execution.ExecutionEngine` -> `broker.GuardedBroker`, and the guard
refuses anything without a valid single-use risk token.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
