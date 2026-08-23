"""The non-bypass property, asserted structurally.

> *"The trading strategy must NEVER have direct unrestricted access to the broker. All
> orders must pass through the risk engine. The broker execution layer must reject any
> order violating these limits regardless of what the strategy requests."*

Three layers, per PROJECT_SPEC §3.1, each tested here:

1. the strategy layer cannot *name* a broker — a static import check over the AST
2. the risk engine mints a single-use, bound token
3. the guard demands one and re-validates independently

The static check is the important one, because layers 2 and 3 protect against a strategy
misbehaving, while layer 1 protects against a future edit quietly wiring around them.
"""

from __future__ import annotations

import ast
import inspect
import pkgutil
from pathlib import Path

import pytest

import tradebot.strategy as strategy_pkg
from tradebot.broker.guarded import GuardedBroker
from tradebot.core.models import OrderIntent
from tradebot.strategy.base import Strategy, StrategyContext
from tradebot.strategy.registry import build_strategy, known_strategies

STRATEGY_DIR = Path(strategy_pkg.__file__).parent
FORBIDDEN_FOR_STRATEGIES = {"broker", "execution", "journal"}


def _strategy_modules() -> list[Path]:
    return sorted(p for p in STRATEGY_DIR.glob("*.py") if p.name != "__init__.py")


def _imported_names(tree: ast.AST) -> set[str]:
    """Every module name a file imports, relative or absolute, at any depth."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.update(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.update(node.module.split("."))
            for alias in node.names:
                names.add(alias.name)
    return names


@pytest.mark.parametrize("path", _strategy_modules(), ids=lambda p: p.name)
def test_no_strategy_module_imports_a_broker_or_the_execution_engine(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    leaked = _imported_names(tree) & FORBIDDEN_FOR_STRATEGIES
    assert not leaked, (
        f"{path.name} imports {sorted(leaked)}. A strategy must not be able to name the "
        f"thing that sends orders (PROJECT_SPEC section 3.1, layer 1)."
    )


def test_the_import_check_would_actually_catch_a_leak(tmp_path):
    """A guard that cannot fail proves nothing."""
    leaky = tmp_path / "leaky_strategy.py"
    leaky.write_text("from ..broker.simulated import SimulatedBroker\n", encoding="utf-8")
    tree = ast.parse(leaky.read_text(encoding="utf-8"))
    assert _imported_names(tree) & FORBIDDEN_FOR_STRATEGIES


def test_the_strategy_package_itself_stays_clean():
    for module in pkgutil.iter_modules([str(STRATEGY_DIR)]):
        source = (STRATEGY_DIR / f"{module.name}.py").read_text(encoding="utf-8")
        assert "place_order" not in source, f"{module.name} references place_order"
        assert "GuardedBroker" not in source, f"{module.name} references the broker"


@pytest.mark.parametrize("name", known_strategies())
def test_a_constructed_strategy_holds_no_object_that_can_send_an_order(name):
    strategy = build_strategy(name)
    for attr, value in vars(strategy).items():
        assert not hasattr(value, "place_order"), (
            f"{name}.{attr} exposes place_order"
        )
        assert not isinstance(value, GuardedBroker)


@pytest.mark.parametrize("name", known_strategies())
def test_a_strategy_returns_an_inert_intent_not_an_action(name):
    """`on_bar` may only return data. An intent has no method that does anything."""
    signature = inspect.signature(build_strategy(name).on_bar)
    assert list(signature.parameters) == ["ctx"]

    for attr in dir(OrderIntent):
        if attr.startswith("_"):
            continue
        assert attr not in ("send", "submit", "execute", "place", "fill", "cancel")


def test_the_strategy_context_cannot_reach_past_the_current_bar():
    """Layer 1's other half: no accessor returns a future row."""
    fields = set(StrategyContext.__slots__)
    assert "_frame" in fields and "_bars" in fields, "history stays private"

    public = {n for n in dir(StrategyContext) if not n.startswith("_")}
    assert public == {
        "bars", "feature_history", "previous", "f", "reject",
        "i", "timestamp", "bar", "features", "instrument", "equity",
        "session_date", "trades_this_session", "rejections",
    }, f"unexpected public surface on StrategyContext: {sorted(public)}"


def test_history_accessors_are_bounded_by_the_current_index(multi_session_bars):
    from tradebot.features.pipeline import build_features
    from tradebot.instruments.registry import get_instrument

    frame = build_features(multi_session_bars).frame
    i = 200
    ctx = StrategyContext(
        i=i, timestamp=frame.index[i].to_pydatetime(), bar=multi_session_bars.iloc[i],
        features=frame.iloc[i], instrument=get_instrument("MNQ"), equity=50_000.0,
        session_date=frame.index[i].date(), trades_this_session=0,
        _frame=frame, _bars=multi_session_bars,
    )
    assert len(ctx.bars) == i + 1
    assert ctx.bars.index[-1] == frame.index[i]
    assert len(ctx.feature_history) == i + 1
    assert ctx.previous(1) is not None
    assert ctx.previous(i + 5) is None


def test_the_execution_engine_is_only_ever_handed_a_guarded_broker():
    """Layer 3 is only worth anything if the raw adapter never gets passed around."""
    from tradebot.execution.engine import ExecutionEngine

    annotation = inspect.signature(ExecutionEngine.__init__).parameters["broker"].annotation
    assert "GuardedBroker" in str(annotation)


def test_the_guard_requires_a_token_positionally():
    params = inspect.signature(GuardedBroker.place_order).parameters
    assert list(params)[:3] == ["self", "order", "token"]
    assert params["token"].default is inspect.Parameter.empty
    assert params["token"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
