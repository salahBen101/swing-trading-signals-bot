from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timezone

import pytest

import tradebot.risk.state_store as state_store_module
from tradebot.risk.state_store import (
    FileRiskStateStore,
    RiskState,
    RiskStateBinding,
    RiskStateStoreError,
    StoredRiskState,
)


NOW = datetime(2026, 8, 24, 14, 30, tzinfo=timezone.utc)


def binding(**changes) -> RiskStateBinding:
    values = {
        "instrument": "MNQ",
        "risk_policy_sha256": "a" * 64,
        "broker_account_id": "paper-50k",
        "broker_name": "simulated",
        "broker_is_paper": True,
        "broker_execution_route": "simulated-local-paper",
        "deployment_context_id": "tradeify-growth-50k:evaluation:stage2",
    }
    values.update(changes)
    return RiskStateBinding(**values)


def risk_state(**changes) -> RiskState:
    values = {
        "equity": 49_950.0,
        "peak_equity": 50_100.0,
        "session_date": date(2026, 8, 24),
        "session_start_equity": 50_000.0,
        "daily_realized_pnl": -50.0,
        "daily_r": -0.25,
        "consecutive_losses": 1,
        "cooldown_until": NOW,
        "trades_today": 1,
        "halted": False,
        "halt_reason": "",
        "open_position_id": "position-1",
        "recent_fingerprints": {"intent-fingerprint": NOW},
        "revision": 7,
        "updated_at": NOW,
        "active_entry_order_id": "entry-2",
        "active_entry_intent_id": "intent-2",
        "active_entry_fingerprint": "entry-fingerprint-2",
        "active_entry_risk_usd": 150.0,
        "active_approved_quantity": 2,
        "entry_ever_filled": True,
        "consumed_entry_order_ids": ("entry-1",),
        "applied_trade_ids": ("trade-1",),
        "last_trade_id": "trade-1",
        "last_trade_closed_at": NOW,
        "last_trade_net_pnl_usd": -50.0,
        "last_trade_contracts": 2,
        "last_trade_was_loss": True,
        "last_trade_approved_risk_usd": 150.0,
        "last_trade_quantity": 2,
    }
    values.update(changes)
    return RiskState(**values)


def stored(**state_changes) -> StoredRiskState:
    return StoredRiskState(binding=binding(), state=risk_state(**state_changes))


def test_file_risk_state_round_trip_preserves_all_fields(tmp_path) -> None:
    path = tmp_path / "nested" / "risk-state.json"
    store = FileRiskStateStore(path)
    expected = stored()

    assert store.load() is None
    store.initialize(expected)

    assert store.load() == expected
    assert store.load(expected_binding=binding()) == expected
    document = json.loads(path.read_text(encoding="utf-8"))
    assert set(document) == {"schema_version", "stored_state"}
    assert document["schema_version"] == 1
    assert document["stored_state"]["binding"] == binding().to_dict()
    assert document["stored_state"]["state"]["consumed_entry_order_ids"] == [
        "entry-1"
    ]


def test_risk_state_payload_round_trip_keeps_legacy_snapshot_contract() -> None:
    original = risk_state()

    assert RiskState.from_dict(original.to_dict()) == original
    assert original.snapshot() == {
        "equity": 49_950.0,
        "peak_equity": 50_100.0,
        "session_date": "2026-08-24",
        "session_start_equity": 50_000.0,
        "daily_realized_pnl": -50.0,
        "daily_r": -0.25,
        "consecutive_losses": 1,
        "cooldown_until": NOW.isoformat(),
        "trades_today": 1,
        "halted": False,
        "halt_reason": "",
    }


def _write_document(path, document: dict) -> None:
    path.write_text(json.dumps(document, allow_nan=True), encoding="utf-8")


def _valid_document() -> dict:
    return {"schema_version": 1, "stored_state": stored().to_dict()}


def _extra_root(document: dict) -> None:
    document["unexpected"] = True


def _boolean_schema(document: dict) -> None:
    document["schema_version"] = True


def _unknown_binding_field(document: dict) -> None:
    document["stored_state"]["binding"]["account_label"] = "wrong"


def _missing_state_field(document: dict) -> None:
    del document["stored_state"]["state"]["trades_today"]


def _boolean_revision(document: dict) -> None:
    document["stored_state"]["state"]["revision"] = True


def _non_finite_equity(document: dict) -> None:
    document["stored_state"]["state"]["equity"] = float("nan")


def _naive_updated_at(document: dict) -> None:
    document["stored_state"]["state"]["updated_at"] = "2026-08-24T14:30:00"


def _duplicate_applied_trade(document: dict) -> None:
    document["stored_state"]["state"]["applied_trade_ids"] = ["trade-1", "trade-1"]


def _wrong_binding_flag_type(document: dict) -> None:
    document["stored_state"]["binding"]["broker_is_paper"] = 1


def _uppercase_policy_hash(document: dict) -> None:
    document["stored_state"]["binding"]["risk_policy_sha256"] = "A" * 64


@pytest.mark.parametrize(
    "mutate",
    [
        _extra_root,
        _boolean_schema,
        _unknown_binding_field,
        _missing_state_field,
        _boolean_revision,
        _non_finite_equity,
        _naive_updated_at,
        _duplicate_applied_trade,
        _wrong_binding_flag_type,
        _uppercase_policy_hash,
    ],
)
def test_file_store_rejects_corrupt_incompatible_or_loosely_typed_state(
    tmp_path, mutate
) -> None:
    path = tmp_path / "risk-state.json"
    document = _valid_document()
    mutate(document)
    _write_document(path, document)

    with pytest.raises(RiskStateStoreError, match="corrupt or incompatible"):
        FileRiskStateStore(path).load()


def test_file_store_rejects_a_runtime_binding_mismatch(tmp_path) -> None:
    store = FileRiskStateStore(tmp_path / "risk-state.json")
    store.initialize(stored())

    with pytest.raises(RiskStateStoreError, match="does not match this runtime"):
        store.load(expected_binding=binding(broker_account_id="another-account"))


@pytest.mark.parametrize(
    "bad_state, message",
    [
        (
            lambda: risk_state(consumed_entry_order_ids=("duplicate", "duplicate")),
            "duplicate identifiers",
        ),
        (
            lambda: risk_state(applied_trade_ids=tuple(f"trade-{i}" for i in range(257))),
            "at most 256",
        ),
        (
            lambda: risk_state(active_entry_order_id=None),
            "inactive entry metadata",
        ),
        (
            lambda: risk_state(last_trade_was_loss=False),
            "must agree",
        ),
        (
            lambda: risk_state(last_trade_quantity=3),
            "quantity fields must agree",
        ),
    ],
)
def test_risk_state_semantic_invariants_are_strict(bad_state, message) -> None:
    with pytest.raises(ValueError, match=message):
        bad_state().validate()


def test_file_store_wraps_invalid_objects_in_store_error(tmp_path) -> None:
    store = FileRiskStateStore(tmp_path / "risk-state.json")

    with pytest.raises(RiskStateStoreError, match="can only persist"):
        store.save(object())  # type: ignore[arg-type]
    with pytest.raises(RiskStateStoreError, match="could not persist"):
        store.save(stored(equity=float("inf")))


def test_atomic_replace_failure_preserves_previous_risk_state(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "risk-state.json"
    store = FileRiskStateStore(path)
    original = stored()
    replacement = StoredRiskState(
        binding=original.binding,
        state=replace(
            original.state,
            revision=original.state.revision + 1,
            equity=49_900.0,
            updated_at=NOW.replace(minute=31),
        ),
    )
    store.initialize(original)

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(state_store_module.os, "replace", fail_replace)
    with pytest.raises(RiskStateStoreError, match="could not persist"):
        store.save(replacement)

    assert store.load() == original
    assert list(tmp_path.glob(".risk-state.json.*.tmp")) == []


def test_compare_and_swap_rejects_a_stale_second_writer(tmp_path) -> None:
    path = tmp_path / "risk-state.json"
    first_store = FileRiskStateStore(path)
    second_store = FileRiskStateStore(path)
    original = stored()
    first_store.initialize(original)
    first_update = StoredRiskState(
        binding=original.binding,
        state=replace(
            original.state,
            revision=original.state.revision + 1,
            updated_at=NOW.replace(minute=31),
            equity=49_925.0,
        ),
    )
    stale_update = StoredRiskState(
        binding=original.binding,
        state=replace(
            original.state,
            revision=original.state.revision + 1,
            updated_at=NOW.replace(minute=32),
            equity=49_875.0,
        ),
    )

    first_store.save(first_update)
    with pytest.raises(RiskStateStoreError, match="stale personal-risk revision"):
        second_store.save(stale_update)

    assert first_store.load() == first_update


def test_concurrent_same_revision_updates_allow_exactly_one_winner(tmp_path) -> None:
    path = tmp_path / "risk-state.json"
    original = stored()
    FileRiskStateStore(path).initialize(original)
    candidates = tuple(
        StoredRiskState(
            binding=original.binding,
            state=replace(
                original.state,
                revision=original.state.revision + 1,
                updated_at=NOW.replace(minute=31 + index),
                equity=49_900.0 - index,
            ),
        )
        for index in range(2)
    )

    def attempt(candidate: StoredRiskState) -> bool:
        try:
            FileRiskStateStore(path).save(candidate)
        except RiskStateStoreError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(attempt, candidates))

    assert sorted(outcomes) == [False, True]
    assert FileRiskStateStore(path).load() in candidates


def test_binding_can_only_tighten_from_unbound_to_exact_identity(tmp_path) -> None:
    path = tmp_path / "risk-state.json"
    store = FileRiskStateStore(path)
    unbound = StoredRiskState(
        binding=binding(
            broker_account_id=None,
            broker_name=None,
            broker_is_paper=None,
            broker_execution_route=None,
        ),
        state=risk_state(),
    )
    store.initialize(unbound)
    tightened = StoredRiskState(
        binding=binding(),
        state=replace(
            unbound.state,
            revision=unbound.state.revision + 1,
            updated_at=NOW.replace(minute=31),
        ),
    )
    store.save(tightened)

    rotated = StoredRiskState(
        binding=replace(tightened.binding, broker_account_id="different-account"),
        state=replace(
            tightened.state,
            revision=tightened.state.revision + 1,
            updated_at=NOW.replace(minute=32),
        ),
    )
    with pytest.raises(RiskStateStoreError, match="binding cannot rotate"):
        store.save(rotated)

    assert store.load() == tightened
