"""Native parity, with controlled Sheets IDs only inside paired setup tests."""

import copy
import itertools
import json
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from automationbench.rubric import create_rubric, partial_credit, task_completed_correctly
from automationbench.runner import AutomationBenchEnv
from automationbench.schema.google_sheets import base as sheets_base

from inspect_automationbench.contracts import AdmissionStop
from inspect_automationbench.native import NativeSession, select_task, verify_native


@contextmanager
def paired_sheets_ids(monkeypatch):
    """Replay the same ID source, without editing seeds or erasing world fields.

    The pinned GoogleSheetsRecord default factory calls this module's uuid.uuid4.
    Replace the module reference, NOT the process-wide uuid.uuid4 function. Each
    context resets the sequence; the native factory and all other setup code run
    unchanged. This helper is never imported by the production adapter.
    """
    sequence = itertools.count(1)
    with monkeypatch.context() as patch:
        patch.setattr(
            sheets_base,
            "uuid",
            SimpleNamespace(uuid4=lambda: uuid.UUID(int=next(sequence), version=4)),
        )
        yield


@pytest.mark.parametrize("domain, task_id", [("hr", 5072), ("finance", 4062)])
@pytest.mark.anyio
async def test_native_setup_and_read_replay(domain, task_id, monkeypatch):
    verify_native()
    dataset, row = select_task(domain, task_id)
    untouched = copy.deepcopy(row)
    original_uuid = sheets_base.uuid
    with paired_sheets_ids(monkeypatch):
        adapter = await NativeSession.create(dataset, row)
    assert sheets_base.uuid is original_uuid
    native = AutomationBenchEnv(dataset=dataset, rubric=create_rubric(), toolset="api")
    with paired_sheets_ids(monkeypatch):
        reference = await native.setup_state(copy.deepcopy(row))
    assert sheets_base.uuid is original_uuid
    assert reference["world"].model_dump() == adapter.state["world"].model_dump()
    assert reference["initial_state"] == adapter.state["initial_state"]
    assert reference["_task_contract_sha256"] == adapter.state["_task_contract_sha256"]
    before = copy.deepcopy(adapter.state["world"].model_dump())
    args = {"query": "slack" if domain == "hr" else "gmail", "top_k": 5}
    expected = await native.call_tool(
        "api_search", native.update_tool_args("api_search", args, [], reference), "reference"
    )
    actual = await adapter.call("api_search", args)
    assert actual == expected.content
    assert json.loads(actual)["count"] > 0
    assert reference["world"].model_dump() == adapter.state["world"].model_dump() == before
    reference_partial = partial_credit(reference)
    reference_pass = task_completed_correctly(reference)
    observed = adapter.finish()
    assert observed["scoring_error"] is None
    assert observed["partial_credit"] == reference_partial
    assert observed["task_completed_correctly"] == reference_pass
    assert observed["assertions"] == reference["_assertion_results"]
    assert len(observed["tool_events"]) == 1
    assert row == untouched


@pytest.mark.parametrize("domain, task_id", [("hr", 5072), ("finance", 4062)])
@pytest.mark.anyio
async def test_uncontrolled_sessions_keep_native_ids_and_isolation(domain, task_id):
    dataset, row = select_task(domain, task_id)
    untouched = copy.deepcopy(row)
    first = await NativeSession.create(dataset, row)
    second = await NativeSession.create(dataset, row)
    left = first.state["world"].google_sheets.rows
    right = second.state["world"].google_sheets.rows
    assert left and len(left) == len(right)
    assert len({r.id for r in left}) == len(left)
    assert len({r.id for r in right}) == len(right)
    for record in [*left, *right]:
        assert uuid.UUID(record.id).version == 4
    before = copy.deepcopy(second.state["world"].model_dump())
    left[0].cells["test_only_isolation_probe"] = "not shared"
    assert second.state["world"].model_dump() == before
    assert first.state["world"] is not second.state["world"]
    assert row == untouched


@pytest.mark.anyio
async def test_role_world_isolation_and_write_denial():
    dataset, row = select_task("hr", 5072)
    analyst = await NativeSession.create(dataset, row)
    executor = await NativeSession.create(dataset, row)
    analyst_before = copy.deepcopy(analyst.state["world"].model_dump())
    executor_before = copy.deepcopy(executor.state["world"].model_dump())
    arguments = {
        "url": "https://slack.com/api/chat.postMessage",
        "method": "POST",
        "params": None,
        "body": json.dumps({"channel": "C_ENG", "text": "must not execute"}),
    }
    # Show that this is a well-formed proposal; only the analyst permission denies it.
    analyst.validate("api_fetch", arguments, read_only=False)
    with pytest.raises(AdmissionStop, match="permission_denied"):
        await analyst.call("api_fetch", arguments, read_only=True)
    assert not analyst.events
    assert analyst.state["world"].model_dump() == analyst_before
    assert executor.state["world"].model_dump() == executor_before
    assert analyst.state["world"] is not executor.state["world"]


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("api_search", {"query": "slack"}),
        ("api_fetch", {"method": "GET", "url": "https://slack.com/api/conversations.list"}),
        ("api_search", {"query": "slack", "top_k": "5"}),
    ],
)
@pytest.mark.anyio
async def test_runtime_schema_is_not_relaxed(name, arguments):
    dataset, row = select_task("hr", 5072)
    session = await NativeSession.create(dataset, row)
    before = copy.deepcopy(session.state["world"].model_dump())
    with pytest.raises(AdmissionStop, match="invalid_response"):
        await session.call(name, arguments)
    assert not session.events
    assert session.state["world"].model_dump() == before


@pytest.mark.anyio
async def test_scoring_exception_is_unknown(monkeypatch):
    dataset, row = select_task("hr", 5072)
    session = await NativeSession.create(dataset, row)

    def broken(state):
        raise RuntimeError("injected scorer failure")

    monkeypatch.setattr("automationbench.rubric.partial_credit", broken)
    result = session.finish()
    assert result["task_completed_correctly"] is None
    assert result["partial_credit"] is None
    assert "injected scorer failure" in result["scoring_error"]


def test_no_holdout_or_unreviewed_tasks():
    with pytest.raises(ValueError):
        select_task("hr", 5036)
