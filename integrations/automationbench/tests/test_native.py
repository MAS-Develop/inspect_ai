"""Native environment equivalence, not model performance or successful task solving."""

import copy

import pytest
from automationbench.rubric import create_rubric, partial_credit, task_completed_correctly
from automationbench.runner import AutomationBenchEnv

from inspect_automationbench.contracts import AdmissionStop
from inspect_automationbench.native import NativeSession, select_task, verify_native


@pytest.mark.parametrize("domain, task_id", [("hr", 5072), ("finance", 4062)])
@pytest.mark.anyio
async def test_native_setup_and_read_replay(domain, task_id):
    verify_native()
    dataset, row = select_task(domain, task_id)
    untouched = copy.deepcopy(row)
    adapter = await NativeSession.create(dataset, row)
    native = AutomationBenchEnv(dataset=dataset, rubric=create_rubric(), toolset="api")
    reference = await native.setup_state(copy.deepcopy(row))
    assert reference["world"].model_dump() == adapter.state["world"].model_dump()
    assert reference["_task_contract_sha256"] == adapter.state["_task_contract_sha256"]
    args = {"query": "gmail"}
    expected = await native.call_tool("api_search", native.update_tool_args("api_search", args, [], reference), "reference")
    actual = await adapter.call("api_search", args)
    assert actual == expected.content
    assert reference["world"].model_dump() == adapter.state["world"].model_dump()
    reference_partial = partial_credit(reference)
    reference_pass = task_completed_correctly(reference)
    observed = adapter.finish()
    assert observed["partial_credit"] == reference_partial
    assert observed["task_completed_correctly"] == reference_pass
    assert observed["assertions"] == reference["_assertion_results"]
    assert row == untouched


@pytest.mark.anyio
async def test_role_world_isolation_and_write_denial():
    dataset, row = select_task("hr", 5072)
    analyst = await NativeSession.create(dataset, row)
    executor = await NativeSession.create(dataset, row)
    before = copy.deepcopy(executor.state["world"].model_dump())
    with pytest.raises(AdmissionStop, match="permission_denied"):
        await analyst.call("api_fetch", {"url": "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                                        "method": "POST"}, read_only=True)
    assert not analyst.events
    assert executor.state["world"].model_dump() == before
    assert analyst.state["world"] is not executor.state["world"]


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
