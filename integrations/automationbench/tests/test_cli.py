"""Exercise the actual default CLI under pytest's blocked internet sockets."""

import json
import socket
import sys

import pytest
from inspect_ai.log import read_eval_log
from pytest_socket import SocketBlockedError

from inspect_automationbench.__main__ import main


@pytest.mark.parametrize("domain, task_id", [("hr", 5072), ("finance", 4062)])
@pytest.mark.parametrize("architecture", ["sas", "wf-s", "wf-p"])
def test_default_cli_exercises_native_tool(tmp_path, monkeypatch, capsys, domain, task_id, architecture):
    monkeypatch.setattr(sys, "argv", [
        "inspect_automationbench", "--domain", domain, "--task-id", str(task_id),
        "--architecture", architecture, "--log-dir", str(tmp_path),
    ])
    main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["mode"] == "scripted_offline"
    assert summary["wiring_completed"] is True
    assert summary["termination"] == "completed"
    assert summary["native_scoring_error"] is None
    assert summary["budget"]["strict_budget_valid"] is False
    log = read_eval_log(summary["log"])
    record = log.samples[0].metadata["admission"]
    roles = 1 if architecture == "sas" else 3
    assert record["model_calls"] == 2 * roles
    assert record["budget"]["known_used_tokens"] == 40 * roles
    assert len(record["native"]["tool_events"]) == 1
    assert record["native"]["tool_events"][0]["name"] == "api_search"
    assert len(record["analysts"]) == roles - 1
    assert all(len(analyst["events"]) == 1 for analyst in record["analysts"])
    assert summary["native_task_success"] == record["native"]["task_completed_correctly"]


@pytest.mark.parametrize(
    "options, termination",
    [(["--budget", "1"], "budget_exhausted_or_unknown"),
     (["--max-calls", "1"], "model_call_limit")],
)
def test_cli_does_not_report_early_stop_as_success(tmp_path, monkeypatch, capsys, options, termination):
    monkeypatch.setattr(sys, "argv", ["inspect_automationbench", "--log-dir", str(tmp_path), *options])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["wiring_completed"] is False
    assert summary["termination"] == termination


def test_cli_scoring_error_is_unknown_and_nonzero(tmp_path, monkeypatch, capsys):
    def broken(state):
        raise RuntimeError("injected CLI scorer failure")

    monkeypatch.setattr("automationbench.rubric.partial_credit", broken)
    monkeypatch.setattr(sys, "argv", ["inspect_automationbench", "--log-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["wiring_completed"] is False
    assert summary["native_task_success"] is None
    assert "injected CLI scorer failure" in summary["native_scoring_error"]


@pytest.mark.parametrize("options", [["--model", "external/not-admitted"], ["--live"]])
def test_cli_requires_both_live_opt_ins(monkeypatch, options):
    monkeypatch.setattr(sys, "argv", ["inspect_automationbench", *options])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 2


@pytest.mark.parametrize("family", [socket.AF_INET, socket.AF_INET6])
def test_acceptance_blocks_internet_sockets(family):
    with pytest.raises(SocketBlockedError), socket.socket(family):
        pytest.fail("Internet sockets must be blocked during admission")
