"""Unit-level admission contracts. These tests do not require either agent SDK."""

import pytest

from inspect_automationbench.contracts import AdmissionStop, Budget, check_call, response_is_acceptable

SCHEMA = {"type": "object", "properties": {"method": {"type": "string"}, "url": {"type": "string"}}, "required": ["url"]}


@pytest.mark.parametrize("reason", ["max_tokens", "model_length", "content_filter", "unknown"])
def test_bad_finish_cannot_authorize_tools(reason):
    with pytest.raises(AdmissionStop):
        response_is_acceptable(reason, "plausible answer", 1)


@pytest.mark.parametrize("text, calls, reason", [("", 0, "stop"), (" ", 0, "stop"), ("answer", 0, "tool_calls")])
def test_invalid_empty_or_inconsistent_response(text, calls, reason):
    with pytest.raises(AdmissionStop):
        response_is_acceptable(reason, text, calls)


def test_valid_tool_response_and_final_text():
    response_is_acceptable("tool_calls", "", 1)
    response_is_acceptable("stop", "done", 0)


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "post", "get"])
def test_analyst_writes_rejected(method):
    with pytest.raises(AdmissionStop):
        check_call("api_fetch", {"url": "x", "method": method}, SCHEMA, read_only=True)


def test_read_only_get_accepted():
    check_call("api_fetch", {"url": "x", "method": "GET"}, SCHEMA, read_only=True)


@pytest.mark.parametrize("args", [{"url": "x", "world": {}}, {"url": 1}, {"url": "x", "unexpected": 1}, {}])
def test_malformed_arguments_rejected(args):
    with pytest.raises(AdmissionStop):
        check_call("api_fetch", args, SCHEMA, read_only=False)


def test_empty_object_native_sentinel_preserved():
    check_call("api_fetch", {"url": "x", "method": {}}, SCHEMA, read_only=False)


def test_outstanding_reservations_prevent_parallel_double_spend():
    budget = Budget(100)
    first = budget.reserve(40, 20)
    with pytest.raises(AdmissionStop):
        budget.reserve(40, 20)
    budget.settle(first, 20, 10)
    second = budget.reserve(40, 20)
    budget.settle(second, 30, 10)
    assert budget.used == 70
    assert budget.snapshot()["usage_complete"] is True
    assert budget.snapshot()["strict_budget_valid"] is False


@pytest.mark.parametrize("input_tokens, output_tokens", [(None, 5), (5, None), (-1, 5), (True, 5)])
def test_unknown_or_invalid_usage_is_not_zero(input_tokens, output_tokens):
    budget = Budget(100)
    ticket = budget.reserve(20, 10)
    with pytest.raises(AdmissionStop, match="usage_unknown"):
        budget.settle(ticket, input_tokens, output_tokens)
    assert budget.snapshot()["outstanding_reserved_tokens"] == 30
    assert budget.snapshot()["within_observed_limit"] is None
    with pytest.raises(AdmissionStop):
        budget.reserve(1, 1)


def test_uncertain_call_blocks_new_admission():
    budget = Budget(100)
    ticket = budget.reserve(30, 20)
    budget.uncertain(ticket)
    with pytest.raises(AdmissionStop):
        budget.reserve(1, 1)


def test_actual_overrun_is_exposed_not_clamped():
    budget = Budget(100)
    ticket = budget.reserve(30, 20)
    with pytest.raises(AdmissionStop, match="budget_overrun"):
        budget.settle(ticket, 110, 10)
    assert budget.used == 120
    assert budget.snapshot()["strict_budget_valid"] is False


def test_duplicate_settlement_rejected():
    budget = Budget(100)
    ticket = budget.reserve(10, 10)
    budget.settle(ticket, 5, 5)
    with pytest.raises(ValueError):
        budget.settle(ticket, 5, 5)
