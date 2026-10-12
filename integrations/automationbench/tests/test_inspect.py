"""Real Inspect eval/react/log execution against a deterministic non-network model."""

import copy

import pytest
from inspect_ai import eval
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ModelOutput,
    ModelUsage,
    get_model,
)
from inspect_ai.tool import ToolCall

from inspect_automationbench.task import RunConfig, automationbench_task


def output(text="finished", *, calls=None, reason="stop"):
    return ModelOutput(model="mockllm", choices=[ChatCompletionChoice(
        message=ChatMessageAssistant(content=text, tool_calls=calls), stop_reason=reason)],
        usage=ModelUsage(input_tokens=30, output_tokens=10, total_tokens=40))


def run_offline(tmp_path, scripted, config=None, domain="hr", task_id=5072):
    task = automationbench_task(domain=domain, task_id=task_id, config=config)
    model = get_model("mockllm/model", custom_outputs=scripted)
    log = eval(task, model=model, display="none", log_dir=str(tmp_path))[0]
    assert log.status == "success", str(log.error)
    assert log.samples and len(log.samples) == 1
    sample = log.samples[0]
    assert sample.error is None, str(sample.error)
    return sample


@pytest.mark.parametrize("domain, task_id", [("hr", 5072), ("finance", 4062)])
def test_inspect_native_tool_and_scorer_end_to_end(tmp_path, domain, task_id):
    sample = run_offline(tmp_path, [
        output("", calls=[ToolCall(id="call-1", function="api_search", arguments={
            "query": "slack" if domain == "hr" else "gmail", "top_k": 5,
        })], reason="tool_calls"),
        output(),
    ], domain=domain, task_id=task_id)
    record = sample.metadata["admission"]
    assert record["termination"] == "completed"
    assert record["model_calls"] == 2
    assert record["budget"]["known_used_tokens"] == 80
    assert len(record["native"]["tool_events"]) == 1
    assert record["native"]["scoring_error"] is None
    assert sample.scores["native_score"].value == record["native"]["task_completed_correctly"]
    assert sum(u.input_tokens + u.output_tokens for u in sample.model_usage.values()) == 80


@pytest.mark.parametrize("reason", ["max_tokens", "content_filter", "model_length"])
def test_invalid_response_does_not_execute_attached_tool(tmp_path, reason):
    sample = run_offline(tmp_path, [output("", calls=[
        ToolCall(id="must-not-run", function="api_search", arguments={"query": "slack", "top_k": 5}),
    ], reason=reason)])
    record = sample.metadata["admission"]
    assert record["termination"] == f"invalid_response: {reason}"
    assert record["native"]["tool_events"] == []


def test_entire_batch_validated_before_first_tool(tmp_path):
    sample = run_offline(tmp_path, [output("", calls=[
        ToolCall(id="valid", function="api_search", arguments={"query": "slack", "top_k": 5}),
        ToolCall(id="invalid", function="api_search", arguments={"unexpected": True}),
    ], reason="tool_calls")])
    record = sample.metadata["admission"]
    assert record["termination"] == "invalid_response: unexpected tool argument"
    assert record["native"]["tool_events"] == []


def test_missing_required_argument_stops_real_inspect_dispatch(tmp_path):
    sample = run_offline(tmp_path, [output("", calls=[
        ToolCall(id="missing-top-k", function="api_search", arguments={"query": "slack"}),
    ], reason="tool_calls")])
    record = sample.metadata["admission"]
    assert record["termination"] == "invalid_response: tool arguments fail native schema"
    assert record["native"]["tool_events"] == []


@pytest.mark.parametrize("architecture", ["wf-s", "wf-p"])
def test_team_usage_and_context_isolation(tmp_path, architecture):
    inputs = []

    def scripted(messages, tools, tool_choice, config):
        inputs.append(copy.deepcopy(messages))
        joined = "\n".join(m.text for m in messages)
        if "Role: records." in joined:
            return output("RECORDS_PRIVATE_FINDING")
        if "Role: policy_and_recent_updates." in joined:
            assert "RECORDS_PRIVATE_FINDING" not in joined
            return output("POLICY_PRIVATE_FINDING")
        assert "RECORDS_PRIVATE_FINDING" in joined
        assert "POLICY_PRIVATE_FINDING" in joined
        return output()

    sample = run_offline(tmp_path, scripted, RunConfig(architecture=architecture))
    record = sample.metadata["admission"]
    assert record["termination"] == "completed"
    assert record["model_calls"] == 3
    assert record["budget"]["known_used_tokens"] == 120
    assert len(inputs) == 3
    for messages in inputs:
        text = "\n".join(m.text for m in messages)
        assert '"assertions"' not in text
        assert '"initial_state"' not in text
    assert sum(u.input_tokens + u.output_tokens for u in sample.model_usage.values()) == 120
