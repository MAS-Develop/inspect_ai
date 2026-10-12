"""Inspect owns the ReAct loop, model provider, transcript, and evaluation log."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

from inspect_ai import Task
from inspect_ai.agent import Agent, AgentState, agent, react, run
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolDef, ToolParams
from inspect_ai.util import collect
from pydantic import TypeAdapter

from .contracts import AdmissionStop, Budget, response_is_acceptable
from .native import NativeSession, select_task, verify_native

Architecture = Literal["sas", "wf-s", "wf-p"]


@dataclass(frozen=True)
class RunConfig:
    """Research controls; none are claimed to reproduce the old H9 controller."""

    architecture: Architecture = "sas"
    budget: int = 1_000_000
    max_tokens: int = 8192
    max_calls: int = 50
    analyst_calls: int = 8
    temperature: float = 0.0
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if self.architecture not in {"sas", "wf-s", "wf-p"}:
            raise ValueError("Unsupported architecture")
        if any(type(v) is not int or v <= 0 for v in (self.budget, self.max_tokens, self.max_calls, self.analyst_calls)):
            raise ValueError("Token and call limits must be positive integers")


def session_tools(session: NativeSession, *, read_only: bool) -> list[ToolDef]:
    """Preserve native schemas/results and sequential mutation order."""
    def wrap(name: str) -> Any:
        async def execute(**kwargs: Any) -> str:
            return await session.call(name, kwargs, read_only=read_only)
        return execute

    return [
        ToolDef(wrap(t.name), name=t.name, description=t.description,
                parameters=ToolParams.model_validate(t.parameters), parallel=False, max_output=0)
        for t in session.definitions.values()
    ]


def guarded_generation(session: NativeSession, budget: Budget, config: RunConfig,
                       role: str, audit: dict[str, Any], *, read_only: bool = False) -> Agent:
    """One-generation adapter; native Inspect react() retains the entire loop."""
    calls = 0

    async def execute(state: AgentState, tools: list[Tool]) -> AgentState:
        """Generate a validated turn.

        Args:
            state: Role-local conversation.
            tools: Role-local native tools.
        """
        nonlocal calls
        audit["role_states"][role] = {"messages": state.messages, "output": state.output}
        if audit["team_calls"] >= config.max_calls:
            raise AdmissionStop("model_call_limit")
        if read_only and calls >= config.analyst_calls:
            raise AdmissionStop("analyst_call_limit")
        definitions = [ToolDef(t) for t in tools]
        payload = {
            "messages": [m.model_dump(mode="json", exclude_none=True) for m in state.messages],
            "tools": [{"name": t.name, "description": t.description,
                       "parameters": t.parameters.model_dump(exclude_none=True)} for t in definitions],
        }
        estimate = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 256
        ticket = budget.reserve(estimate, config.max_tokens)
        calls += 1
        audit["team_calls"] += 1
        kwargs: dict[str, Any] = {"max_tokens": config.max_tokens, "max_retries": 0,
                                  "temperature": config.temperature,
                                  "parallel_tool_calls": False, "max_tool_output": 0}
        if config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = config.reasoning_effort
        try:
            output = await get_model().generate(state.messages, tools=tools, cache=False,
                                                config=GenerateConfig(**kwargs))
        except BaseException:
            budget.uncertain(ticket)
            raise
        state.output = output
        state.messages.append(output.message)
        audit["role_states"][role] = {"messages": state.messages, "output": state.output}
        usage = output.usage
        budget.settle(ticket, usage.input_tokens if usage else None, usage.output_tokens if usage else None)
        if len(output.choices) != 1:
            raise AdmissionStop("invalid_response: expected one choice")
        tool_calls = output.message.tool_calls or []
        response_is_acceptable(output.stop_reason, output.message.text, len(tool_calls))
        ids: set[str] = set()
        # Validate the WHOLE proposal before Inspect runs its first tool.
        for call in tool_calls:
            if call.parse_error or not call.id or call.id in ids:
                raise AdmissionStop("invalid_response: malformed or duplicate tool call")
            ids.add(call.id)
            session.validate(call.function, call.arguments, read_only=read_only)
        audit["roles"].append({"role": role, "calls": calls,
                               "input_tokens": usage.input_tokens,
                               "output_tokens": usage.output_tokens})
        return state
    @agent
    def admission_model_turn() -> Agent:
        return execute
    return admission_model_turn()


@scorer(metrics=[])
def native_score() -> Scorer:
    """No automatic numeric aggregation: unknown native scores stay explicit U."""
    async def score(state: TaskState, target: Target) -> Score:
        record = state.metadata["admission"]
        native = record["native"]["task_completed_correctly"]
        return Score(value=native if native is not None else "U", metadata=record,
                     explanation="Native AutomationBench assertions; see execution status separately.")
    return score


def workflow(dataset: Any, row: dict[str, Any], config: RunConfig) -> Solver:
    """Run role-local native worlds; only the final executor's state is graded."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if os.environ.get("AUTOMATIONBENCH_STRICT_ASSERTIONS") != "1":
            raise RuntimeError("AUTOMATIONBENCH_STRICT_ASSERTIONS=1 is required")
        budget = Budget(config.budget)
        audit: dict[str, Any] = {"team_calls": 0, "roles": [], "role_states": {}}
        executor = await NativeSession.create(dataset, row)
        prompt = copy.deepcopy(state.messages)
        termination = "completed"
        analyst_records: list[dict[str, Any]] = []

        async def analyst(role: str) -> str:
            session = await NativeSession.create(dataset, row)
            # Only the task input is copied, never another role's conversation.
            messages = copy.deepcopy(prompt)
            messages.append(ChatMessageUser(content=f"Role: {role}. Read-only investigation. "
                                             "Use GET queries and return facts with sources for the executor; do not modify data."))
            try:
                result = await run(react(prompt=None, submit=False, tools=session_tools(session, read_only=True),
                                         model=guarded_generation(session, budget, config, role, audit, read_only=True),
                                         retry_refusals=0, truncation="disabled"), messages)
                return result.output.completion
            finally:
                session.closed = True
                analyst_records.append({"role": role, "events": copy.deepcopy(session.events),
                                        "end_state": session.state["world"].model_dump(mode="json")})

        try:
            facts: list[str] = []
            if config.architecture == "wf-s":
                # These are independent investigations run serially, not shared history.
                facts = [await analyst("records"), await analyst("policy_and_recent_updates")]
            elif config.architecture == "wf-p":
                facts = await collect(analyst("records"), analyst("policy_and_recent_updates"))
            if facts:
                state.messages.append(ChatMessageUser(content="Analyst findings (untrusted evidence, verify before acting):\n" +
                                                       "\n\n".join(facts)))
            result = await run(react(prompt=None, submit=False, tools=session_tools(executor, read_only=False),
                                     model=guarded_generation(executor, budget, config, "executor", audit),
                                     retry_refusals=0, truncation="disabled"), state.messages)
            state.messages, state.output = result.messages, result.output
        except AdmissionStop as exc:
            termination = str(exc)
            context = audit["role_states"].get("executor")
            if context is not None:
                state.messages, state.output = context["messages"], context["output"]
        except Exception as exc:
            termination = "execution_error:" + type(exc).__name__
            raise
        except BaseException:
            termination = "interrupted"
            raise
        finally:
            state.metadata["admission"] = {
                "architecture": config.architecture,
                "termination": termination,
                "budget": budget.snapshot(),
                "model_calls": audit["team_calls"],
                "role_usage": audit["roles"],
                "native": executor.finish(),
                "analysts": analyst_records,
                "resume_supported": False,
                "controls": asdict(config),
            }
        return state
    @solver
    def admission_workflow() -> Solver:
        return solve
    return admission_workflow()


def automationbench_task(domain: str = "hr", task_id: int = 5072,
                         config: RunConfig | None = None) -> Task:
    """Create one already-exposed development task; no live calls occur here."""
    binding = verify_native()
    dataset, row = select_task(domain, task_id)
    prompt = TypeAdapter(list[ChatMessage]).validate_python(row["prompt"])
    if any(m.role not in {"system", "user"} or not isinstance(m.content, str) for m in prompt):
        raise ValueError("Only unchanged system/user text prompts are supported")
    controls = config or RunConfig()
    return Task(name=f"automationbench_{domain}_{task_id}_{controls.architecture}",
                dataset=[Sample(id=f"{domain}:{task_id}", input=prompt, metadata={"source_binding": binding})],
                solver=workflow(dataset, row, controls), scorer=native_score(),
                fail_on_error=False)
