"""Use the frozen AutomationBench runtime; never send private state to a model."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from .contracts import AdmissionStop, check_call, fingerprint

NATIVE_COMMIT = "4a8e1061254004d9dac807054eed33fad7d1ff14"
NATIVE_BLOBS = {
    "runner.py": "f7a7fcaab322b18018594a99a5aaa0007d414819",
    "rubric/__init__.py": "70d3287403a620063d9a8d01c3aeb6ae954ae0ba",
    "tools/api/__init__.py": "6a8ffef18e6f0c5d5d0744a7bd818ceee008a755",
    "tools/api/fetch.py": "883a6c90ff7f27ac5a3787b9db15e857153342cc",
}


def verify_native() -> dict[str, Any]:
    """Verify the installed version and critical files, not merely a version label."""
    import automationbench

    if importlib.metadata.version("automation-bench") != "1.0.6":
        raise RuntimeError("This admission adapter requires AutomationBench 1.0.6")
    root = Path(automationbench.__file__).parent
    actual = {}
    for name, expected in NATIVE_BLOBS.items():
        data = (root / name).read_bytes()
        actual[name] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual[name] != expected:
            raise RuntimeError(f"Native source drift: {name}")
    return {"required_commit": NATIVE_COMMIT, "verified_critical_blobs": actual}


def info_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Decode native JSON info without replacing missing or invalid information."""
    info = row["info"]
    info = json.loads(info) if isinstance(info, str) else info
    if not isinstance(info, dict):
        raise ValueError("Native task info is not an object")
    return info


def make_environment(dataset: Any) -> Any:
    """Let the official environment format questions into model prompts."""
    from automationbench.rubric import create_rubric
    from automationbench.runner import AutomationBenchEnv

    return AutomationBenchEnv(dataset=dataset, rubric=create_rubric(), toolset="api")


def select_task(domain: str, task_id: int) -> tuple[Any, dict[str, Any]]:
    """Select one exposed example and apply native question/prompt formatting."""
    from automationbench.domains import get_domain_dataset

    if domain not in {"hr", "finance"}:
        raise ValueError("Initial admission is restricted to hr and finance")
    if (domain, task_id) not in {("hr", 5072), ("finance", 4062)}:
        raise ValueError("Only the two already-exposed development tasks are admitted")
    dataset = get_domain_dataset(domain)
    matches = [i for i, row in enumerate(dataset) if str(row.get("example_id")) == str(task_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected one native example_id={task_id}; found {len(matches)}")
    selected = dataset.select(matches)
    env = make_environment(selected)
    # Raw dataset rows contain question and JSON-encoded info, not runtime prompt/state.
    row = copy.deepcopy(dict(env.dataset[0]))
    row["info"] = info_dict(row)
    if not row["info"].get("assertions"):
        raise ValueError("Native task has no assertions")
    if not isinstance(row.get("prompt"), list):
        raise ValueError("Native formatter did not produce a chat prompt")
    return selected, row


class NativeSession:
    """One native world and native tool registry per role, never shared writes."""

    def __init__(self, env: Any, row: dict[str, Any], state: dict[str, Any]) -> None:
        self.env = env
        self.row = copy.deepcopy(row)
        self.state = state
        self.events: list[dict[str, Any]] = []
        self.definitions = {t.name: t for t in state["tool_defs"]}
        self.closed = False

    @classmethod
    async def create(cls, dataset: Any, row: dict[str, Any]) -> NativeSession:
        """Delegate normalization, allowed services, and world setup to upstream."""
        env = make_environment(dataset)
        prepared = copy.deepcopy(row)
        prepared["info"] = info_dict(prepared)
        state = await env.setup_state(prepared)
        if state is None or "world" not in state:
            raise RuntimeError("Native initialization returned no world")
        return cls(env, row, state)

    def validate(self, name: str, arguments: dict[str, Any], *, read_only: bool) -> None:
        """Check a complete tool proposal before any operation is dispatched."""
        if self.closed:
            raise AdmissionStop("session_closed")
        if name not in self.definitions:
            raise AdmissionStop("invalid_response: unknown tool")
        check_call(name, arguments, self.definitions[name].parameters, read_only=read_only)

    async def call(self, name: str, arguments: dict[str, Any], *, read_only: bool = False) -> str:
        """Use the native argument injection and native tool dispatcher exactly once."""
        self.validate(name, arguments, read_only=read_only)
        args = self.env.update_tool_args(name, copy.deepcopy(arguments), [], self.state)
        result = await self.env.call_tool(name, args, f"admission-{len(self.events)}")
        content = result.content
        if not isinstance(content, str):
            raise RuntimeError("This text-only adapter does not accept multimodal tool results")
        self.events.append({"name": name, "arguments": copy.deepcopy(arguments), "result": content})
        return content

    def finish(self) -> dict[str, Any]:
        """Retain native scoring; assertion exceptions remain unknown, not zero."""
        import os

        from automationbench.rubric import partial_credit, task_completed_correctly

        if os.environ.get("AUTOMATIONBENCH_STRICT_ASSERTIONS") != "1":
            raise RuntimeError("Set AUTOMATIONBENCH_STRICT_ASSERTIONS=1 so scorer errors are not zeros")
        self.closed = True
        result: dict[str, Any] = {
            "task_completed_correctly": None,
            "partial_credit": None,
            "scoring_error": None,
            "task_contract_sha256": self.state.get("_task_contract_sha256"),
            "initial_state": copy.deepcopy(self.state["initial_state"]),
            "end_state": self.state["world"].model_dump(mode="json"),
            "tool_events": copy.deepcopy(self.events),
        }
        try:
            result["partial_credit"] = partial_credit(self.state)
            result["task_completed_correctly"] = task_completed_correctly(self.state)
            result["assertions"] = copy.deepcopy(self.state.get("_assertion_results", []))
        except Exception as exc:
            result["partial_credit"] = None
            result["scoring_error"] = type(exc).__name__ + ": " + str(exc)
        result["initial_state_sha256"] = fingerprint(result["initial_state"])
        result["end_state_sha256"] = fingerprint(result["end_state"])
        return result
