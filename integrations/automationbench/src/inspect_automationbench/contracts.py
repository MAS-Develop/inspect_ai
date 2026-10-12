"""Fail-closed admission rules, independent of model and benchmark SDKs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator


class AdmissionStop(RuntimeError):
    """A recorded stop, not a model task-success judgment."""


def check_call(name: str, arguments: dict[str, Any], schema: dict[str, Any], *, read_only: bool) -> None:
    """Reject malformed arguments and disallowed analyst operations before dispatch."""
    if "world" in arguments:
        raise AdmissionStop("invalid_response: world is a host-only argument")
    allowed = set(schema.get("properties", {}))
    if set(arguments) - allowed:
        raise AdmissionStop("invalid_response: unexpected tool argument")
    # Mirror the frozen runner's empty-object sentinel before schema validation.
    normalized = {k: v for k, v in arguments.items() if not (isinstance(v, dict) and not v)}
    if list(Draft202012Validator(schema).iter_errors(normalized)):
        raise AdmissionStop("invalid_response: tool arguments fail native schema")
    if read_only:
        if name not in {"api_search", "api_fetch", "base64_encode"}:
            raise AdmissionStop("permission_denied: analyst tool")
        if name == "api_fetch" and normalized.get("method", "GET") != "GET":
            raise AdmissionStop("permission_denied: analysts may only issue GET")


def response_is_acceptable(stop_reason: str, text: str, calls: int) -> None:
    """No tools may run on a truncated, filtered, empty, or inconsistent response."""
    if stop_reason not in {"stop", "tool_calls"}:
        raise AdmissionStop(f"invalid_response: {stop_reason}")
    if stop_reason == "tool_calls" and not calls:
        raise AdmissionStop("invalid_response: missing tool calls")
    if not calls and not text.strip():
        raise AdmissionStop("invalid_response: empty response")


def fingerprint(value: Any) -> str:
    """Hash JSON data without converting unknown objects to misleading strings."""
    import hashlib

    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass
class Budget:
    """Single-event-loop reservation ledger; estimates are NOT certified hard bounds.

    reserve/settle contain no awaits, so their multi-field invariants cannot
    interleave on the sample's event loop. This is not a cross-process ledger.
    Unknown billable usage retains its reservation and blocks further admission.
    """

    limit: int
    used: int = 0
    unknown: bool = False
    _next: int = 0
    _pending: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit <= 0:
            raise ValueError("budget must be a positive integer")

    def reserve(self, input_estimate: int, max_output: int) -> int:
        if type(input_estimate) is not int or input_estimate < 0 or type(max_output) is not int or max_output <= 0:
            raise ValueError("invalid token reservation")
        amount = input_estimate + max_output
        if self.unknown or self.used + sum(self._pending.values()) + amount > self.limit:
            raise AdmissionStop("budget_exhausted_or_unknown")
        ticket = self._next
        self._next += 1
        self._pending[ticket] = amount
        return ticket

    def settle(self, ticket: int, input_tokens: int | None, output_tokens: int | None) -> None:
        if ticket not in self._pending:
            raise ValueError("unknown or already-settled budget ticket")
        if any(type(v) is not int or v < 0 for v in (input_tokens, output_tokens)):
            self.unknown = True
            raise AdmissionStop("usage_unknown")
        self._pending.pop(ticket)
        self.used += input_tokens + output_tokens
        if self.used + sum(self._pending.values()) > self.limit:
            self.unknown = True
            raise AdmissionStop("budget_overrun")

    def uncertain(self, ticket: int) -> None:
        if ticket not in self._pending:
            raise ValueError("unknown budget ticket")
        self.unknown = True

    def snapshot(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "known_used_tokens": self.used,
            "outstanding_reserved_tokens": sum(self._pending.values()),
            "usage_complete": not self.unknown and not self._pending,
            "within_observed_limit": self.used <= self.limit if not self.unknown and not self._pending else None,
            "strict_budget_valid": False,
            "admission_method": "utf8_json_size_estimate_plus_max_output",
        }
