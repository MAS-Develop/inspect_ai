"""Offline by default. Real model calls require --live and an explicit model."""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    """Run a single exposed development task and print an admission summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["hr", "finance"], default="hr")
    parser.add_argument("--task-id", type=int, default=5072)
    parser.add_argument("--architecture", choices=["sas", "wf-s", "wf-p"], default="sas")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--budget", type=int, default=1_000_000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-calls", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--log-dir", default="./logs/automationbench-admission")
    args = parser.parse_args()
    if args.model and not args.live:
        parser.error("--model requires --live; offline mode never uses an external provider")
    if args.live and not args.model:
        parser.error("--live requires an explicit --model; no provider is silently selected")

    from inspect_ai import eval
    from inspect_ai.model import ModelOutput, ModelUsage, get_model

    from .task import RunConfig, automationbench_task

    os.environ["AUTOMATIONBENCH_STRICT_ASSERTIONS"] = "1"
    config = RunConfig(architecture=args.architecture, budget=args.budget,
                       max_tokens=args.max_tokens, max_calls=args.max_calls,
                       temperature=args.temperature, reasoning_effort=args.reasoning_effort)
    task = automationbench_task(args.domain, args.task_id, config)
    if args.live:
        model = get_model(args.model)
    else:
        def scripted(input, tools, tool_choice, config):
            output = ModelOutput.from_content("mockllm", "Offline wiring check; no task solution attempted.")
            output.usage = ModelUsage(input_tokens=10, output_tokens=10, total_tokens=20)
            return output
        model = get_model("mockllm/model", custom_outputs=scripted)
    logs = eval(task, model=model, log_dir=args.log_dir, display="none")
    log = logs[0]
    samples = log.samples or []
    record = samples[0].metadata.get("admission", {}) if samples else {}
    successful = log.status == "success" and len(samples) == 1 and not samples[0].error
    print(json.dumps({"mode": "live" if args.live else "scripted_offline",
                      "inspect_status": log.status, "wiring_completed": successful,
                      "native_task_success": record.get("native", {}).get("task_completed_correctly"),
                      "native_scoring_error": record.get("native", {}).get("scoring_error"),
                      "termination": record.get("termination"),
                      "budget": record.get("budget"), "log": log.location},
                     ensure_ascii=False, indent=2))
    if not successful or record.get("native", {}).get("scoring_error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
