"""Collect integration-only evidence. No installs, live providers, or pushes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INTEGRATION = ROOT / "integrations/automationbench"


def source_hashes() -> dict[str, str]:
    paths = [INTEGRATION / "pyproject.toml", ROOT / ".github/workflows/automationbench-admission.yml"]
    for folder in ("src", "tests", "scripts"):
        paths.extend((INTEGRATION / folder).rglob("*.py"))
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "admission-results" / stamp).absolute()
    output.mkdir(parents=True, exist_ok=False)
    home = output / "clean-home"
    home.mkdir()
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(home), "LANG": "C", "PYTHONUTF8": "1", "PYTHONNOUSERSITE": "1",
        "AUTOMATIONBENCH_STRICT_ASSERTIONS": "1",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    }
    # Preserve interpreter loader paths, but no provider credentials or proxies.
    for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    python = str(Path(sys.executable).absolute())
    guard = "import pytest_socket; pytest_socket.disable_socket(allow_unix_socket=True); "
    checks = [
        ("inspect-commit", ["git", "rev-parse", "HEAD"]),
        ("python", [python, "--version"]),
        ("installed-packages", [python, "-m", "pip", "freeze"]),
        ("pip-check", [python, "-m", "pip", "check"]),
        ("native-binding", [python, "-c", guard +
         "import json; from inspect_automationbench.native import verify_native; "
         "print(json.dumps(verify_native(), indent=2))"]),
        ("ruff", [python, "-m", "ruff", "check", str(INTEGRATION)]),
    ]
    for run in (1, 2):
        checks.append((f"pytest-{run}", [
            python, "-c", guard + "import pytest, sys; raise SystemExit(pytest.main(sys.argv[1:]))",
            "-c", str(INTEGRATION / "pyproject.toml"), str(INTEGRATION / "tests"),
            "--disable-socket", "--allow-unix-socket",
            f"--basetemp={output / f'temp-{run}'}", f"--junitxml={output / f'pytest-{run}.xml'}", "-v",
        ]))
    before = source_hashes()
    (output / "source-before.json").write_text(json.dumps(before, indent=2) + "\n")
    results = []
    for name, command in checks:
        started = datetime.now(timezone.utc).isoformat()
        try:
            with (output / f"{name}.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=600, check=False)
            code = process.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            code = 124 if isinstance(exc, subprocess.TimeoutExpired) else 127
            with (output / f"{name}.log").open("a", encoding="utf-8") as log:
                log.write(f"\n{type(exc).__name__}: {exc}\n")
        results.append({"check": name, "command": command, "started_utc": started,
                        "finished_utc": datetime.now(timezone.utc).isoformat(), "exit_code": code})
        (output / "commands.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{name}: exit {code}", flush=True)
        text = (output / f"{name}.log").read_text(encoding="utf-8", errors="replace")
        print(text[-20000:] if code else text[-1200:], flush=True)
    after = source_hashes()
    (output / "source-after.json").write_text(json.dumps(after, indent=2) + "\n")
    passed = all(result["exit_code"] == 0 for result in results) and before == after
    summary = {"passed": passed, "source_unchanged": before == after,
               "scope": "integration-only; scripted models; no live calls", "checks": results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Passed: {passed}; source unchanged: {before == after}")
    print(f"Evidence: {output}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
