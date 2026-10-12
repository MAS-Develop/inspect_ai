# Offline acceptance repair (2026-10-12)

Baseline: `552a0158fa1c0fd5f8a3a1f89498ca514ccb6ef2`.
The supplied independent Linux acceptance archive reported **36 passed, 4 failed**
on both runs, plus two Ruff I001 import errors. Those are integration test
failures, not model task scores. Its initial environment setup errors recovered.

## Changes and invariants

- Scripted valid `api_search` calls now supply `top_k=5`, as required by the
  pinned runtime schema. Invalid-response and batch-atomicity tests also use
  a schema-valid first call so they exercise the intended failure path.
- The analyst write-denial test supplies `method`, `url`, `params`, and `body`,
  first verifies schema acceptance with executor permissions, then verifies
  `permission_denied` and unchanged analyst/executor worlds.
- The paired initialization/replay test controls only the Sheets module's UUID
  source, separately reset for each setup. No task seed, assertion, scorer,
  production initializer, or world field is changed or discarded. Full world,
  initial state, task contract, tool result and native-score equality remain
  asserted. Separate unmocked tests verify native IDs and state isolation.
- Import formatting is fixed, not ignored. Missing runtime arguments still stop
  dispatch; neither schema validation nor analyst permissions are relaxed.
- The default CLI now makes one native discovery call per role before ending.
  `wiring_completed` requires normal completion and known, error-free scoring.
  Budget/call-limit stops or scorer errors return exit 1, even when Inspect
  itself successfully saved an evaluation log. Native task failure is still a
  valid *wiring* result; these scripts do not attempt to solve the tasks.
- Added regressions cover both HR5072 and Finance4062, all three supported
  architectures, early-stop/scorer CLI failures, both live opt-ins and blocked
  IPv4/IPv6 sockets. Existing admission contracts are retained.

## Reproduce

From this fork's repository root, using its Python 3.13 virtual environment:

```bash
# Installation is allowed to use the network. Acceptance itself does not install.
python -m pip install -e . -e './integrations/automationbench[native,test]'
python integrations/automationbench/scripts/offline_acceptance.py
```

The evidence runner uses the calling Python interpreter with a clean environment
and separate HOME (no inherited provider credentials). It verifies dependencies,
checks the frozen native sources, runs Ruff, then runs the entire integration
suite twice, blocking internet sockets before test collection and during tests.
It saves every command, exit code, log, JUnit report and before/after source hash,
continues collecting evidence after a failed check, and returns nonzero unless
all checks pass. The output directory must be new; it never deletes old results.
Inspect `.eval` logs, including the six domain/architecture CLI cases in each
pass, are retained under the two test temporary directories.

Offline means the Python socket boundary is blocked; this is not an OS-level
sandbox certification. Installation and GitHub source transfer are networked.
The workflow runs on GitHub-hosted Linux, not on the user's Mac. A green run does
not certify the full Inspect repository, live-provider compatibility, model
quality, or cross-platform behavior. Current input admission is still an
estimate: **`strict_budget_valid=false` remains unchanged**. No H9/401/three-vote
migration or paid model evaluation is included in this repair.
