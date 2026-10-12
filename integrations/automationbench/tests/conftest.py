"""No credentials, network requests, or live providers in admission tests."""

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def strict_assertions(monkeypatch):
    monkeypatch.setenv("AUTOMATIONBENCH_STRICT_ASSERTIONS", "1")
