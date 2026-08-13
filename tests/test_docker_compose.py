"""Step 17 — Docker Compose packaging. Structural checks on docker-compose.yml itself
(via `docker compose config`, which validates syntax + resolves env var substitution
without building/starting anything) — always run, no gate needed, since they're fast
and don't touch Postgres/Qdrant. The actual build-and-run verification (both images
build, all 4 services start healthy, a real ingest+query+MCP round trip works through
the compose network) was done manually against a live Docker daemon — see BUILD_LOG's
Step 17 entry for what was checked and why that isn't captured as an automated test
here (a full image build is multiple minutes and not something every CI run should
pay for on every commit; Step 18 revisits test-suite composition/tiers more broadly).
"""

import json
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="docker not available"
)


def _compose_config() -> dict:
    result = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        cwd=__file__.rsplit("/tests/", 1)[0],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_four_services_defined():
    config = _compose_config()
    assert set(config["services"].keys()) == {"postgres", "qdrant", "app", "mcp"}


def test_three_persistent_volumes_defined():
    config = _compose_config()
    assert set(config["volumes"].keys()) == {"postgres_data", "qdrant_data", "dead_letter_data"}


def test_app_and_mcp_share_the_compose_network():
    config = _compose_config()
    for service in ("postgres", "qdrant", "app", "mcp"):
        assert "llm_wiki_net" in config["services"][service]["networks"]


def test_app_waits_for_postgres_and_qdrant_healthy():
    config = _compose_config()
    depends_on = config["services"]["app"]["depends_on"]
    assert depends_on["postgres"]["condition"] == "service_healthy"
    assert depends_on["qdrant"]["condition"] == "service_healthy"


def test_mcp_never_mounts_postgres_or_qdrant_volumes():
    """Structural backing for "zero direct connections" at the compose level too —
    the MCP service should have no volume mounts at all (it's stateless, Section 4)."""
    config = _compose_config()
    assert config["services"]["mcp"].get("volumes", []) == []


def test_two_separate_dockerfiles_referenced():
    config = _compose_config()
    assert config["services"]["app"]["build"]["dockerfile"] == "Dockerfile"
    assert config["services"]["mcp"]["build"]["dockerfile"] == "Dockerfile.mcp"
