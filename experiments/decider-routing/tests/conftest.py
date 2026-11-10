import pytest

HARBOR_ENV = (
    "STRANDS_MODEL",
    "HARBOR_TASK_WORKDIR",
    "HARBOR_REASONING_EFFORT",
    "HARBOR_MCP_SERVERS",
    "STRANDSHARNESS_AGENT_CONFIG",
    "STRANDSHARNESS_AGENT_CONFIG_B64",
    "STRANDSHARNESS_SEQUENTIAL_TOOLS",
    "ANTHROPIC_BASE_URL",
)


@pytest.fixture(autouse=True)
def isolated_trial(monkeypatch, tmp_path):
    for name in HARBOR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "session-token")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "aws-credentials"))
    monkeypatch.chdir(tmp_path)
