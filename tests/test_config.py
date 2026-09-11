import pytest

from sift_agent.config import get_llm


def test_none_provider_returns_none():
    assert get_llm("none") is None


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        get_llm("telepathy")


def test_missing_required_env_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_llm("openai")


def test_env_overrides_beat_os_environ(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-environ")
    llm = get_llm("openai", env={"OPENAI_API_KEY": "sk-from-dict"})
    assert llm.openai_api_key.get_secret_value() == "sk-from-dict"


def test_env_overrides_do_not_mutate_os_environ(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_llm("openai", env={"OPENAI_API_KEY": "sk-from-dict"})
    import os

    assert "OPENAI_API_KEY" not in os.environ


def test_azure_builds_from_env_dict(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    llm = get_llm(
        "azure",
        env={
            "AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "k",
            "AZURE_OPENAI_DEPLOYMENT": "dep",
        },
    )
    assert llm.deployment_name == "dep"


def test_openai_compatible_requires_model():
    with pytest.raises(ValueError, match="requires --model"):
        get_llm("openai-compatible", env={"OPENAI_BASE_URL": "http://localhost:1"})


def test_bedrock_explicit_credentials():
    llm = get_llm(
        "bedrock",
        env={
            "AWS_ACCESS_KEY_ID": "a",
            "AWS_SECRET_ACCESS_KEY": "b",
            "AWS_REGION": "us-east-1",
        },
    )
    assert llm is not None
