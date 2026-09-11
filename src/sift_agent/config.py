"""Multi-provider LLM factory.

See REQUIRED_ENV and `sift providers` for each provider's configuration.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from langchain_core.language_models import BaseChatModel

PROVIDERS = ("openai", "anthropic", "azure", "bedrock", "openai-compatible", "none")

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
    "azure": "gpt-4o",  # deployment name fallback
    "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "openai-compatible": "",
    "none": "",
}

REQUIRED_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, optional AZURE_OPENAI_API_VERSION; --model = deployment name",
    "bedrock": "AWS credentials via standard boto3 chain (AWS_PROFILE or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY), optional AWS_REGION",
    "openai-compatible": "OPENAI_BASE_URL, OPENAI_API_KEY ('EMPTY' for local servers); --model required",
    "none": "no LLM — deterministic heuristic insights",
}


def get_llm(
    provider: str,
    model: str | None = None,
    temperature: float = 0.2,
    env: Mapping[str, str] | None = None,
    timeout: float = 120.0,
) -> BaseChatModel | None:
    """Build a chat model for the given provider. Returns None for offline mode ('none').

    ``env`` entries override environment variables for this call only, so
    concurrent runs (e.g. two Streamlit sessions) can use different
    credentials without mutating the process-wide ``os.environ``.
    ``timeout`` bounds each request in seconds (retries are handled upstream).
    """
    provider = (provider or "none").strip().lower().replace("_", "-")
    overrides = {k: v for k, v in (env or {}).items() if v}

    def lookup(name: str) -> str | None:
        return overrides.get(name) or os.environ.get(name)

    def require(name: str) -> str:
        value = lookup(name)
        if not value:
            raise ValueError(
                f"provider '{provider}' requires the {name} environment variable to be set"
            )
        return value

    if provider == "none":
        return None

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or DEFAULT_MODELS["openai"],
            temperature=temperature,
            openai_api_key=require("OPENAI_API_KEY"),
            request_timeout=timeout,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model or DEFAULT_MODELS["anthropic"],
            temperature=temperature,
            max_tokens=4096,
            anthropic_api_key=require("ANTHROPIC_API_KEY"),
            default_request_timeout=timeout,
        )

    if provider == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=model
            or lookup("AZURE_OPENAI_DEPLOYMENT")
            or DEFAULT_MODELS["azure"],
            azure_endpoint=require("AZURE_OPENAI_ENDPOINT"),
            api_version=lookup("AZURE_OPENAI_API_VERSION") or "2024-10-21",
            temperature=temperature,
            openai_api_key=require("AZURE_OPENAI_API_KEY"),
            request_timeout=timeout,
        )

    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        kwargs: dict = {}
        region = lookup("AWS_REGION") or lookup("AWS_DEFAULT_REGION")
        if region:
            kwargs["region_name"] = region
        access_key = lookup("AWS_ACCESS_KEY_ID")
        secret_key = lookup("AWS_SECRET_ACCESS_KEY")
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
        kwargs["timeout"] = timeout
        return ChatBedrockConverse(
            model=model or DEFAULT_MODELS["bedrock"],
            temperature=temperature,
            **kwargs,
        )

    if provider == "openai-compatible":
        from langchain_openai import ChatOpenAI

        if not model:
            raise ValueError(
                "provider 'openai-compatible' requires --model (e.g. qwen2.5:14b, llama3.1:8b)"
            )
        return ChatOpenAI(
            model=model,
            base_url=require("OPENAI_BASE_URL"),
            openai_api_key=lookup("OPENAI_API_KEY") or "EMPTY",
            temperature=temperature,
            request_timeout=timeout,
        )

    raise ValueError(
        f"unknown provider '{provider}'. Supported: {', '.join(PROVIDERS)}"
    )
