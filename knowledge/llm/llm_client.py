#!/usr/bin/env python3
"""
LLM client for StellCoilBench Knowledge Base brief and propose endpoints.

Provides a unified interface for calling large language models to generate
research briefs and case proposal actions. Supports multiple providers:

- **OpenAI**: GPT-4, GPT-4o-mini, etc. (OPENAI_API_KEY)
- **Anthropic**: Claude models (ANTHROPIC_API_KEY)
- **OpenAI-compatible**: Local models via Ollama, vLLM, etc. (KB_LLM_BASE_URL)

Configuration
-------------
Environment variables:
- KB_LLM_PROVIDER: "openai" | "anthropic" | "openai_compatible" (default: openai)
- KB_LLM_MODEL: Model name (default: gpt-4o-mini)
- KB_LLM_API_KEY: Fallback API key
- KB_LLM_BASE_URL: Base URL for OpenAI-compatible APIs (e.g. http://localhost:11434/v1)
- OPENAI_API_KEY, ANTHROPIC_API_KEY: Provider-specific keys
"""
from __future__ import annotations

import json
import os
from typing import Any


def _get_provider() -> str:
    """Return the configured LLM provider (openai, anthropic, or openai_compatible)."""
    return os.environ.get("KB_LLM_PROVIDER", "openai").lower()


def _get_model() -> str:
    """Return the configured model name."""
    return os.environ.get("KB_LLM_MODEL", "gpt-4o-mini")


def _get_api_key() -> str | None:
    """Return the API key for the current provider, or None if not set."""
    provider = _get_provider()
    if provider == "openai" or provider == "openai_compatible":
        return os.environ.get("OPENAI_API_KEY") or os.environ.get("KB_LLM_API_KEY")
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KB_LLM_API_KEY")
    return os.environ.get("KB_LLM_API_KEY")


def _get_base_url() -> str | None:
    """Return base URL for OpenAI-compatible APIs (e.g. local Ollama, vLLM)."""
    return os.environ.get("KB_LLM_BASE_URL")


def is_available() -> bool:
    """Return True if an LLM is configured and can be called.

    Checks for API key (or base URL for openai_compatible provider).
    """
    if _get_api_key():
        return True
    if _get_provider() == "openai_compatible" and _get_base_url():
        return True
    return False


def complete(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    """Call the LLM and return the assistant reply content.

    Parameters
    ----------
    messages : list[dict]
        Chat messages in OpenAI format: [{"role": "user"|"system"|"assistant", "content": "..."}].
    max_tokens : int, optional
        Maximum tokens to generate (default 4096).
    temperature : float, optional
        Sampling temperature, 0–2 (default 0.3).

    Returns
    -------
    str
        The assistant's reply text.

    Raises
    ------
    RuntimeError
        If the provider is unknown, not configured, or the API call fails.
    """
    provider = _get_provider()
    model = _get_model()
    api_key = _get_api_key()
    base_url = _get_base_url()

    if provider == "anthropic":
        return _complete_anthropic(messages, model=model, api_key=api_key, max_tokens=max_tokens, temperature=temperature)
    if provider in ("openai", "openai_compatible"):
        return _complete_openai(
            messages,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    raise RuntimeError(f"Unknown KB_LLM_PROVIDER: {provider}. Use openai, anthropic, or openai_compatible.")


def _complete_openai(
    messages: list[dict[str, str]],
    *,
    model: str,
    api_key: str | None,
    base_url: str | None,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call OpenAI or OpenAI-compatible API (e.g. Ollama)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("Install openai: pip install openai")

    client_kw: dict[str, Any] = {}
    if api_key:
        client_kw["api_key"] = api_key
    if base_url:
        client_kw["base_url"] = base_url

    client = OpenAI(**client_kw)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not resp.choices:
        raise RuntimeError("LLM returned no choices")
    content = resp.choices[0].message.content
    return content or ""


def _complete_anthropic(
    messages: list[dict[str, str]],
    *,
    model: str,
    api_key: str | None,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call Anthropic Claude API. Converts OpenAI-style messages to Anthropic format."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise RuntimeError("Install anthropic: pip install anthropic")

    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY or KB_LLM_API_KEY required")

    client = Anthropic(api_key=api_key)

    # Anthropic uses system + messages; first user message can be system
    system = ""
    anthropic_messages: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system = content
        else:
            anthropic_messages.append({"role": role, "content": content})

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": anthropic_messages,
    }
    if system:
        kwargs["system"] = system

    resp = client.messages.create(**kwargs)
    if not resp.content:
        return ""
    part = resp.content[0]
    if hasattr(part, "text"):
        return part.text
    return str(part)


def complete_json(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 4096,
) -> dict[str, Any] | list[Any]:
    """Call the LLM and parse the response as JSON.

    Automatically strips markdown code fences (```json ... ```) if present.
    Uses lower temperature (0.2) for more deterministic JSON output.

    Parameters
    ----------
    messages : list[dict]
        Chat messages in OpenAI format.
    max_tokens : int, optional
        Maximum tokens (default 4096).

    Returns
    -------
    dict | list
        Parsed JSON (dict or list).

    Raises
    ------
    json.JSONDecodeError
        If the response is not valid JSON.
    """
    content = complete(messages, max_tokens=max_tokens, temperature=0.2)
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "```":
                end = i
                break
        content = "\n".join(lines[start:end])
    return json.loads(content)
