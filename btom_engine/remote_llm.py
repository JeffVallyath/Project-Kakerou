"""Remote LLM client — routes heavy reasoning to cloud APIs.

Supports Gemini (primary), OpenAI, and Anthropic.
Falls back to local LM Studio if no API key is configured.

Usage:
    from btom_engine.remote_llm import remote_chat
    result = remote_chat(
        system="You are an investigator.",
        user="Find info about Jeff Vallyath.",
        max_tokens=800,
    )
    # result = {"content": "...", "reasoning": "...", "model": "gemini-2.5-flash"}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from btom_engine.config import (
    LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS,
    REMOTE_LLM_PROVIDER, REMOTE_LLM_API_KEY,
    REMOTE_LLM_MODEL, REMOTE_LLM_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


def _has_remote() -> bool:
    """Check if a remote LLM is configured."""
    return bool(REMOTE_LLM_API_KEY and REMOTE_LLM_API_KEY.strip())


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _call_gemini(
    system: str,
    user: str,
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> dict[str, str]:
    """Call Gemini API via Google AI Studio REST endpoint."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{REMOTE_LLM_MODEL}:generateContent?key={REMOTE_LLM_API_KEY}"
    )

    # Build request body
    body: dict[str, Any] = {
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            # Disable thinking to avoid token budget waste on structured tasks
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    # System instruction (Gemini supports this natively)
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    resp = httpx.post(url, json=body, timeout=REMOTE_LLM_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()

    # Extract text from Gemini response (skip thinking parts)
    try:
        candidate = data["candidates"][0]
        parts = candidate["content"]["parts"]
        # Filter out thought parts — only keep actual output
        text = "".join(
            p.get("text", "") for p in parts
            if not p.get("thought", False)
        )
    except (KeyError, IndexError):
        text = ""

    return {"content": text, "model": REMOTE_LLM_MODEL, "provider": "gemini"}


# ---------------------------------------------------------------------------
# OpenAI (GPT-4o, GPT-4o-mini, etc.)
# ---------------------------------------------------------------------------

def _call_openai(
    system: str,
    user: str,
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> dict[str, str]:
    """Call OpenAI API."""
    url = "https://api.openai.com/v1/chat/completions"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    resp = httpx.post(
        url,
        headers={"Authorization": f"Bearer {REMOTE_LLM_API_KEY}"},
        json={
            "model": REMOTE_LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=REMOTE_LLM_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"] or ""

    return {"content": text, "model": REMOTE_LLM_MODEL, "provider": "openai"}


# ---------------------------------------------------------------------------
# Local fallback (LM Studio)
# ---------------------------------------------------------------------------

def _call_local(
    system: str,
    user: str,
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> dict[str, str]:
    """Fall back to local LM Studio endpoint."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    resp = httpx.post(
        f"{LLM_BASE_URL}/v1/chat/completions",
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=LLM_TIMEOUT_SECONDS + 30,
    )
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    text = msg.get("content", "") or ""

    # Qwen3 thinking model fallback
    if not text.strip() and msg.get("reasoning_content"):
        text = msg["reasoning_content"]

    return {"content": text, "model": LLM_MODEL, "provider": "local"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "gemini": _call_gemini,
    "openai": _call_openai,
}


def remote_chat(
    user: str,
    system: str = "",
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> dict[str, str]:
    """Send a message to the remote LLM (or local fallback).

    Returns dict with keys: content, model, provider.
    """
    if _has_remote() and REMOTE_LLM_PROVIDER in _PROVIDERS:
        try:
            result = _PROVIDERS[REMOTE_LLM_PROVIDER](
                system=system, user=user,
                max_tokens=max_tokens, temperature=temperature,
            )
            logger.info(
                "Remote LLM call (%s/%s): %d chars",
                result["provider"], result["model"], len(result["content"]),
            )
            return result
        except Exception as e:
            logger.warning("Remote LLM failed, falling back to local: %s", e)

    # Fallback to local
    return _call_local(
        system=system, user=user,
        max_tokens=max_tokens, temperature=temperature,
    )


def remote_chat_json(
    user: str,
    system: str = "",
    max_tokens: int = 800,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Remote chat that parses JSON from the response.

    Returns the parsed JSON dict, or empty dict on failure.
    """
    result = remote_chat(
        user=user, system=system,
        max_tokens=max_tokens, temperature=temperature,
    )
    raw = result["content"]

    # Strip markdown code blocks
    raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "")
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start:end + 1]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from remote LLM: %s", raw[:200])
        return {}
