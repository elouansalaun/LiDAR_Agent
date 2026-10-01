"""The LLM call, isolated in one place.

The graph only needs "given the conversation, return Claude's next message".
Keeping that behind a small class makes it easy to swap models, and to
replace Claude by a scripted fake in the tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY (and the optional settings below) from .env

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"  # tool orchestration does not need deep reasoning; raise to "high" if plans get sloppy

# Models that accept server-side refusal fallbacks with `fallbacks: "default"`.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}


def has_credentials() -> bool:
    """True if the SDK can find credentials: an API key / token in the environment or .env,
    or a profile created with `ant auth login`."""
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").is_dir()


class ClaudeLLM:
    def __init__(self, model: str | None = None, effort: str | None = None, max_tokens: int = 16000):
        self.model = model or os.getenv("LIDAR_AGENT_MODEL", DEFAULT_MODEL)
        self.effort = effort or os.getenv("LIDAR_AGENT_EFFORT", DEFAULT_EFFORT)
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()  # credentials from ANTHROPIC_API_KEY or `ant auth login`

    def __call__(self, system: str, messages: list, tools: list[dict]):
        """Send the conversation to Claude and return the response message."""
        options = {}
        if self.model in FALLBACK_MODELS:
            # If a safety classifier declines a request, the API retries it on a
            # suitable fallback model instead of returning a refusal.
            options = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},  # caches the system prompt, tools and history between steps
            **options,
        )


def make_llm(offline: bool = False):
    """Claude when credentials are available, otherwise the offline demo planner.

    offline=True forces the offline planner even when a key exists (handy for demos).
    """
    if offline or not has_credentials():
        from src.agent.offline import OfflinePlanner

        return OfflinePlanner()
    return ClaudeLLM()
