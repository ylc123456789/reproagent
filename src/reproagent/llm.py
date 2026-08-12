"""OpenAI-compatible LLM API client — pure transport layer.

Prompt construction lives in prompts.py.  This module only handles
serialisation, HTTP calls, retries, and trace logging.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime

from .models import ReproTask


def call_llm(task: ReproTask, system: str, user: str, *, trace_label: str = "") -> str:
    """Call the LLM with a fresh system+user pair. No chat history.

    Retries transient network errors (timeout, connection reset, 5xx)
    up to 3 times with exponential back-off.
    """
    if task.mock_llm:
        return mock_response(user)
    return _openai_compatible(task, system, user, trace_label=trace_label)


def _openai_compatible(task: ReproTask, system: str, user: str, *,
                        trace_label: str = "", max_retries: int = 3) -> str:
    """Call an OpenAI-compatible chat completions API with retries on transient errors."""
    api_key = os.environ.get(task.api_key_env)
    if not api_key:
        raise RuntimeError(f"{task.api_key_env} is not set. Use --mock-llm for local testing.")
    model = task.model or "gpt-4.1-mini"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    url = _chat_completions_url(task.api_base)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()
            if trace_label:
                _write_llm_trace(task, trace_label, system, user, text)
            return text
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
            last_error = exc
            status = getattr(exc, "code", 0) if isinstance(exc, urllib.error.HTTPError) else 0
            if status and status < 500:
                raise
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError(
        f"LLM API call failed after {max_retries} retries: {last_error}"
    )


def _chat_completions_url(api_base: str) -> str:
    """Build the chat completions endpoint URL."""
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _write_llm_trace(task: ReproTask, trace_label: str, system: str, user: str, response: str) -> None:
    """Save LLM prompt and response to the workspace logs directory."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", trace_label).strip("_") or "llm"
    logs_dir = task.workspace_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    prefix = logs_dir / f"llm_{stamp}_{safe}"
    (prefix.with_suffix(".prompt.txt")).write_text(
        f"[system]\n{system}\n\n[user]\n{user}", encoding="utf-8")
    (prefix.with_suffix(".response.txt")).write_text(response, encoding="utf-8")


def mock_response(user: str) -> str:
    """Deterministic mock actions for tests — returns probe then finish."""
    if "Begin." in user or "What is your first action" in user:
        return '{"thinking": "mock: probe the repo", "action": "run_commands", "stage_hint": "probe", "commands": ["head -20 README.md"]}'
    if "Last Result" in user:
        return '{"thinking": "mock: done", "action": "finish", "finish_status": "completed", "finish_summary": "Mock run completed."}'
    return '{"thinking": "mock: done", "action": "finish", "finish_status": "completed", "finish_summary": "Mock run completed."}'
