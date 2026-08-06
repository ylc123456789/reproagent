"""Call OpenAI-compatible chat completion APIs."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMClient:
    """Small client for OpenAI-compatible chat completions."""
    api_base: str
    api_key_env: str
    model: str

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Request a JSON response and parse it."""
        text = self.complete(system, user, response_format={"type": "json_object"})
        return json.loads(text)

    def complete(self, system: str, user: str, response_format: dict[str, str] | None = None) -> str:
        """Request a chat completion with retries on transient errors.

        Retries up to 3 times on network errors (URLError, TimeoutError,
        OSError) and server errors (HTTP 5xx).  4xx errors raise immediately.
        Back-off: 2s, 4s, 8s (capped at 30s).
        """
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key environment variable: {self.api_key_env}")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    f"{self.api_base.rstrip('/')}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(min(2 ** attempt * 2, 30))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code >= 500 and attempt < 2:
                    time.sleep(min(2 ** attempt * 2, 30))
                    continue
                body = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"LLM request failed: {exc.code} {body}") from exc
        raise RuntimeError(
            f"LLM API call failed after 3 retries: {last_error}"
        )