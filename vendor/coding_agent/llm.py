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
        """Request a raw chat completion response with retry on transient errors."""
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

        last_error = None
        for attempt in range(4):
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
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                last_error = RuntimeError(f"LLM request failed: {exc.code} {body}")
                if exc.code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise last_error from exc
            except (OSError, TimeoutError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError("LLM request failed after retries") from exc
        raise last_error  # type: ignore[misc]