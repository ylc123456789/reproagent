from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMClient:
    api_base: str
    api_key_env: str
    model: str

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        text = self.complete(system, user, response_format={"type": "json_object"})
        return json.loads(text)

    def complete(self, system: str, user: str, response_format: dict[str, str] | None = None) -> str:
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
        request = urllib.request.Request(
            f"{self.api_base.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM request failed: {exc.code} {body}") from exc
        return data["choices"][0]["message"]["content"]
