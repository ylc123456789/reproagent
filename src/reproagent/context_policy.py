"""Resolve prompt context limits from model capabilities."""
from __future__ import annotations

from pydantic import BaseModel

_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "deepseek-v4-pro": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
}
_DEFAULT_WINDOW = 128_000


class ContextPolicy(BaseModel):
    """Limits for packing prompt context, scaled to the model window."""
    step_history: int = 8
    file_cache_count: int = 6
    file_cache_chars: int = 4000
    observation_tail: int = 500
    readme_chars: int = 16000
    tree_limit: int = 500

    @classmethod
    def for_model(cls, model: str | None) -> "ContextPolicy":
        """Resolve context policy limits for a given model by its context window size."""
        window = _MODEL_CONTEXT_WINDOWS.get(
            (model or "").lower().split("/")[-1], _DEFAULT_WINDOW,
        )
        if window >= 500_000:
            return cls(step_history=15, file_cache_count=10, file_cache_chars=8000,
                       observation_tail=1500, readme_chars=30000, tree_limit=800)
        if window >= 128_000:
            return cls(step_history=8, file_cache_count=6, file_cache_chars=4000,
                       observation_tail=500, readme_chars=16000, tree_limit=500)
        return cls(step_history=4, file_cache_count=3, file_cache_chars=2000,
                   observation_tail=300, readme_chars=8000, tree_limit=300)
