"""Choose context packing limits from model capabilities."""
from __future__ import annotations

from dataclasses import dataclass

from .models import CodeTaskSpec


DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000
MODEL_CONTEXT_WINDOWS = {
    "deepseek-v4-pro": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
}


@dataclass(frozen=True)
class ContextPolicy:
    """Resolved limits for packing prompt context."""
    context_window_tokens: int
    input_budget_tokens: int
    repo_tree_limit: int
    snippet_count: int
    snippet_chars: int
    diff_chars: int
    recent_file_count: int
    recent_file_chars: int
    step_observation_chars: int
    read_file_chars: int


def resolve_context_policy(spec: CodeTaskSpec) -> ContextPolicy:
    """Resolve prompt context limits for a task spec."""
    window = spec.model_context_window_tokens or _model_context_window(spec.model)
    reserved = max(spec.context_output_reserve_tokens, int(window * spec.context_margin_ratio))
    input_budget = max(8_000, window - reserved)
    if spec.max_context_tokens is not None:
        input_budget = min(input_budget, spec.max_context_tokens)
    scale = _scale(input_budget)
    return ContextPolicy(
        context_window_tokens=window,
        input_budget_tokens=input_budget,
        repo_tree_limit=min(2_000, max(300, 300 + scale * 250)),
        snippet_count=min(64, max(8, 8 + scale * 8)),
        snippet_chars=min(64_000, max(12_000, 12_000 + scale * 8_000)),
        diff_chars=min(96_000, max(8_000, 8_000 + scale * 12_000)),
        recent_file_count=min(6, max(2, 2 + scale)),
        recent_file_chars=min(96_000, max(24_000, 24_000 + scale * 12_000)),
        step_observation_chars=min(12_000, max(2_000, 2_000 + scale * 1_500)),
        read_file_chars=min(180_000, max(30_000, 30_000 + scale * 25_000)),
    )


def _model_context_window(model: str) -> int:
    """Look up the context window for a model name."""
    normalized = model.lower().split("/")[-1]
    return MODEL_CONTEXT_WINDOWS.get(normalized, DEFAULT_CONTEXT_WINDOW_TOKENS)


def _scale(input_budget_tokens: int) -> int:
    """Map an input token budget to a small scaling tier."""
    if input_budget_tokens >= 700_000:
        return 5
    if input_budget_tokens >= 350_000:
        return 4
    if input_budget_tokens >= 180_000:
        return 3
    if input_budget_tokens >= 90_000:
        return 2
    if input_budget_tokens >= 45_000:
        return 1
    return 0
