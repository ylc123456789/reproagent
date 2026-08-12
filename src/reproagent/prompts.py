"""Compatibility shim — prompts moved to controller.prompts."""

from .controller.prompts import SYSTEM_PROMPT, build_initial_context, build_turn_prompt

__all__ = ["SYSTEM_PROMPT", "build_initial_context", "build_turn_prompt"]
