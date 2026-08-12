"""Compatibility shim — command execution moved to runtime.runner."""

from .runtime.runner import is_safe_command, run_commands

__all__ = ["is_safe_command", "run_commands"]
