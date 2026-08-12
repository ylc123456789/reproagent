"""Compatibility shim — repository context moved to repository.context."""

from .repository.context import clone_repo, collect_context

__all__ = ["clone_repo", "collect_context"]
