"""Prompt context budget policies, plus public compatibility re-exports.

clone_repo / collect_context / setup_workspace are public API stability
exports from the flat ``context.py`` module era: they re-export the real
implementations in ``repository.context``. This is interface stability
only — it does not revive the old linear workflow, which no longer
exists anywhere in the production path.
"""

from ..repository.context import clone_repo, collect_context, setup_workspace

__all__ = ["clone_repo", "collect_context", "setup_workspace"]
