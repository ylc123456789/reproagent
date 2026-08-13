"""Prompt context budget policies, plus legacy re-exports.

clone_repo/collect_context lived in the flat context.py module before the
readability refactor moved them to repository.context.  The same-named
package below shadows the old module, so the legacy public path
``from reproagent.context import collect_context`` must be re-exported
here to stay compatible.
"""

from ..repository.context import clone_repo, collect_context, setup_workspace

__all__ = ["clone_repo", "collect_context", "setup_workspace"]
