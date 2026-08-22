"""Public import stability: the legacy context path must keep working.

Interface stability only — the re-exports point at the real
repository.context implementations; the old linear workflow is gone and
nothing here brings it back.
"""

import reproagent.context


def test_context_legacy_exports_are_the_real_implementations():
    """from reproagent.context import ... must resolve to repository.context."""
    import reproagent.repository.context as real
    from reproagent.context import clone_repo, collect_context, setup_workspace

    assert clone_repo is real.clone_repo
    assert collect_context is real.collect_context
    assert setup_workspace is real.setup_workspace
    assert reproagent.context.__all__ == ["clone_repo", "collect_context", "setup_workspace"]


def test_context_policy_still_importable():
    """The package's own module stays available alongside the re-exports."""
    from reproagent.context.policy import ContextPolicy

    # gpt-4o sits in the 128k-window tier
    assert ContextPolicy.for_model("gpt-4o").step_history == 8
