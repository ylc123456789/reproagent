"""Compatibility locks for pre-refactor module paths.

The readability refactor moved modules into packages.  The old top-level
paths must keep working for external callers (ResAgent adapters, older
scripts).  Each test asserts identity: the old path must resolve to the
SAME object as the new path, not a second implementation.

Regression case: a same-named package shadows a shim module (the context/
package shadowed context.py), so the import fails even though the file
exists — updating tests to new paths alone does not catch this.
"""


def test_legacy_context_path():
    from reproagent.context import clone_repo, collect_context
    from reproagent.repository.context import clone_repo as new_clone, collect_context as new_collect

    assert clone_repo is new_clone
    assert collect_context is new_collect


def test_legacy_runtime_paths():
    from reproagent.audit import audit_environment
    from reproagent.dataset_cache import prepare_dataset_links, render_dataset_block, scan_dataset_roots
    from reproagent.env import build_backend_command, ensure_environment, find_conda
    from reproagent.hardware import collect_hardware_text
    from reproagent.runner import is_safe_command, run_commands
    from reproagent.runtime.audit import audit_environment as new_audit
    from reproagent.runtime.dataset_cache import (
        prepare_dataset_links as new_links,
        render_dataset_block as new_render,
        scan_dataset_roots as new_scan,
    )
    from reproagent.runtime.environment import (
        build_backend_command as new_build,
        ensure_environment as new_ensure,
        find_conda as new_conda,
    )
    from reproagent.runtime.hardware import collect_hardware_text as new_hardware
    from reproagent.runtime.runner import is_safe_command as new_safe, run_commands as new_run

    assert audit_environment is new_audit
    assert (scan_dataset_roots, prepare_dataset_links, render_dataset_block) == (new_scan, new_links, new_render)
    assert (ensure_environment, build_backend_command, find_conda) == (new_ensure, new_build, new_conda)
    assert collect_hardware_text is new_hardware
    assert (run_commands, is_safe_command) == (new_run, new_safe)


def test_legacy_policy_and_prompts_paths():
    from reproagent.context_policy import ContextPolicy
    from reproagent.context.policy import ContextPolicy as new_policy
    from reproagent.prompts import SYSTEM_PROMPT, build_initial_context, build_turn_prompt
    from reproagent.controller.prompts import (
        SYSTEM_PROMPT as new_system,
        build_initial_context as new_initial,
        build_turn_prompt as new_turn,
    )

    assert ContextPolicy is new_policy
    assert (SYSTEM_PROMPT, build_initial_context, build_turn_prompt) == (new_system, new_initial, new_turn)


def test_legacy_coding_and_controller_paths():
    from reproagent.coding import run_coding_agent_for_patch
    from reproagent.controller import run_controller
    from reproagent.controller.loop import run_controller as new_controller
    from reproagent.integrations.codingagent import run_coding_agent_for_patch as new_patch

    assert run_coding_agent_for_patch is new_patch
    assert run_controller is new_controller
