"""Provide the top-level entrypoint for running a coding task."""
from __future__ import annotations

from .controller import run_step_controller
from .models import CodeTaskSpec, PatchReport


def run_code_task(spec: CodeTaskSpec) -> PatchReport:
    """Run a coding task through the step controller."""
    return run_step_controller(spec)
