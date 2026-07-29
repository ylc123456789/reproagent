from __future__ import annotations

from .controller import run_step_controller
from .models import CodeTaskSpec, PatchReport


def run_code_task(spec: CodeTaskSpec) -> PatchReport:
    return run_step_controller(spec)
