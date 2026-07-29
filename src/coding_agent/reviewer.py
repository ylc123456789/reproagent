from __future__ import annotations

from .models import CodeTaskSpec, CommandResult, PatchReport


def review_outcome(
    spec: CodeTaskSpec,
    changed_files: list[str],
    diff_path,
    verification_results: list[CommandResult],
    residual_risks: list[str] | None = None,
) -> PatchReport:
    residual = list(residual_risks or [])
    if not changed_files:
        return PatchReport(
            status="failed",
            changed_files=[],
            diff_path=diff_path,
            verification_results=verification_results,
            summary="No files were changed.",
            residual_risks=residual,
        )
    if verification_results and not all(result.succeeded for result in verification_results):
        return PatchReport(
            status="failed",
            changed_files=changed_files,
            diff_path=diff_path,
            verification_results=verification_results,
            summary="Patch was applied, but one or more verification commands failed.",
            residual_risks=residual,
        )
    if not verification_results:
        residual.append("No verification commands were provided, so completion is based on patch application only.")
    return PatchReport(
        status="completed",
        changed_files=changed_files,
        diff_path=diff_path,
        verification_results=verification_results,
        summary=f"Applied a minimal patch for: {spec.task_goal}",
        residual_risks=residual,
    )
