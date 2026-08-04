"""Convert edit and verification evidence into a final report."""
from __future__ import annotations

from .models import CodeTaskSpec, CommandResult, PatchReport


def review_outcome(
    spec: CodeTaskSpec,
    changed_files: list[str],
    diff_path,
    verification_results: list[CommandResult],
    residual_risks: list[str] | None = None,
) -> PatchReport:
    """Review changes and verification into a report."""
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
    failed_verifications = [r for r in verification_results if not r.succeeded]
    if failed_verifications:
        residual.append(
            f"{len(failed_verifications)} verification command(s) returned non-zero: "
            + "; ".join(f"{r.command} (rc={r.returncode})" for r in failed_verifications)
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
