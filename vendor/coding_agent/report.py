"""Write run artifacts such as state, diffs, and patch reports."""
from __future__ import annotations

from pathlib import Path


from .models import AgentState, CodeTaskSpec, PatchReport


def prepare_output_dir(spec: CodeTaskSpec) -> Path:
    """Create or reuse the run output directory."""
    output_dir = spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    return output_dir


def write_state(state: AgentState, output_dir: Path) -> None:
    """Write the serialized agent state."""
    (output_dir / "state.json").write_text(
        state.model_dump_json(indent=2),
        encoding="utf-8",
    )


def write_initial_diff(initial_diff: str, output_dir: Path) -> Path:
    """Write the initial repository diff."""
    path = output_dir / "initial_diff.patch"
    path.write_text(initial_diff, encoding="utf-8")
    return path


def write_diff(diff_text: str, output_dir: Path) -> Path:
    """Write the current repository diff."""
    path = output_dir / "diff.patch"
    path.write_text(diff_text, encoding="utf-8")
    return path


def write_patch_report(spec: CodeTaskSpec, report: PatchReport, output_dir: Path) -> Path:
    """Write a markdown patch report."""
    lines = [
        "# Coding Agent Patch Report",
        "",
        "## Task Goal",
        "",
        spec.task_goal,
        "",
        "## Status",
        "",
        report.status,
        "",
        "## Summary",
        "",
        report.summary,
        "",
        "## Changed Files",
        "",
    ]
    lines.extend(f"- `{path}`" for path in report.changed_files)
    if not report.changed_files:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Diff",
            "",
            f"`{report.diff_path}`" if report.diff_path else "None",
            "",
            "## Verification",
            "",
        ]
    )
    if report.verification_results:
        for result in report.verification_results:
            status = "passed" if result.succeeded else "failed"
            lines.append(
                f"- `{result.command}`: {status} "
                f"(returncode={result.returncode}, stdout=`{result.stdout_path}`, stderr=`{result.stderr_path}`)"
            )
    else:
        lines.append("- No verification commands were run.")
    lines.extend(["", "## Residual Risks", ""])
    lines.extend(f"- {risk}" for risk in report.residual_risks)
    if not report.residual_risks:
        lines.append("- None recorded.")
    path = output_dir / "patch_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
