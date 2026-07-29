from __future__ import annotations

import json

from .llm import LLMClient
from .models import CodeTaskSpec, EditPlan, RepoContext


def build_edit_plan(spec: CodeTaskSpec, context: RepoContext, client: LLMClient) -> EditPlan:
    system = (
        "You are the Architect for a lightweight coding agent. "
        "Return only JSON matching the requested schema. "
        "Do not propose silent changes to model architecture, loss functions, optimizers, "
        "dataset splits, or evaluation metrics. If such changes seem required, set feasibility "
        "to needs_context, blocked, or unsafe and explain in needs_user_input."
    )
    user = {
        "task_goal": spec.task_goal,
        "constraints": spec.constraints,
        "verify_commands": spec.verify_commands,
        "repo_tree": context.tree[:300],
        "snippets": [snippet.model_dump() for snippet in context.snippets],
        "schema": {
            "summary": "string",
            "target_files": ["relative/path.py"],
            "allowed_edit_type": "logging_only|config_only|bugfix|new_file|general",
            "risks": ["string"],
            "verification": ["string"],
            "needs_user_input": ["string"],
            "feasibility": "ready_to_edit|needs_context|blocked|unsafe",
        },
    }
    return EditPlan.model_validate(client.complete_json(system, json.dumps(user, indent=2)))
