from __future__ import annotations

import json

from .llm import LLMClient
from .models import CodeTaskSpec, EditPlan, RepoContext


def build_patch(spec: CodeTaskSpec, context: RepoContext, plan: EditPlan, client: LLMClient) -> str:
    system = (
        "You are the Editor for a lightweight coding agent. "
        "Return only a unified diff patch. Use relative file paths. "
        "Keep edits minimal and within the plan. Do not include markdown fences."
    )
    user = {
        "task_goal": spec.task_goal,
        "constraints": spec.constraints,
        "edit_plan": plan.model_dump(),
        "snippets": [snippet.model_dump() for snippet in context.snippets if snippet.path in plan.target_files],
        "fallback_snippets": [snippet.model_dump() for snippet in context.snippets[:8]],
    }
    return client.complete(system, json.dumps(user, indent=2)).strip()
