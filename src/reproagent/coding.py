"""Compatibility shim — patch orchestration merged into integrations.codingagent."""

from .integrations.codingagent import run_coding_agent_for_patch

__all__ = ["run_coding_agent_for_patch"]
