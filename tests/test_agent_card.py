"""Capability card (agent.yaml) validation — scientific orchestration V2.

The card is reproagent's half of the capability contract: the ResAgent
CapabilityRegistry loads it and routes capabilities to this module from
it.  These tests lock the contract-relevant fields without adding a YAML
dependency: required keys, the exact frozen capability vocabulary, and
the evidence-not-conclusion output semantics.
"""

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CARD_PATH = _REPO_ROOT / "agent.yaml"


def test_card_exists_and_declares_required_fields():
    text = _CARD_PATH.read_text(encoding="utf-8")
    for field in (
        "name: reproagent",
        "role: experiment_operator",
        "side_effects: workspace_and_environment",
        "input_contract: ReproTask",
        "output_contract: AgentState/repro_result",
        "status: available",
    ):
        assert field in text, f"agent.yaml missing: {field}"


def test_card_declares_exactly_the_operator_capabilities():
    """Only the two experiment-execution capabilities; no duplicates of
    other modules' capabilities (modify_code / analyze_results / ...)."""
    text = _CARD_PATH.read_text(encoding="utf-8")
    declared = re.findall(r"^\s*-\s*([a-z_]+)\s*$", text, re.MULTILINE)
    assert sorted(declared) == ["execute_experiment", "reproduce_experiment"]


def test_card_marks_output_as_evidence_not_conclusion():
    """The operator produces evidence; final scientific conclusions are
    ExpAgent's responsibility."""
    text = _CARD_PATH.read_text(encoding="utf-8")
    assert "evidence" in text.lower()
    assert "never the operator" in text
