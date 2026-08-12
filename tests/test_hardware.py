from reproagent.runtime.hardware import collect_hardware_text


def test_hardware_text_contains_policy():
    text = collect_hardware_text(timeout=5)
    assert "Policy:" in text
    assert "CPU" in text
