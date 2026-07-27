from reproagent.report import _clean_text


def test_clean_text_removes_common_mojibake():
    text = "Smoke Test 鈥?Spiral ODE and paper鈥檚 metrics"

    cleaned = _clean_text(text)

    assert "鈥" not in cleaned
    assert "paper's" in cleaned
