from reproagent.report import _clean_text


def test_clean_text_removes_common_mojibake():
    text = "Smoke Test 鈥?Spiral ODE and paper鈥檚 metrics"

    cleaned = _clean_text(text)

    assert "鈥" not in cleaned
    assert "paper's" in cleaned


def test_clean_text_repairs_medium_run_mojibake_samples():
    text = "GPU鈥慳ccelerated ODE鈥慛et time鈥憇eries full鈥憇cale torch鈮?.3.0"

    cleaned = _clean_text(text)

    assert "鈥" not in cleaned
    assert "GPU-accelerated" in cleaned
    assert "ODE-Net" in cleaned
    assert "time-series" in cleaned
    assert "full-scale" in cleaned
    assert "torch>=.3.0" in cleaned


def test_clean_text_repairs_latin1_utf8_mojibake():
    text = "paperâ€™s GPUâ€“accelerated run"

    cleaned = _clean_text(text)

    assert cleaned == "paper’s GPU–accelerated run"
