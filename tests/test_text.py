from reproagent.text import normalize_text


def test_normalize_text_uses_ascii_punctuation():
    text = "About 10–20 minutes; paper’s “GPU” run…"

    cleaned = normalize_text(text)

    assert cleaned == 'About 10-20 minutes; paper\'s "GPU" run...'


def test_normalize_text_repairs_common_windows_mojibake():
    text = "paperâ€™s GPUâ€“accelerated run"

    cleaned = normalize_text(text)

    assert cleaned == "paper's GPU-accelerated run"


def test_normalize_text_repairs_existing_report_mojibake_samples():
    text = "GPU鈥慳ccelerated ODE鈥慛et time鈥憇eries full鈥憇cale torch鈮?.3.0"

    cleaned = normalize_text(text)

    assert "鈥" not in cleaned
    assert "GPU-accelerated" in cleaned
    assert "ODE-Net" in cleaned
    assert "time-series" in cleaned
    assert "full-scale" in cleaned
    assert "torch>=.3.0" in cleaned
