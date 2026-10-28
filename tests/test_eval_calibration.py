from services.eval.calibration import calibration_buckets, calibration_gap_summary


def test_calibration_buckets_groups_correctly():
    rows = [
        {"confidence": 0.95, "truth": "Spring", "predicted": "Spring"},
        {"confidence": 0.92, "truth": "Spring", "predicted": "Autumn"},
        {"confidence": 0.75, "truth": "Winter", "predicted": "Winter"},
    ]
    buckets = calibration_buckets(rows)
    top = next(b for b in buckets if b["range"] == "0.90-1.01")
    assert top["n"] == 2
    assert top["empirical_accuracy"] == 0.5  # one right, one wrong, both "90%+ confident"


def test_calibration_buckets_excludes_missing_confidence():
    rows = [{"confidence": None, "truth": "Spring", "predicted": "Spring"}]
    buckets = calibration_buckets(rows)
    assert all(b["n"] == 0 for b in buckets)


def test_calibration_gap_summary_insufficient_data():
    rows = [{"confidence": 0.9, "truth": "Spring", "predicted": "Spring"}]  # only 1 sample
    summary = calibration_gap_summary(calibration_buckets(rows))
    assert "Insufficient data" in summary


def test_calibration_gap_summary_flags_a_real_gap():
    # 10 samples all reporting 0.95 confidence, but only half are correct ->
    # a 45-point gap between reported and empirical accuracy.
    rows = [
        {"confidence": 0.95, "truth": "Spring", "predicted": "Spring" if i % 2 == 0 else "Winter"}
        for i in range(10)
    ]
    summary = calibration_gap_summary(calibration_buckets(rows))
    assert "poorly calibrated" in summary
