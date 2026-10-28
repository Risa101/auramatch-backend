from services.eval.report import render_report


def test_empty_context_produces_honest_placeholders_not_fake_numbers():
    """Section 27's core requirement: no dataset -> no fabricated numbers."""
    report = render_report({})
    assert "no data yet" in report
    assert "# AuraMatch Real Photo Evaluation" in report
    # Must not contain a bare percentage-looking accuracy claim anywhere in
    # the overall metrics table when nothing was measured.
    assert "| Accuracy | no data yet |" in report


def test_populated_context_renders_real_numbers():
    ctx = {
        "dataset": {"n_images": 10, "n_labeled": 10, "n_skipped": 0, "per_season": {"Spring": 5, "Summer": 5, "Autumn": 0, "Winter": 0}},
        "overall_metrics": {"accuracy": 0.8, "balanced_accuracy": 0.75, "macro_f1": 0.7, "consistency": 0.9, "failure_rate": 0.0, "low_reliability_rate": 0.1},
        "confusion_matrix": {"Spring": {"Spring": 5, "Summer": 0, "Autumn": 0, "Winter": 0, "NO_PREDICTION": 0}},
        "per_season": [{"season": "Spring", "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 5}],
    }
    report = render_report(ctx)
    assert "80.0%" in report
    assert "| Spring | 1.000 | 1.000 | 1.000 | 5 |" in report


def test_report_never_claims_high_confidence_without_data():
    report = render_report({})
    assert "Recommended Calibration" in report
    assert "Not enough labeled data" in report
