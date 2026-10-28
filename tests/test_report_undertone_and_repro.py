from services.eval.report import render_report


def test_undertone_section_no_data_yet_when_empty():
    report = render_report({})
    assert "## Undertone Results" in report
    assert "no data yet" in report


def test_undertone_section_renders_real_confusion_matrix_and_per_class():
    ctx = {
        "undertone_confusion_matrix": {
            "Warm": {"Warm": 5, "Cool": 1, "Neutral": 0, "NO_PREDICTION": 0},
            "Cool": {"Warm": 0, "Cool": 4, "Neutral": 0, "NO_PREDICTION": 0},
            "Neutral": {"Warm": 0, "Cool": 0, "Neutral": 2, "NO_PREDICTION": 0},
        },
        "undertone_per_class": [
            {"undertone": "Warm", "precision": 1.0, "recall": 0.833, "f1": 0.909, "support": 6},
        ],
    }
    report = render_report(ctx)
    assert "| Warm | Warm | Cool | Neutral | NO_PREDICTION |" not in report  # sanity: not a literal header typo
    assert "| Warm | 5 | 1 | 0 | 0 |" in report
    assert "0.909" in report


def test_reproducibility_section_no_data_yet_when_absent():
    report = render_report({})
    assert "## Reproducibility Information" in report
    assert report.count("no data yet") >= 1


def test_reproducibility_section_renders_provided_markdown():
    ctx = {"reproducibility_markdown": "* Run timestamp: 2026-08-30T00:00:00Z\n* Code commit: abc123"}
    report = render_report(ctx)
    assert "2026-08-30T00:00:00Z" in report
    assert "abc123" in report
