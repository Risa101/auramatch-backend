"""Tests for services.eval.metrics -- pure arithmetic over synthetic label
rows. These do NOT stand in for a real accuracy measurement (no real photos
are involved); they only verify the metrics math itself is correct, so a
future real evaluation run can be trusted.
"""

from services.eval.metrics import (
    accuracy, balanced_accuracy, confusion_matrix, macro_f1,
    precision_recall_f1, rate,
)

CLASSES = ["Spring", "Summer", "Autumn", "Winter"]


def _rows(pairs):
    return [{"truth": t, "predicted": p} for t, p in pairs]


def test_accuracy_perfect():
    rows = _rows([("Spring", "Spring"), ("Winter", "Winter")])
    assert accuracy(rows) == 1.0


def test_accuracy_empty_is_none():
    assert accuracy([]) is None


def test_accuracy_partial():
    rows = _rows([("Spring", "Spring"), ("Winter", "Summer")])
    assert accuracy(rows) == 0.5


def test_balanced_accuracy_corrects_for_imbalance():
    # 9 Spring (all correct) + 1 Winter (wrong) -> raw accuracy is 90%, but
    # balanced accuracy averages per-class recall (Spring=1.0, Winter=0.0) = 0.5.
    rows = _rows([("Spring", "Spring")] * 9 + [("Winter", "Spring")])
    assert accuracy(rows) == 0.9
    assert balanced_accuracy(rows, CLASSES) == 0.5


def test_precision_recall_f1_basic():
    rows = _rows([
        ("Spring", "Spring"), ("Spring", "Autumn"),
        ("Autumn", "Spring"), ("Autumn", "Autumn"),
    ])
    p, r, f1, support = precision_recall_f1(rows, "Spring")
    assert p == 0.5  # 1 true positive, 1 false positive
    assert r == 0.5  # 1 true positive, 1 false negative
    assert f1 == 0.5
    assert support == 2


def test_precision_is_none_when_no_predictions_for_class():
    rows = _rows([("Spring", "Autumn")])
    p, r, f1, support = precision_recall_f1(rows, "Winter")
    assert p is None  # no predictions of Winter at all
    assert r is None  # (support=0 -> tp+fn=0)
    assert f1 is None


def test_confusion_matrix_includes_no_prediction_bucket():
    rows = [
        {"truth": "Spring", "predicted": "Spring"},
        {"truth": "Spring", "predicted": None},
    ]
    m = confusion_matrix(rows, CLASSES)
    assert m["Spring"]["Spring"] == 1
    assert m["Spring"]["NO_PREDICTION"] == 1


def test_macro_f1_averages_over_classes_with_support():
    rows = _rows([("Spring", "Spring"), ("Summer", "Summer")])
    # No Autumn/Winter rows at all -> those classes contribute nothing (no
    # support), not a spurious 0 that would drag the average down unfairly.
    assert macro_f1(rows, CLASSES) == 1.0


def test_rate_helper():
    rows = [{"reliability": "low"}, {"reliability": "high"}, {"reliability": "low"}]
    assert rate(rows, lambda r: r["reliability"] == "low") == 2 / 3
    assert rate([], lambda r: True) is None
