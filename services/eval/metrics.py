"""Classification metrics for the real-photo evaluation harness.

Pure functions over plain dicts (`{"truth": ..., "predicted": ...}` rows) --
no dependency on personal_color_service or any image I/O, so these are unit-
testable with synthetic label lists (see tests/test_eval_metrics.py) without
needing real photos. The photos are what's missing; the arithmetic isn't.
"""

from __future__ import annotations

from collections import defaultdict


def confusion_matrix(rows, classes) -> dict:
    """rows=[{"truth":..,"predicted":..}], classes=ordered label list.
    A `predicted` of None (analysis failed / no prediction) is bucketed under
    the synthetic class "NO_PREDICTION" -- it must be visible in the matrix,
    never silently dropped (a failure is a real outcome, not a missing row).
    """
    matrix = {t: defaultdict(int) for t in classes}
    for r in rows:
        truth = r["truth"]
        pred = r["predicted"] if r["predicted"] in classes else "NO_PREDICTION"
        if truth not in matrix:
            matrix[truth] = defaultdict(int)
        matrix[truth][pred] += 1
    return {t: dict(v) for t, v in matrix.items()}


def accuracy(rows) -> float | None:
    if not rows:
        return None
    correct = sum(1 for r in rows if r["predicted"] == r["truth"])
    return correct / len(rows)


def balanced_accuracy(rows, classes) -> float | None:
    """Mean of per-class recall -- robust to class imbalance (e.g. a dataset
    with far more Spring photos than Winter ones shouldn't let Spring's
    recall dominate the headline number).
    """
    recalls = []
    for c in classes:
        support = [r for r in rows if r["truth"] == c]
        if not support:
            continue
        recalls.append(sum(1 for r in support if r["predicted"] == c) / len(support))
    return sum(recalls) / len(recalls) if recalls else None


def precision_recall_f1(rows, cls) -> tuple:
    tp = sum(1 for r in rows if r["truth"] == cls and r["predicted"] == cls)
    fp = sum(1 for r in rows if r["truth"] != cls and r["predicted"] == cls)
    fn = sum(1 for r in rows if r["truth"] == cls and r["predicted"] != cls)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    support = tp + fn
    return precision, recall, f1, support


def macro_f1(rows, classes) -> float | None:
    f1s = [f1 for _, _, f1, support in (precision_recall_f1(rows, c) for c in classes) if f1 is not None and support > 0]
    return sum(f1s) / len(f1s) if f1s else None


def rate(rows, predicate) -> float | None:
    """Generic `fraction of rows matching predicate` helper -- used for
    failure rate, low-reliability rate, quality-rejection rate, etc.
    """
    if not rows:
        return None
    return sum(1 for r in rows if predicate(r)) / len(rows)
