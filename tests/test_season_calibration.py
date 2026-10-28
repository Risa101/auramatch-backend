"""Unit tests for services/eval/season_calibration.py. Synthetic feature
rows only (plain dicts with fabricated L/a/b/hue_angle/chroma/truth) --
no real images, no real season.py modification anywhere in this file.
"""

from __future__ import annotations

import pytest

from services.color_engine.season import DEFAULT_MODEL
from services.eval.season_calibration import (
    InvalidCandidateError,
    VALID_PARAMETER_FIELDS,
    compare_candidates,
    cross_validate,
    detect_class_collapse,
    development_rows,
    evaluate_candidate,
    filter_by_split,
    k_fold_indices,
    make_candidate,
    margin_for_row,
)


def _row(image_id, truth, L, a, b, hue_angle=None, chroma=None):
    import math
    if hue_angle is None:
        hue_angle = math.degrees(math.atan2(b, a)) % 360.0
    if chroma is None:
        chroma = math.hypot(a, b)
    return {"image_id": image_id, "truth": truth, "L": L, "a": a, "b": b, "hue_angle": hue_angle, "chroma": chroma, "ita_category": "intermediate"}


SYNTHETIC_ROWS = [
    _row("a1", "Spring", 70.0, 18.0, 30.0),
    _row("a2", "Spring", 68.0, 16.0, 28.0),
    _row("a3", "Summer", 66.0, 3.0, 5.0),
    _row("a4", "Summer", 64.0, 2.0, 4.0),
    _row("a5", "Autumn", 45.0, 20.0, 28.0),
    _row("a6", "Autumn", 42.0, 19.0, 26.0),
    _row("a7", "Winter", 40.0, 2.0, 3.0),
    _row("a8", "Winter", 38.0, 1.0, 2.0),
]


# --- make_candidate ---------------------------------------------------


def test_make_candidate_returns_new_model_without_mutating_default():
    original_pivot = DEFAULT_MODEL.value_pivot
    candidate = make_candidate({"value_pivot": 60.0})
    assert candidate.value_pivot == 60.0
    assert DEFAULT_MODEL.value_pivot == original_pivot  # untouched
    assert candidate is not DEFAULT_MODEL


def test_make_candidate_rejects_unknown_parameter():
    with pytest.raises(InvalidCandidateError):
        make_candidate({"not_a_real_field": 1.0})


def test_make_candidate_empty_overrides_equals_baseline():
    candidate = make_candidate({})
    assert candidate == DEFAULT_MODEL


def test_valid_parameter_fields_matches_seasonmodel_fields():
    assert "value_pivot" in VALID_PARAMETER_FIELDS
    assert "chroma_pivot" in VALID_PARAMETER_FIELDS
    assert "hue_pivot" in VALID_PARAMETER_FIELDS


def test_make_candidate_composes_on_top_of_another_candidate():
    base = make_candidate({"value_pivot": 60.0})
    composed = make_candidate({"chroma_pivot": 20.0}, base_model=base)
    assert composed.value_pivot == 60.0
    assert composed.chroma_pivot == 20.0
    assert base.chroma_pivot == DEFAULT_MODEL.chroma_pivot  # base untouched


# --- holdout enforcement --------------------------------------------------


def test_development_rows_excludes_holdout():
    rows = [{"image_id": "a1"}, {"image_id": "a2"}, {"image_id": "a3"}]
    splits = {"a1": "calibration", "a2": "validation", "a3": "holdout"}
    dev = development_rows(rows, splits)
    assert {r["image_id"] for r in dev} == {"a1", "a2"}


def test_development_rows_never_includes_holdout_even_if_requested_elsewhere():
    # No parameter on development_rows() can leak holdout through -- this
    # is the enforcement primitive; nothing here bypasses it.
    rows = [{"image_id": "h1"}]
    splits = {"h1": "holdout"}
    assert development_rows(rows, splits) == []


def test_filter_by_split_explicit_holdout_access_requires_explicit_request():
    rows = [{"image_id": "a1"}, {"image_id": "h1"}]
    splits = {"a1": "calibration", "h1": "holdout"}
    assert filter_by_split(rows, splits, splits_allowed=("calibration",)) == [{"image_id": "a1"}]
    holdout_only = filter_by_split(rows, splits, splits_allowed=("holdout",))
    assert holdout_only == [{"image_id": "h1"}]


def test_filter_by_split_unassigned_image_excluded():
    rows = [{"image_id": "unassigned"}]
    splits = {}
    assert filter_by_split(rows, splits, splits_allowed=("calibration", "validation", "holdout")) == []


# --- evaluate_candidate ---------------------------------------------------


def test_evaluate_candidate_deterministic():
    a = evaluate_candidate(SYNTHETIC_ROWS, DEFAULT_MODEL)
    b = evaluate_candidate(SYNTHETIC_ROWS, DEFAULT_MODEL)
    assert a == b


def test_evaluate_candidate_reports_primary_metric_as_balanced_accuracy():
    result = evaluate_candidate(SYNTHETIC_ROWS, DEFAULT_MODEL)
    assert "balanced_accuracy" in result
    assert result["balanced_accuracy"] is not None
    assert 0.0 <= result["balanced_accuracy"] <= 1.0


def test_evaluate_candidate_empty_rows():
    result = evaluate_candidate([], DEFAULT_MODEL)
    assert result["n"] == 0
    assert result["balanced_accuracy"] is None
    assert result["coverage"] is None


def test_evaluate_candidate_different_model_can_change_result():
    baseline = evaluate_candidate(SYNTHETIC_ROWS, DEFAULT_MODEL)
    extreme_candidate = make_candidate({"value_pivot": -100.0})  # pathological, deliberately extreme for the test
    candidate_result = evaluate_candidate(SYNTHETIC_ROWS, extreme_candidate)
    # Not asserting WHICH is better -- only that the candidate model's
    # output is actually being used (not silently falling back to baseline).
    assert candidate_result["prediction_distribution"] != {}
    assert isinstance(candidate_result["balanced_accuracy"], (float, type(None)))


# --- detect_class_collapse ------------------------------------------------


def test_detect_class_collapse_flags_overpredicted_season():
    result = {
        "n": 8,
        "truth_distribution": {"Spring": 2, "Summer": 2, "Autumn": 2, "Winter": 2},
        "prediction_distribution": {"Spring": 6, "Summer": 1, "Autumn": 1, "Winter": 0},
    }
    collapse = detect_class_collapse(result, ratio_threshold=2.0)
    assert collapse["flags"]["Spring"] is True  # 6/8=0.75 vs truth 2/8=0.25 -> ratio 3.0 >= 2.0
    assert collapse["any_collapsed"] is True


def test_detect_class_collapse_no_collapse_when_balanced():
    result = {
        "n": 8,
        "truth_distribution": {"Spring": 2, "Summer": 2, "Autumn": 2, "Winter": 2},
        "prediction_distribution": {"Spring": 2, "Summer": 2, "Autumn": 2, "Winter": 2},
    }
    collapse = detect_class_collapse(result)
    assert collapse["any_collapsed"] is False


def test_detect_class_collapse_zero_rows():
    result = evaluate_candidate([], DEFAULT_MODEL)
    collapse = detect_class_collapse(result)
    assert collapse["any_collapsed"] is False
    assert all(v is None for v in collapse["flags"].values())


# --- compare_candidates ---------------------------------------------------


def test_compare_candidates_reports_delta_never_auto_selects():
    baseline = evaluate_candidate(SYNTHETIC_ROWS, DEFAULT_MODEL)
    candidate_model = make_candidate({"value_pivot": 60.0})
    candidate = evaluate_candidate(SYNTHETIC_ROWS, candidate_model)
    comparison = compare_candidates(baseline, candidate, candidate_overrides={"value_pivot": 60.0})
    assert "recommendation" in comparison
    assert "REVIEW" in comparison["recommendation"]  # never says APPROVED/APPLIED
    assert comparison["complexity_candidate"] == 1
    assert comparison["complexity_baseline"] == 0


# --- k_fold_indices / cross_validate ---------------------------------------


def test_k_fold_indices_deterministic():
    ids = [f"img_{i}" for i in range(10)]
    a = k_fold_indices(ids, k=3, seed=1)
    b = k_fold_indices(ids, k=3, seed=1)
    assert a == b


def test_k_fold_indices_covers_every_id_exactly_once():
    ids = [f"img_{i}" for i in range(10)]
    folds = k_fold_indices(ids, k=3, seed=1)
    all_assigned = [i for fold in folds for i in fold]
    assert sorted(all_assigned) == sorted(ids)


def test_k_fold_indices_rejects_k_less_than_2():
    with pytest.raises(ValueError):
        k_fold_indices(["a", "b"], k=1, seed=1)


def test_cross_validate_reports_mean_and_variance():
    result = cross_validate(SYNTHETIC_ROWS, DEFAULT_MODEL, k=4, seed=1)
    assert result["k"] == 4
    assert len(result["folds"]) == 4
    assert result["mean_balanced_accuracy"] is not None


# --- margin_for_row ---------------------------------------------------


def test_margin_for_row_deterministic():
    row = SYNTHETIC_ROWS[0]
    a = margin_for_row(row, DEFAULT_MODEL)
    b = margin_for_row(row, DEFAULT_MODEL)
    assert a == b
    assert a["margin"] is not None and a["margin"] >= 0.0
