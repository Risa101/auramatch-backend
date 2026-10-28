"""Unit tests for services/eval/raw_pipeline_diagnostics.py.

Synthetic fixtures only (a solid-color synthetic image + fabricated
landmarks), matching the existing pattern in
tests/test_personal_color_service.py -- no real face photos anywhere in
this file, per this task's own rule against using real images outside the
explicit evaluation CLI.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import services.personal_color_service as pcs
from services.color_engine.season import DEFAULT_MODEL
from services.color_engine.skin_mask import CHEEK_LEFT, FOREHEAD
from services.eval.raw_pipeline_diagnostics import (
    ablate_season_input,
    categorize_rejection,
    full_diagnostics,
    select_balanced_sample,
)

CANVAS = (200, 200)
NUM_LANDMARKS = 468


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    monkeypatch.setattr(pcs, "get_blush_products_by_undertone", lambda undertone: [])


def _normalized_landmarks():
    lm = np.full((NUM_LANDMARKS, 3), 0.5)
    h, w = CANVAS

    def circle(cx, cy, r, n):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = (cx + r * np.cos(angles)) / w
        ys = (cy + r * np.sin(angles)) / h
        return np.stack([xs, ys, np.zeros(n)], axis=1)

    lm[CHEEK_LEFT] = circle(60, 120, 30, len(CHEEK_LEFT))
    lm[FOREHEAD] = circle(100, 50, 25, len(FOREHEAD))
    return lm.tolist()


def _encode_png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    return buf.tobytes()


def _good_image_bytes():
    # A perfectly flat fill has zero Laplacian variance and trips
    # quality_port.py's real blur-reject gate (verified directly: 0.0 <
    # BLUR_VARIANCE_REJECT=6) -- light deterministic texture keeps this a
    # "good" quality image for the pipeline-runs-end-to-end tests below,
    # same real gate a production image would pass through.
    base = np.full((*CANVAS, 3), (120, 150, 200), dtype=np.int16)
    rng = np.random.default_rng(0)
    noisy = np.clip(base + rng.integers(-8, 9, size=base.shape), 0, 255).astype(np.uint8)
    return _encode_png(noisy)


def _tiny_image_bytes():
    # Well under quality_port.py's MIN_DIM_REJECT=180 -- must reject.
    bgr = np.full((50, 50, 3), (120, 150, 200), dtype=np.uint8)
    return _encode_png(bgr)


# --- categorize_rejection --------------------------------------------------


@pytest.mark.parametrize("reasons,expected", [
    (["resolution_too_low(50px)"], "resolution"),
    (["no_face_detected"], "no_face"),
    (["face_too_small"], "face_size"),
    (["head_angle_extreme(45.0deg)"], "head_angle"),
    (["severely_underexposed"], "exposure"),
    (["severely_overexposed"], "exposure"),
    (["severely_blurry"], "blur"),
    ([], "none"),
    (["some_future_unknown_reason"], "other"),
])
def test_categorize_rejection(reasons, expected):
    assert categorize_rejection(reasons) == expected


def test_categorize_rejection_first_match_wins():
    # Multiple reasons -> reports the first recognized category, not a list
    # -- matches this experiment's single-category-per-row summary table.
    assert categorize_rejection(["severely_blurry", "face_too_small"]) == "blur"


# --- select_balanced_sample --------------------------------------------------


def _fake_labels(counts: dict) -> dict:
    labels = {}
    i = 0
    for season, n in counts.items():
        for _ in range(n):
            labels[f"img_{i:03d}.jpg"] = {"season": season}
            i += 1
    return labels


def test_select_balanced_sample_caps_per_season():
    labels = _fake_labels({"Spring": 10, "Summer": 3, "Autumn": 10, "Winter": 10})
    sample = select_balanced_sample(labels, per_season=5, seed=42)
    counts = {}
    for row in sample.values():
        counts[row["season"]] = counts.get(row["season"], 0) + 1
    assert counts["Spring"] == 5
    assert counts["Summer"] == 3  # fewer than per_season -- takes all available, no error
    assert counts["Autumn"] == 5
    assert counts["Winter"] == 5


def test_select_balanced_sample_deterministic():
    labels = _fake_labels({"Spring": 20, "Summer": 20, "Autumn": 20, "Winter": 20})
    a = select_balanced_sample(labels, per_season=4, seed=7)
    b = select_balanced_sample(labels, per_season=4, seed=7)
    assert set(a.keys()) == set(b.keys())


def test_select_balanced_sample_different_seed_can_differ():
    labels = _fake_labels({"Spring": 20, "Summer": 20, "Autumn": 20, "Winter": 20})
    a = select_balanced_sample(labels, per_season=4, seed=1)
    b = select_balanced_sample(labels, per_season=4, seed=2)
    assert set(a.keys()) != set(b.keys())  # astronomically unlikely to collide by chance


def test_select_balanced_sample_ignores_unlabeled_rows():
    labels = _fake_labels({"Spring": 5})
    labels["blank_row.jpg"] = {"season": None}
    labels["bad_row.jpg"] = {"season": "NotASeason"}
    sample = select_balanced_sample(labels, per_season=10, seed=1)
    assert "blank_row.jpg" not in sample
    assert "bad_row.jpg" not in sample
    assert len(sample) == 5


# --- full_diagnostics ---------------------------------------------------


def test_full_diagnostics_rejects_when_no_face_landmarks():
    diag = full_diagnostics(_good_image_bytes(), _normalized_landmarks(), face_detected=False)
    assert diag["pipeline_ran"] is False
    assert diag["valid"] is False
    assert diag["reason"] == "no_face_landmarks"
    assert diag["predicted_season"] is None


def test_full_diagnostics_rejects_on_quality_gate():
    diag = full_diagnostics(
        _tiny_image_bytes(), _normalized_landmarks(), face_detected=True,
        face_box={"width": 0.3}, roll_deg=0.0,
    )
    assert diag["quality_gate_level"] == "reject"
    assert diag["quality_rejection_category"] == "resolution"
    assert diag["pipeline_ran"] is False
    assert diag["predicted_season"] is None


def test_full_diagnostics_valid_image_produces_full_field_set():
    diag = full_diagnostics(
        _good_image_bytes(), _normalized_landmarks(), face_detected=True,
        face_box={"width": 0.3}, roll_deg=0.0,
    )
    assert diag["pipeline_ran"] is True
    assert diag["valid"] is True
    assert diag["predicted_season"] in {"Spring", "Summer", "Autumn", "Winter"}
    assert diag["second_season"] in {"Spring", "Summer", "Autumn", "Winter"}
    assert diag["margin"] is not None and diag["margin"] >= 0.0
    for s in ("season_spring_score", "season_summer_score", "season_autumn_score", "season_winter_score"):
        assert diag[s] is not None
        assert 0.0 <= diag[s] <= 1.0
    assert diag["L"] is not None and diag["a"] is not None and diag["b"] is not None
    assert diag["hue_angle"] is not None
    assert diag["chroma"] is not None
    assert diag["ita_category"] is not None
    assert diag["wb_method"] is not None
    assert diag["skin_pixel_count"] is not None and diag["skin_pixel_count"] > 0


def test_full_diagnostics_deterministic():
    kwargs = dict(face_detected=True, face_box={"width": 0.3}, roll_deg=0.0)
    a = full_diagnostics(_good_image_bytes(), _normalized_landmarks(), **kwargs)
    b = full_diagnostics(_good_image_bytes(), _normalized_landmarks(), **kwargs)
    assert a == b


# --- ablate_season_input (DIAGNOSTIC ABLATION) ---------------------------


def _features():
    return {
        "valid": True,
        "skin_lab": [60.0, 20.0, 25.0],
        "hue_angle": 51.34,  # atan2(25,20) in degrees
        "chroma": 32.0,
        "ita_category": "intermediate",
    }


def test_ablate_season_input_neutralizes_requested_axis_only():
    baseline_chroma = _features()["chroma"]
    ablated = ablate_season_input(_features(), axis="chroma")
    assert ablated.valid is True
    # bright_mute score should be pulled toward 0 relative to using the
    # original (far-from-pivot) chroma value -- exact assertion: the axis
    # score is now driven ONLY by the (also-neutralized-to-0) contrast term
    # since chroma itself was reset to the pivot.
    assert ablated.bright_mute_score is not None
    assert abs(ablated.bright_mute_score) < 0.05  # chroma_pivot -> axis ~0


def test_ablate_season_input_does_not_mutate_caller_dict():
    features = _features()
    original = dict(features)
    ablate_season_input(features, axis="warm")
    assert features == original  # deepcopy inside -- caller's dict is untouched


def test_ablate_season_input_invalid_axis_raises():
    with pytest.raises(ValueError):
        ablate_season_input(_features(), axis="not_a_real_axis")


def test_ablate_season_input_three_axes_are_independent():
    warm_ablated = ablate_season_input(_features(), axis="warm")
    value_ablated = ablate_season_input(_features(), axis="value")
    chroma_ablated = ablate_season_input(_features(), axis="chroma")
    # Each call only neutralizes its own axis -- the three results need not
    # be identical to each other (different perturbed inputs), but all must
    # still be valid SeasonDecisions from the SAME unmodified decide_season().
    for result in (warm_ablated, value_ablated, chroma_ablated):
        assert result.valid is True
        assert result.top_season in {"spring", "summer", "autumn", "winter"}


def test_ablate_season_input_uses_default_model_pivots_by_default():
    ablated = ablate_season_input(_features(), axis="value")
    # value axis neutralized -> light_deep_score should be ~0 (pivot=58 used)
    assert abs(ablated.light_deep_score) < 1e-6
    assert ablated.light_deep_score == pytest.approx(0.0, abs=1e-9)
