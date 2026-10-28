"""Tests for services.color_engine.skin_mask.

Uses synthetic landmark coordinates on a small canvas rather than a real
face photo -- this module only cares about geometry (convex hulls + carve-
outs + luminance percentiles), which is fully exercisable without MediaPipe.
"""

import numpy as np

from services.color_engine.skin_mask import (
    CHEEK_LEFT, CHEEK_RIGHT, FOREHEAD, _LEFT_EYE, build_skin_mask,
)

CANVAS = (100, 100)  # (H, W)
NUM_LANDMARKS = 468


def _base_landmarks():
    """All 468 points parked at a far corner; only specific groups get
    meaningful coordinates per test.
    """
    return np.full((NUM_LANDMARKS, 2), 1.0)


def _hull_around(center, radius, n):
    """n points on a small circle -- a valid, fillable convex polygon."""
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)


def test_cheeks_and_forehead_regions_are_filled():
    lm = _base_landmarks()
    lm[CHEEK_LEFT] = _hull_around((20, 60), 8, len(CHEEK_LEFT))
    lm[CHEEK_RIGHT] = _hull_around((80, 60), 8, len(CHEEK_RIGHT))
    lm[FOREHEAD] = _hull_around((50, 20), 10, len(FOREHEAD))

    image = np.full((*CANVAS, 3), 150, dtype=np.uint8)  # flat mid-gray, BGR
    result = build_skin_mask(image, lm)

    assert result.num_pixels > 0
    assert result.cheeks.sum() > 0
    assert result.forehead.sum() > 0
    # cheeks and forehead are spatially disjoint by construction here
    assert not np.any(result.cheeks & result.forehead)
    # weight map matches region membership
    assert np.all(result.weights[result.cheeks] == 1.0)
    assert np.all(result.weights[result.forehead] == 0.5)
    assert np.all(result.weights[~result.mask] == 0.0)


def test_eye_landmarks_carve_out_pixels_from_an_overlapping_cheek():
    lm = _base_landmarks()
    lm[CHEEK_LEFT] = _hull_around((30, 50), 15, len(CHEEK_LEFT))  # a big cheek hull
    lm[FOREHEAD] = _hull_around((50, 20), 10, len(FOREHEAD))
    # Put a "left eye" squarely inside the cheek hull above.
    lm[_LEFT_EYE] = _hull_around((30, 50), 5, len(_LEFT_EYE))

    image = np.full((*CANVAS, 3), 150, dtype=np.uint8)
    # Build a comparison mask with the eye moved far away (no carve-out).
    lm_no_eye = lm.copy()
    lm_no_eye[_LEFT_EYE] = 1.0
    result_no_eye = build_skin_mask(image, lm_no_eye, exclude_dilate_px=0)
    result_with_eye = build_skin_mask(image, lm, exclude_dilate_px=0)

    # The eye-carved mask must have strictly fewer skin pixels than the one
    # without any eye landmark overlapping the cheek.
    assert result_with_eye.mask.sum() < result_no_eye.mask.sum()


def test_bright_specular_pixels_are_rejected():
    lm = _base_landmarks()
    lm[CHEEK_LEFT] = _hull_around((50, 50), 20, len(CHEEK_LEFT))
    lm[FOREHEAD] = 1.0  # keep forehead empty/degenerate for this test

    image = np.full((*CANVAS, 3), 140, dtype=np.uint8)
    # Blast a small circular "specular highlight" (near-white) inside the cheek.
    yy, xx = np.ogrid[:CANVAS[0], :CANVAS[1]]
    highlight = (xx - 50) ** 2 + (yy - 50) ** 2 <= 4**2
    image[highlight] = 255

    result = build_skin_mask(image, lm, dark_percentile=5.0, bright_percentile=90.0)
    # The highlight center must NOT survive into the final mask.
    assert result.mask[50, 50] == False  # noqa: E712 (explicit bool compare reads clearer here)


def test_too_few_landmarks_raises_or_is_handled_by_caller():
    """skin_mask itself trusts its caller re: landmark count -- the input
    contract (>= 468 points) is enforced one layer up in
    services/personal_color_service.py, not here. Document that boundary
    with an explicit index-error check instead of silently mis-measuring.
    """
    lm = np.full((10, 2), 1.0)
    image = np.full((*CANVAS, 3), 150, dtype=np.uint8)
    try:
        build_skin_mask(image, lm)
        assert False, "expected an IndexError from out-of-range landmark groups"
    except IndexError:
        pass
