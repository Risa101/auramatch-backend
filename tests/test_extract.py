"""Tests for services.color_engine.extract -- the robust skin-Lab aggregator.

Focuses on the two properties the implementation brief calls out explicitly:
(1) determinism (Phase 25/18 -- same input must give the same output, no
hidden randomness), and (2) that contamination (a cluster of very different-
colored pixels mixed into the skin region) gets down-weighted rather than
silently averaged in and skewing the result.
"""

import numpy as np

from services.color_engine.extract import extract_skin_features
from services.color_engine.skin_mask import CHEEK_LEFT, FOREHEAD, build_skin_mask

CANVAS = (120, 120)
NUM_LANDMARKS = 468


def _landmarks_with_cheek_and_forehead():
    lm = np.full((NUM_LANDMARKS, 2), 1.0)
    angles = np.linspace(0, 2 * np.pi, len(CHEEK_LEFT), endpoint=False)
    lm[CHEEK_LEFT] = np.stack([60 + 25 * np.cos(angles), 70 + 25 * np.sin(angles)], axis=1)
    angles_f = np.linspace(0, 2 * np.pi, len(FOREHEAD), endpoint=False)
    lm[FOREHEAD] = np.stack([60 + 15 * np.cos(angles_f), 30 + 15 * np.sin(angles_f)], axis=1)
    return lm


def _skin_toned_image():
    # A plausible mid-tone skin BGR fill (OpenCV channel order).
    img = np.full((*CANVAS, 3), (120, 150, 200), dtype=np.uint8)  # BGR -> warm-ish RGB
    return img


def test_extract_is_deterministic():
    lm = _landmarks_with_cheek_and_forehead()
    image = _skin_toned_image()
    mask = build_skin_mask(image, lm)

    a = extract_skin_features(image, mask)
    b = extract_skin_features(image, mask)
    assert a == b


def test_extract_valid_true_on_a_clean_uniform_region():
    lm = _landmarks_with_cheek_and_forehead()
    image = _skin_toned_image()
    mask = build_skin_mask(image, lm)

    feats = extract_skin_features(image, mask)
    assert feats["valid"] is True
    assert feats["skin_lab"] is not None
    assert len(feats["skin_lab"]) == 3
    assert feats["confidence"] in ("low", "medium")  # never "high" -- see module docstring
    assert feats["kept_weight_fraction"] == 1.0  # uniform region -> nothing to drop


def test_extract_invalid_when_too_few_skin_pixels():
    lm = np.full((NUM_LANDMARKS, 2), 1.0)  # degenerate landmarks -> ~empty mask
    image = _skin_toned_image()
    mask = build_skin_mask(image, lm)

    feats = extract_skin_features(image, mask, min_skin_pixels=100)
    assert feats["valid"] is False
    assert "too few skin pixels" in feats["reason"]


def test_contamination_is_downweighted_not_averaged_in():
    """Half the cheek region is genuine skin tone; the other half is
    contaminated with a very different color (e.g. stray hair/shadow that
    slipped past the mask boundary). The representative Lab should land much
    closer to the DOMINANT (larger) cluster than a naive mean would.
    """
    lm = _landmarks_with_cheek_and_forehead()
    image = _skin_toned_image()
    mask = build_skin_mask(image, lm)

    contaminated = image.copy()
    ys, xs = np.where(mask.cheeks)
    # Contaminate the left half of the cheek pixels with a very dark color.
    left_half = xs < np.median(xs)
    contaminated[ys[left_half], xs[left_half]] = (10, 10, 10)

    feats = extract_skin_features(contaminated, mask)
    assert feats["valid"] is True
    # A naive mean of ~50% skin (L~65) + ~50% near-black (L~4) would land
    # around L~35. The robust aggregator must land much lighter than that,
    # i.e. closer to the dominant (uncontaminated) cluster.
    assert feats["skin_lab"][0] > 45.0
