"""Tests for services.color_engine.white_balance.

Verifies the Shades-of-Gray estimator actually neutralizes a synthetic color
cast, and that white_balance() round-trips a neutral scene unchanged (no
"aggressive color correction that changes the person's true skin
characteristics" -- see the implementation-phase brief, Phase 7).
"""

import numpy as np
import pytest

from services.color_engine import white_balance as wb


def test_estimate_illuminant_unit_norm():
    rng = np.random.default_rng(0)
    img = rng.uniform(0.1, 0.9, size=(20, 20, 3))
    illum = wb.estimate_illuminant(img)
    assert illum.shape == (3,)
    assert np.linalg.norm(illum) == pytest.approx(1.0, abs=1e-9)


def test_gray_world_recovers_neutral_illuminant():
    """A perfectly neutral (gray) scene's illuminant estimate must itself be
    neutral -- i.e. all three channels equal.
    """
    gray = np.full((30, 30, 3), 0.5)
    illum = wb.estimate_illuminant(gray, p=1)
    assert illum[0] == pytest.approx(illum[1], abs=1e-9)
    assert illum[1] == pytest.approx(illum[2], abs=1e-9)


def test_white_balance_neutralizes_a_synthetic_color_cast():
    """A neutral gray scene rendered under a strong warm (reddish) cast
    should come back close to neutral (a* ~ b* ~ 0 in Lab) after correction.
    """
    from services.color_engine.color_math import srgb_to_lab

    h, w = 40, 40
    neutral = np.full((h, w, 3), 128, dtype=np.uint8)
    cast = neutral.astype(float).copy()
    cast[..., 0] *= 1.35  # boost R -> warm cast (image arrays are RGB here)
    cast[..., 2] *= 0.75  # cut B
    cast = np.clip(cast, 0, 255).astype(np.uint8)

    corrected = wb.white_balance(cast, p=6)
    lab = srgb_to_lab(corrected.astype(float) / 255.0)
    mean_a, mean_b = lab[..., 1].mean(), lab[..., 2].mean()

    uncorrected_lab = srgb_to_lab(cast.astype(float) / 255.0)
    uncorrected_b = uncorrected_lab[..., 2].mean()

    assert abs(mean_a) < 3.0
    assert abs(mean_b) < 3.0
    assert abs(mean_b) < abs(uncorrected_b)


def test_white_balance_is_identity_on_already_neutral_scene():
    neutral = np.full((20, 20, 3), 150, dtype=np.uint8)
    corrected = wb.white_balance(neutral, p=6)
    assert np.allclose(corrected.astype(int), neutral.astype(int), atol=1)


def test_white_balance_rejects_degenerate_black_region():
    black = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        wb.white_balance(black)


def test_correct_neutralizes_a_pixel_matching_the_illuminant():
    """The intended invariant (per the module docstring): a pixel whose color
    IS the estimated illuminant (e.g. a gray card lit by that illuminant)
    must be corrected to a true neutral gray at the illuminant's mean
    brightness -- that is the entire point of white balance.
    """
    illum = np.array([0.6, 0.5, 0.4])
    out = wb.correct(illum, illum)
    assert out[0] == pytest.approx(out[1], abs=1e-9)
    assert out[1] == pytest.approx(out[2], abs=1e-9)
    assert out.mean() == pytest.approx(illum.mean(), abs=1e-9)
