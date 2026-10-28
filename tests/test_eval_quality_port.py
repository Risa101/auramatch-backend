import cv2
import numpy as np

from services.eval.quality_port import assess_image_quality


def _textured_image(size=400):
    """A checkerboard has real high-frequency content -- a sharp image
    should NOT trip the blur heuristic.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    block = 10
    for y in range(0, size, block):
        for x in range(0, size, block):
            if ((x // block) + (y // block)) % 2 == 0:
                img[y:y + block, x:x + block] = 200
            else:
                img[y:y + block, x:x + block] = 60
    return img


def test_good_sharp_well_exposed_image_with_a_normal_face():
    img = _textured_image()
    result = assess_image_quality(img, face_width_fraction=0.4, roll_deg=2.0)
    assert result["level"] in ("good", "warning")  # texture-dependent, but never reject


def test_no_face_is_a_reject():
    img = _textured_image()
    result = assess_image_quality(img, face_width_fraction=None, roll_deg=None)
    assert result["level"] == "reject"
    assert "no_face_detected" in result["reasons"]


def test_severely_blurry_image_is_rejected():
    sharp = _textured_image()
    blurry = cv2.GaussianBlur(sharp, (31, 31), 15)
    result = assess_image_quality(blurry, face_width_fraction=0.4, roll_deg=0.0)
    assert result["level"] == "reject"
    assert any("blurry" in r for r in result["reasons"])


def test_severely_underexposed_image_is_rejected():
    dark = np.full((300, 300, 3), 5, dtype=np.uint8)
    result = assess_image_quality(dark, face_width_fraction=0.4, roll_deg=0.0)
    assert result["level"] == "reject"
    assert "severely_underexposed" in result["reasons"]


def test_extreme_head_roll_is_rejected():
    img = _textured_image()
    result = assess_image_quality(img, face_width_fraction=0.4, roll_deg=55.0)
    assert result["level"] == "reject"
    assert any("head_angle_extreme" in r for r in result["reasons"])


def test_moderate_tilt_is_only_a_warning():
    img = _textured_image()
    result = assess_image_quality(img, face_width_fraction=0.4, roll_deg=20.0)
    assert result["level"] == "warning"
    assert any("head_angle_tilted" in r for r in result["reasons"])
