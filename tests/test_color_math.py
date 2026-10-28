"""Tests for services.color_engine.color_math.

CIEDE2000 vectors are the canonical 34-pair reference table from Sharma, Wu &
Dalal (2005), Table 1 -- any correct implementation must reproduce these to
4 decimal places. (Same table used to verify the original implementation in
auramatchgenz/backend/tests/test_color_math.py; reused here since this
module is a verbatim port.)
"""

import numpy as np
import pytest

from services.color_engine import color_math as cm

SHARMA_2005 = [
    (50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425),
    (50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615),
    (50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412),
    (50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669),
    (50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461),
    (50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065),
    (50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492),
    (50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977),
    (50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030),
    (50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000),
    (60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644),
    (63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630),
    (61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731),
    (35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645),
    (22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373),
    (36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146),
    (90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441),
    (90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381),
    (6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377),
    (2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082),
]


@pytest.mark.parametrize("row", SHARMA_2005)
def test_ciede2000_sharma_2005(row):
    L1, a1, b1, L2, a2, b2, expected = row
    got = cm.ciede2000([L1, a1, b1], [L2, a2, b2])
    assert got == pytest.approx(expected, abs=1e-4)


def test_ciede2000_is_symmetric():
    a, b = [55.0, 10.0, 20.0], [60.0, -5.0, 15.0]
    assert cm.ciede2000(a, b) == pytest.approx(cm.ciede2000(b, a), abs=1e-9)


def test_ciede2000_identity_is_zero():
    lab = [40.0, 12.0, 18.0]
    assert cm.ciede2000(lab, lab) == pytest.approx(0.0, abs=1e-9)


def test_srgb_white_is_d65():
    """Pure white sRGB [1,1,1] must map to L*=100, a*=b*=0 under D65."""
    lab = cm.srgb_to_lab(np.array([1.0, 1.0, 1.0]))
    assert lab[0] == pytest.approx(100.0, abs=1e-3)
    assert lab[1] == pytest.approx(0.0, abs=1e-3)
    assert lab[2] == pytest.approx(0.0, abs=1e-3)


def test_srgb_black_is_zero_lightness():
    lab = cm.srgb_to_lab(np.array([0.0, 0.0, 0.0]))
    assert lab[0] == pytest.approx(0.0, abs=1e-6)


def test_lab_roundtrip():
    rgb = np.array([0.6, 0.35, 0.28])  # a plausible mid-tone skin-ish sRGB
    lab = cm.srgb_to_lab(rgb)
    back = cm.lab_to_srgb(lab)
    assert back == pytest.approx(rgb, abs=1e-4)


def test_ita_category_boundaries():
    assert cm.ita_category(56) == "very_light"
    assert cm.ita_category(55) == "light"
    assert cm.ita_category(42) == "light"
    assert cm.ita_category(41) == "intermediate"
    assert cm.ita_category(29) == "intermediate"
    assert cm.ita_category(28) == "tan"
    assert cm.ita_category(11) == "tan"
    assert cm.ita_category(10) == "brown"
    assert cm.ita_category(-29) == "brown"
    assert cm.ita_category(-30) == "dark"


def test_hue_angle_wraps_to_0_360():
    lab = np.array([50.0, -10.0, -10.0])  # third quadrant -> angle > 180
    h = cm.hue_angle(lab)
    assert 0.0 <= h < 360.0
