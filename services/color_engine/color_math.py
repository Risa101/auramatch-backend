"""Core colorimetry for AuraMatch.

Ported verbatim from auramatchgenz/backend/app/color_engine/color_math.py
(consolidated in here so the production analysis pipeline has no runtime
dependency on a second, separately-hosted service -- see
services/color_engine_service.py's deprecation note for why that mattered).

Pure-math color primitives, no image/IO dependencies.

Conventions:
    * sRGB input, D65 white point, CIELAB (CIE 1976 L*a*b*) output.
    * sRGB component values are floats in [0, 1]. Use ``srgb_to_lab(rgb / 255)``
      for 8-bit images.
    * Every function accepts either a single ``(3,)`` triplet or an array of
      shape ``(..., 3)`` and operates vectorized over the leading axes.

CIEDE2000 follows Sharma, Wu & Dalal (2005), "The CIEDE2000 Color-Difference
Formula: Implementation Notes, Supplementary Test Data, and Mathematical
Observations" -- verified against their 34-pair reference table in
auramatchgenz/backend/tests/test_color_math.py; see tests/test_color_math.py
in this project for the same reference check ported alongside this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Constants ---------------------------------------------------------------

# sRGB (linear) -> CIE XYZ, D65, 2° observer. IEC 61966-2-1 matrix.
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)

# D65 reference white in the same XYZ scale as above (Y normalized to 1.0).
_D65 = np.array([0.95047, 1.0, 1.08883])

# Bradford cone-response domain (XYZ -> LMS) for chromatic adaptation.
_XYZ_TO_LMS_BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ]
)
_LMS_TO_XYZ_BRADFORD = np.linalg.inv(_XYZ_TO_LMS_BRADFORD)

# CIE standard constants: epsilon = (6/29)^3, kappa = (29/3)^3.
_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0

_POW7_25 = 25.0**7


# --- sRGB <-> linear ---------------------------------------------------------


def srgb_to_linear(rgb):
    """sRGB gamma -> linear-light RGB. Input/output in [0, 1]."""
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    """Linear-light RGB -> sRGB gamma. Input/output in [0, 1]."""
    rgb = np.asarray(rgb, dtype=float)
    rgb = np.clip(rgb, 0.0, None)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1.0 / 2.4) - 0.055)


# --- sRGB <-> XYZ ------------------------------------------------------------


def srgb_to_xyz(rgb):
    """sRGB (gamma, [0,1]) -> CIE XYZ (D65, Y in [0,1])."""
    lin = srgb_to_linear(rgb)
    return lin @ _RGB_TO_XYZ.T


def xyz_to_srgb(xyz):
    """CIE XYZ (D65) -> sRGB (gamma, [0,1]). Not clipped to gamut."""
    xyz = np.asarray(xyz, dtype=float)
    lin = xyz @ _XYZ_TO_RGB.T
    return linear_to_srgb(lin)


# --- Chromatic adaptation (Bradford von Kries) -------------------------------


def chromatic_adaptation_matrix(source_white_rgb, target_white_rgb=None, *, normalize=True):
    """Bradford chromatic-adaptation transform as a linear-RGB -> linear-RGB 3x3.

    Adapts colors seen under ``source_white_rgb`` to ``target_white_rgb``
    (default equal-energy = D65 chromaticity) via von Kries scaling in
    Bradford LMS cone space -- adapts off-neutral colors (skin) the way human
    cones do, so warm/cool undertone stays accurate under a strong cast.
    """
    src = np.asarray(source_white_rgb, dtype=float)
    tgt = np.array([1.0, 1.0, 1.0]) if target_white_rgb is None else np.asarray(target_white_rgb, dtype=float)

    src_lms = _XYZ_TO_LMS_BRADFORD @ (_RGB_TO_XYZ @ src)
    tgt_lms = _XYZ_TO_LMS_BRADFORD @ (_RGB_TO_XYZ @ tgt)
    gain = np.diag(tgt_lms / src_lms)

    m = _XYZ_TO_RGB @ _LMS_TO_XYZ_BRADFORD @ gain @ _XYZ_TO_LMS_BRADFORD @ _RGB_TO_XYZ
    if normalize:
        m = float(src.mean()) * m
    return m


# --- XYZ <-> CIELAB ----------------------------------------------------------


def _lab_f(t):
    return np.where(t > _EPS, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)


def _lab_f_inv(ft):
    ft3 = ft**3
    return np.where(ft3 > _EPS, ft3, (116.0 * ft - 16.0) / _KAPPA)


def xyz_to_lab(xyz):
    """CIE XYZ (D65, Y in [0,1]) -> CIELAB."""
    xyz = np.asarray(xyz, dtype=float)
    xr = xyz[..., 0] / _D65[0]
    yr = xyz[..., 1] / _D65[1]
    zr = xyz[..., 2] / _D65[2]
    fx, fy, fz = _lab_f(xr), _lab_f(yr), _lab_f(zr)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab):
    """CIELAB -> CIE XYZ (D65, Y in [0,1])."""
    lab = np.asarray(lab, dtype=float)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    xr = _lab_f_inv(fx)
    yr = np.where(L > _KAPPA * _EPS, fy**3, L / _KAPPA)
    zr = _lab_f_inv(fz)
    return np.stack([xr * _D65[0], yr * _D65[1], zr * _D65[2]], axis=-1)


# --- sRGB <-> CIELAB ---------------------------------------------------------


def srgb_to_lab(rgb):
    """sRGB (gamma, [0,1]) -> CIELAB. The main entry point for the pipeline."""
    return xyz_to_lab(srgb_to_xyz(rgb))


def lab_to_srgb(lab):
    """CIELAB -> sRGB (gamma, [0,1]). Not clipped to gamut."""
    return xyz_to_srgb(lab_to_xyz(lab))


# --- Derived Lab features ----------------------------------------------------


def chroma(lab):
    """C*ab = sqrt(a*^2 + b*^2)."""
    lab = np.asarray(lab, dtype=float)
    return np.hypot(lab[..., 1], lab[..., 2])


def hue_angle(lab):
    """h_ab = atan2(b*, a*) in degrees, wrapped to [0, 360)."""
    lab = np.asarray(lab, dtype=float)
    return np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0


def ita(lab):
    """Individual Typology Angle = atan2(L* - 50, b*) in degrees.

    NOTE: ITA uses only L* and b*. Never decide cool/pink undertone from ITA
    alone -- combine with a*/hue.
    """
    lab = np.asarray(lab, dtype=float)
    return np.degrees(np.arctan2(lab[..., 0] - 50.0, lab[..., 2]))


# --- CIEDE2000 ---------------------------------------------------------------


def ciede2000(lab1, lab2, k_L=1.0, k_C=1.0, k_H=1.0):
    """CIEDE2000 color difference (Delta E 00) between two CIELAB colors.

    Accepts single triplets or broadcastable arrays of shape ``(..., 3)``.
    Implementation per Sharma, Wu & Dalal (2005).
    """
    lab1 = np.asarray(lab1, dtype=float)
    lab2 = np.asarray(lab2, dtype=float)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = (C1 + C2) / 2.0
    C_bar7 = C_bar**7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + _POW7_25)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    C1pC2p = C1p * C2p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where(C1pC2p == 0.0, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1pC2p) * np.sin(np.radians(dhp) / 2.0)

    Lp_bar = (L1 + L2) / 2.0
    Cp_bar = (C1p + C2p) / 2.0

    hp_sum = h1p + h2p
    hp_diff = np.abs(h1p - h2p)
    hp_bar = np.where(
        hp_diff <= 180.0,
        hp_sum / 2.0,
        np.where(hp_sum < 360.0, (hp_sum + 360.0) / 2.0, (hp_sum - 360.0) / 2.0),
    )
    hp_bar = np.where(C1pC2p == 0.0, hp_sum, hp_bar)

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )

    dtheta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    Cp_bar7 = Cp_bar**7
    R_C = 2.0 * np.sqrt(Cp_bar7 / (Cp_bar7 + _POW7_25))
    S_L = 1.0 + (0.015 * (Lp_bar - 50.0) ** 2) / np.sqrt(20.0 + (Lp_bar - 50.0) ** 2)
    S_C = 1.0 + 0.045 * Cp_bar
    S_H = 1.0 + 0.015 * Cp_bar * T
    R_T = -np.sin(np.radians(2.0 * dtheta)) * R_C

    term_L = dLp / (k_L * S_L)
    term_C = dCp / (k_C * S_C)
    term_H = dHp / (k_H * S_H)
    dE = np.sqrt(term_L**2 + term_C**2 + term_H**2 + R_T * term_C * term_H)

    return float(dE) if np.ndim(dE) == 0 else dE


#: CIEDE2000 under its CIE name. Alias of :func:`ciede2000`.
delta_e_2000 = ciede2000


# --- Skin-tone features ------------------------------------------------------


def ita_category(ita_value: float) -> str:
    """Classify an Individual Typology Angle into a skin-tone category.

    Bins (Chardon et al., 1991), the standard dermatological cut points:
    ``very_light`` > 55 >= ``light`` > 41 >= ``intermediate`` > 28 >= ``tan``
    > 10 >= ``brown`` > -30 >= ``dark``.
    """
    if ita_value > 55:
        return "very_light"
    if ita_value > 41:
        return "light"
    if ita_value > 28:
        return "intermediate"
    if ita_value > 10:
        return "tan"
    if ita_value > -30:
        return "brown"
    return "dark"


@dataclass
class SkinFeatures:
    """Derived colorimetric features for one representative Lab color."""

    L: float
    a: float
    b: float
    chroma: float
    hue_angle: float
    ita: float
    ita_category: str


def skin_features(lab) -> SkinFeatures:
    """L*, a*, b*, chroma, hue angle, ITA and ITA category for a Lab triplet."""
    lab = np.asarray(lab, dtype=float)
    ita_val = float(ita(lab))
    return SkinFeatures(
        L=float(lab[0]),
        a=float(lab[1]),
        b=float(lab[2]),
        chroma=float(chroma(lab)),
        hue_angle=float(hue_angle(lab)),
        ita=ita_val,
        ita_category=ita_category(ita_val),
    )
