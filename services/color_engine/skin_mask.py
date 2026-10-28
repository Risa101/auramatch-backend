"""Landmark-restricted skin mask construction for AuraMatch.

Adapted from auramatchgenz/backend/app/color_engine/skin_mask.py. The one
change from the original: that version ran its own server-side MediaPipe
FaceMesh to get landmarks. Here, landmarks are computed ONCE in the browser
(src/utils/faceLandmarkerMP.js, already used by MakeoverStudio) and sent to
this backend alongside the image -- avoiding a second, heavier MediaPipe
Python dependency (which would also collide with this project's existing
opencv-python-headless install; see auramatchgenz/backend/requirements.txt's
own warning about two cv2 builds in one environment). `landmarks_px` below is
exactly what auramatchgenz's `FaceMeshResult.pixels` produced -- an (N, 2)
array of pixel-space (x, y) coordinates from the same 468/478-point MediaPipe
topology -- so the landmark index groups are unchanged from the original.

Turns face landmarks into a boolean mask of the pixels trusted as a
representative skin sample. Emphasizes the mid-cheek "apple" (the best
representative skin region per the Humanae analysis, arXiv:2505.14931) and
includes the forehead as a secondary region, excluding eyes, brows, lips,
nostrils and (via luminance percentiles) shadowed/specular pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color_math import srgb_to_lab

# --- Landmark index groups (MediaPipe canonical 468-pt topology) ------------

CHEEK_LEFT = [50, 101, 100, 118, 117, 119, 205, 206, 36, 142, 123]
CHEEK_RIGHT = [280, 330, 329, 347, 346, 348, 425, 426, 266, 371, 352]

FOREHEAD = [10, 151, 9, 67, 297, 69, 299, 109, 338, 108, 337, 104, 333]

_RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
_LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
_RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
_LEFT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
_NOSE_BASE = [2, 98, 97, 326, 327, 129, 358, 64, 294, 19]

_EXCLUSIONS = [_LEFT_EYE, _RIGHT_EYE, _LEFT_BROW, _RIGHT_BROW, _LIPS, _NOSE_BASE]


@dataclass
class SkinMask:
    """Result of :func:`build_skin_mask`.

    Attributes:
        mask: bool (H, W) -- the final set of pixels to sample.
        cheeks: bool (H, W) -- cheek-only subset of ``mask``.
        forehead: bool (H, W) -- forehead-only subset of ``mask``.
        weights: float (H, W) -- per-pixel sampling weight (0 outside ``mask``,
            ``cheek_weight`` on cheeks, ``forehead_weight`` on the forehead).
    """

    mask: np.ndarray
    cheeks: np.ndarray
    forehead: np.ndarray
    weights: np.ndarray

    @property
    def num_pixels(self) -> int:
        return int(self.mask.sum())


def _fill_hull(canvas: np.ndarray, points: np.ndarray) -> None:
    pts = np.rint(points).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillConvexPoly(canvas, cv2.convexHull(pts), 1)


def _region_mask(shape, landmarks_px: np.ndarray, index_groups) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    for group in index_groups:
        _fill_hull(m, landmarks_px[group])
    return m.astype(bool)


def _to_rgb01(image) -> np.ndarray:
    """BGR image (uint8 [0,255] or float [0,1]) -> RGB float in [0,1]."""
    img = np.asarray(image)
    if img.dtype == np.uint8:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    return np.ascontiguousarray(img[..., ::-1]).astype(float)


def build_skin_mask(
    image_bgr,
    landmarks_px: np.ndarray,
    *,
    dark_percentile: float = 5.0,
    bright_percentile: float = 95.0,
    exclude_dilate_px: int = 5,
    cheek_weight: float = 1.0,
    forehead_weight: float = 0.5,
) -> SkinMask:
    """Build a skin mask from pixel-space face landmarks.

    Args:
        image_bgr: (H, W, 3) BGR image (uint8 [0,255] or float [0,1]). Used for
            the shadow/specular luminance rejection.
        landmarks_px: (N, 2) float pixel coordinates (x, y), N >= 468, from the
            standard MediaPipe face-mesh topology.
        dark_percentile: reject skin pixels darker than this L* percentile
            (shadows), computed within the candidate region.
        bright_percentile: reject skin pixels brighter than this L* percentile
            (specular highlights).
        exclude_dilate_px: dilate the eye/brow/lip/nose exclusions by this many
            pixels to be safe near boundaries.
        cheek_weight, forehead_weight: per-region weights for the weight map.

    Returns:
        A :class:`SkinMask`.
    """
    h, w = np.asarray(image_bgr).shape[:2]
    shape = (h, w)
    landmarks_px = np.asarray(landmarks_px, dtype=float)

    cheeks = _region_mask(shape, landmarks_px, [CHEEK_LEFT, CHEEK_RIGHT])
    forehead = _region_mask(shape, landmarks_px, [FOREHEAD])

    excl = _region_mask(shape, landmarks_px, _EXCLUSIONS)
    if exclude_dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * exclude_dilate_px + 1, 2 * exclude_dilate_px + 1)
        )
        excl = cv2.dilate(excl.astype(np.uint8), k).astype(bool)
    keep_region = ~excl
    cheeks &= keep_region
    forehead &= keep_region
    region = cheeks | forehead

    L = srgb_to_lab(_to_rgb01(image_bgr))[..., 0]
    vals = L[region]
    if vals.size:
        lo = np.percentile(vals, dark_percentile)
        hi = np.percentile(vals, bright_percentile)
        lum_ok = (L >= lo) & (L <= hi)
        cheeks &= lum_ok
        forehead &= lum_ok

    mask = cheeks | forehead
    weights = np.zeros(shape, dtype=float)
    weights[forehead] = forehead_weight
    weights[cheeks] = cheek_weight  # cheeks win any overlap

    return SkinMask(mask=mask, cheeks=cheeks, forehead=forehead, weights=weights)
