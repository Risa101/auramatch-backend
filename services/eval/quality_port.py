"""Python port of src/utils/imageQuality.js's quality heuristics, for the
batch (Python) evaluation harness -- the production quality gate runs in
the browser (canvas ImageData), so a batch script needs an equivalent that
works from a decoded image array instead.

IMPORTANT -- DUPLICATION RISK, STATED PLAINLY: this is a manually
synchronized copy of the JS thresholds/formulas, not a shared
implementation. If src/utils/imageQuality.js's constants change, this file
must be updated by hand or the evaluation report (audit real-photo brief,
section 25: "evaluate the quality gate on real images") will be silently
measuring a stale gate. There is no automated cross-language parity test
for this today -- a real follow-up (not done here) would be a small golden-
image fixture checked against both implementations.

Face-size/roll checks reuse whatever landmarks the caller already has (from
scripts/export_landmarks.py's faceBox/rollDeg fields) rather than
recomputing geometry here.
"""

from __future__ import annotations

import cv2
import numpy as np

# Mirrors the constants in src/utils/imageQuality.js exactly (see that file
# for the rationale/uncalibrated-heuristic caveat, which applies here too).
MIN_DIM_REJECT = 180
MIN_DIM_WARN = 420
FACE_FRAC_REJECT = 0.08
FACE_FRAC_WARN = 0.18
ROLL_REJECT_DEG = 40
ROLL_WARN_DEG = 15
LUMA_DARK_REJECT = 25
LUMA_DARK_WARN = 55
LUMA_BRIGHT_REJECT = 245
LUMA_BRIGHT_WARN = 220
CLIPPED_FRACTION_WARN = 0.35
BLUR_VARIANCE_REJECT = 6
BLUR_VARIANCE_WARN = 25
BLUR_SAMPLE_WIDTH = 200


def _laplacian_variance(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def assess_image_quality(image_bgr, face_width_fraction: float | None, roll_deg: float | None) -> dict:
    reasons, warnings = [], []

    h, w = image_bgr.shape[:2]
    min_dim = min(h, w)
    if min_dim < MIN_DIM_REJECT:
        reasons.append(f"resolution_too_low({min_dim}px)")
    elif min_dim < MIN_DIM_WARN:
        warnings.append(f"resolution_low({min_dim}px)")

    if face_width_fraction is None:
        reasons.append("no_face_detected")
    else:
        if face_width_fraction < FACE_FRAC_REJECT:
            reasons.append("face_too_small")
        elif face_width_fraction < FACE_FRAC_WARN:
            warnings.append("face_small")
        if roll_deg is not None:
            abs_roll = abs(roll_deg)
            if abs_roll > ROLL_REJECT_DEG:
                reasons.append(f"head_angle_extreme({abs_roll:.1f}deg)")
            elif abs_roll > ROLL_WARN_DEG:
                warnings.append(f"head_angle_tilted({abs_roll:.1f}deg)")

    scale = min(1.0, BLUR_SAMPLE_WIDTH / w)
    small = cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mean_luma = float(gray.mean())
    clipped_fraction = float(np.mean((gray < 8) | (gray > 247)))
    blur_variance = _laplacian_variance(gray)

    if mean_luma < LUMA_DARK_REJECT:
        reasons.append("severely_underexposed")
    elif mean_luma < LUMA_DARK_WARN:
        warnings.append("underexposed")
    if mean_luma > LUMA_BRIGHT_REJECT:
        reasons.append("severely_overexposed")
    elif mean_luma > LUMA_BRIGHT_WARN:
        warnings.append("overexposed")
    if clipped_fraction > CLIPPED_FRACTION_WARN:
        warnings.append("high_clipping")

    if blur_variance < BLUR_VARIANCE_REJECT:
        reasons.append("severely_blurry")
    elif blur_variance < BLUR_VARIANCE_WARN:
        warnings.append("possibly_blurry")

    diagnostics = {
        "min_dim": min_dim, "mean_luma": round(mean_luma, 2),
        "clipped_fraction": round(clipped_fraction, 4),
        "blur_variance": round(blur_variance, 2),
        "face_width_fraction": face_width_fraction, "roll_deg": roll_deg,
    }

    if reasons:
        return {"level": "reject", "reasons": reasons, "diagnostics": diagnostics}
    if warnings:
        return {"level": "warning", "reasons": warnings, "diagnostics": diagnostics}
    return {"level": "good", "reasons": [], "diagnostics": diagnostics}
