"""Ablation / baseline comparison harness (audit real-photo brief, sections
17-20): "we need to know whether the sophisticated components are actually
improving the result. Do not assume they do."

Three real, ALREADY-EXISTING pipelines, compared on the same image rather
than re-implemented from scratch (so nobody has to trust that a
reimplementation matches the original):

  Baseline A: the ORIGINAL production classifier,
              services/undertone_service.py::analyze_season() -- global Lab
              a*/b* threshold mask, NO white balance, single mean over all
              surviving pixels. Still present in this codebase (only
              stopped being CALLED by the frontend, never deleted) so it
              serves as a real, working baseline.
  Baseline B: the new consolidated engine,
              services/personal_color_service.py::analyze_personal_color()
              -- landmark-restricted mask, Shades-of-Gray white balance,
              K-means robust aggregation, 3-axis ranked season scoring.
  Candidate C: "calibrated ColorEngine". Honestly: IDENTICAL to Baseline B
              today. No real labeled data exists yet to justify changing
              services/color_engine/season.py's SeasonModel constants (see
              the audit's own rule against tuning thresholds before real
              evaluation, and season.py's own docstring calling them
              "PROVISIONAL... NOT dataset-calibrated truth"). This function
              still returns a distinct field so the report schema doesn't
              have to change the day a real calibrated model exists --
              until then, treat B and C as the same number.

Plus two true single-variable ablations (vary exactly one component,
everything else held fixed -- unlike the baseline comparison above, which
varies several components at once):

  * white_balance_ablation: RAW vs white-balanced skin Lab on the IDENTICAL
    landmark mask (section 20).
  * skin_sampling_ablation: mean / median / trimmed-mean / K-means
    aggregation on the IDENTICAL masked, white-balanced pixels (section 19).
"""

from __future__ import annotations

import numpy as np

from services.color_engine import color_math as cm
from services.color_engine.extract import extract_skin_features, masked_skin_lab
from services.color_engine.extract import _representative_skin  # internal; reused deliberately for ablation research, not production behavior
from services.personal_color_service import analyze_personal_color
from services.undertone_service import analyze_season as _legacy_analyze_season


def baseline_a_legacy(image_bytes: bytes) -> dict:
    """The original, pre-refactor production classifier. No landmarks
    needed -- it does its own whole-image global-threshold masking.
    """
    try:
        return _legacy_analyze_season(image_bytes)
    except Exception as exc:  # decode failure, etc. -- record, don't crash the batch
        return {"season": None, "error": str(exc)}


def baseline_b_new_engine(image_bytes: bytes, landmarks_norm) -> dict:
    return analyze_personal_color(image_bytes, landmarks_norm)


def candidate_c_calibrated(image_bytes: bytes, landmarks_norm) -> dict:
    """See module docstring: no calibration has happened yet, so this is
    identical to Baseline B today.
    """
    return baseline_b_new_engine(image_bytes, landmarks_norm)


def _mean_agg(lab, weights):
    return np.average(lab, axis=0, weights=weights)


def _median_agg(lab, _weights):
    return np.median(lab, axis=0)


def _trimmed_mean_agg(lab, weights, trim: float = 0.1):
    """Drop the extreme `trim` fraction on each tail, ranked by L* (where
    shadow/specular contamination shows up most), then weighted-mean the rest.
    """
    order = np.argsort(lab[:, 0])
    n = len(order)
    lo, hi = int(n * trim), max(int(n * (1 - trim)), int(n * trim) + 1)
    keep = order[lo:hi]
    return np.average(lab[keep], axis=0, weights=weights[keep])


def skin_sampling_ablation(image_bgr, skin_mask, *, apply_white_balance: bool = True) -> dict:
    """Compare aggregation strategies on IDENTICAL masked+white-balanced
    pixels -- isolates "does K-means actually help" from every other
    variable (audit real-photo brief, section 19).
    """
    lab, weights, wb_method, skin_px = masked_skin_lab(
        image_bgr, skin_mask, apply_white_balance=apply_white_balance
    )
    if skin_px == 0:
        return {"error": "no skin pixels in mask"}

    kmeans_rep, dom_frac, kept_frac, max_cde = _representative_skin(
        lab, weights, k=3, merge_delta_e=10.0, random_state=0
    )
    return {
        "wb_method": wb_method,
        "skin_pixel_count": skin_px,
        "mean": [round(float(v), 4) for v in _mean_agg(lab, weights)],
        "median": [round(float(v), 4) for v in _median_agg(lab, weights)],
        "trimmed_mean": [round(float(v), 4) for v in _trimmed_mean_agg(lab, weights)],
        "kmeans": [round(float(v), 4) for v in kmeans_rep],
        "kmeans_kept_weight_fraction": round(kept_frac, 4),
        "kmeans_max_centroid_delta_e": round(max_cde, 4),
    }


def white_balance_ablation(image_bgr, skin_mask) -> dict:
    """RAW vs white-balanced skin Lab on the SAME mask (section 20). Per the
    brief: WB should improve agreement with ground truth, not merely change
    the number -- this function only reports the Lab/Delta-E shift; whether
    that shift is toward or away from the TRUTH can only be judged once
    these rows are joined against real labels in the evaluation report.
    """
    raw = extract_skin_features(image_bgr, skin_mask, apply_white_balance=False)
    wb = extract_skin_features(image_bgr, skin_mask, apply_white_balance=True)
    delta_e = None
    if raw.get("valid") and wb.get("valid"):
        delta_e = float(cm.delta_e_2000(raw["skin_lab"], wb["skin_lab"]))
    return {"raw": raw, "white_balanced": wb, "delta_e_raw_vs_wb": delta_e}
