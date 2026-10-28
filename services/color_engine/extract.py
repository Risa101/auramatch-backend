"""Skin color feature extraction for AuraMatch -- the module that emits
real numbers from a white-balanced image + skin mask.

Adapted from auramatchgenz/backend/app/color_engine/extract.py, SCOPED DOWN:
the original also samples hair and iris regions (needs a separate hair-
segmentation model) to compute skin<->hair / skin<->iris CIEDE2000 contrast,
which feeds the "bright/mute" axis and the top ("high") confidence tier.

That hair/iris sampling is deliberately NOT ported here -- standing up a
second segmentation model was out of scope for this refactor. This is a real
capability gap, not an oversight: `season.py`'s bright/mute axis degrades
gracefully to chroma alone without a contrast signal (see its `contrasts`
handling), and `_confidence` below caps the result at "medium" instead of
ever claiming the "high" tier the original reserved for contrast-validated
reads. Wiring in a hair segmenter (auramatchgenz already has one:
app/color_engine/hair_segmenter.py) is the natural next step -- see the
audit follow-up's "Recommended Next Step".

Robust skin aggregation (kept from the original, unscoped):
    * K-means (k=3) on the masked skin Lab, weighted by the mid-cheek weight
      map. Merge clusters whose centroid is within ``merge_delta_e`` of the
      dominant (highest summed-weight) cluster, then take the weighted mean
      of the merged pixels. On clean unimodal skin all clusters merge -> an
      unbiased mean; with contamination (stray shadow/specular/edge pixels)
      the far cluster is dropped -> a robust mean. This is what answers the
      audit's Phase 6 ask for "median/trimmed statistics over a simple mean."
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

from . import color_math as cm
from .skin_mask import SkinMask
from .white_balance import white_balance

# Confidence thresholds (provisional, pending real-face calibration -- see
# the audit's ground-truth findings; same caveat as season.py's SeasonModel).
_PX_MIN = 200
_SPREAD_MAX = 20.0
_KEPT_MIN = 0.70

# Skin-locus plausibility -- a trust gate for white balance. Human skin of any
# ethnicity, under correct WB, is warm: clearly positive a*/b* with non-trivial
# chroma. Card-less WB (Shades of Gray) can over-correct on a color-dominant
# scene and drive skin toward gray/cool -- the measured undertone is then
# untrustworthy. Used ONLY to detect that (lower confidence + a flag), NEVER
# to alter the measurement.
_SKIN_MIN_CHROMA = 8.0
_SKIN_MIN_A = 0.0
_SKIN_MIN_B = 3.0


def _to_rgb01(image_bgr) -> np.ndarray:
    img = np.asarray(image_bgr)
    if img.dtype == np.uint8:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    return np.ascontiguousarray(img[..., ::-1]).astype(float)


def _representative_skin(lab, weights, k, merge_delta_e, random_state):
    """Return (rep_lab, dominant_weight_fraction, kept_weight_fraction, max_centroid_dE)."""
    n = len(lab)
    if not np.any(weights > 0):
        weights = np.ones(n, dtype=float)

    n_distinct = len(np.unique(lab, axis=0))
    k_eff = max(1, min(k, n_distinct))
    if k_eff == 1:
        return np.average(lab, axis=0, weights=weights), 1.0, 1.0, 0.0

    km = KMeans(n_clusters=k_eff, n_init=10, random_state=random_state)
    labels = km.fit_predict(lab, sample_weight=weights)
    centers = km.cluster_centers_

    cluster_w = np.array([weights[labels == i].sum() for i in range(k_eff)])
    total_w = cluster_w.sum()
    dom = int(np.argmax(cluster_w))
    dom_frac = float(cluster_w[dom] / total_w)

    d_to_dom = np.array([float(cm.delta_e_2000(centers[i], centers[dom])) for i in range(k_eff)])
    keep = np.isin(labels, np.where(d_to_dom <= merge_delta_e)[0])
    kept_frac = float(weights[keep].sum() / total_w)
    rep = np.average(lab[keep], axis=0, weights=weights[keep])

    return rep, dom_frac, kept_frac, float(d_to_dom.max())


def _skin_wb_trustworthy(rep_lab, feats) -> bool:
    """True if the WB-corrected skin lands in the plausible (warm) skin locus."""
    return (
        feats.chroma >= _SKIN_MIN_CHROMA
        and float(rep_lab[1]) >= _SKIN_MIN_A
        and float(rep_lab[2]) >= _SKIN_MIN_B
    )


def _confidence(skin_px, max_centroid_dE, kept_frac, wb_trustworthy) -> str:
    """Coarse confidence from skin quality alone (no hair/iris signal -- see
    module docstring). Deliberately capped at "medium": the "high" tier is
    reserved for a build that also validates against hair/iris contrast,
    which this scoped-down extractor does not compute -- claiming "high" here
    would overstate what was actually measured.
    """
    if not wb_trustworthy:
        return "low"
    if skin_px < _PX_MIN or kept_frac < _KEPT_MIN or max_centroid_dE > _SPREAD_MAX:
        return "low"
    return "medium"


def masked_skin_lab(image_bgr, skin_mask: SkinMask, *, apply_white_balance: bool = True):
    """The shared first half of feature extraction: white-balance (or not)
    the frame, convert to Lab, and pull out the masked skin pixels + their
    per-pixel weights.

    Split out from :func:`extract_skin_features` so evaluation tooling can
    run the SAME masking/white-balance step and then compare different
    aggregation strategies (mean/median/trimmed-mean/K-means) on identical
    input pixels -- see scripts/evaluate_personal_color.py's skin-sampling
    ablation (audit real-photo brief, section 19). Not used to change
    production behavior; extract_skin_features() below is unchanged.

    Returns (lab_pixels (N,3), weights (N,), wb_method, skin_pixel_count).
    """
    if apply_white_balance:
        wb_image = white_balance(image_bgr)
        wb_method = "shades_of_gray"
    else:
        wb_image = np.asarray(image_bgr)
        wb_method = "precorrected"

    lab_img = cm.srgb_to_lab(_to_rgb01(wb_image))
    mask = skin_mask.mask
    return lab_img[mask], skin_mask.weights[mask], wb_method, int(mask.sum())


def extract_skin_features(
    image_bgr,
    skin_mask: SkinMask,
    *,
    apply_white_balance: bool = True,
    k: int = 3,
    merge_delta_e: float = 10.0,
    min_skin_pixels: int = 100,
    random_state: int = 0,
) -> dict:
    """Extract color features from a face image + skin mask.

    Returns a plain dict compatible with ``season.decide_season``'s expected
    fields (valid, reason, skin_lab, hue_angle, chroma, ita_category,
    confidence, wb_trustworthy), plus diagnostics (skin_pixel_count,
    kept_weight_fraction, max_centroid_delta_e, wb_method).
    """
    lab_pixels, weights, wb_method, skin_px = masked_skin_lab(
        image_bgr, skin_mask, apply_white_balance=apply_white_balance
    )
    if skin_px < min_skin_pixels:
        return {
            "valid": False,
            "reason": f"too few skin pixels ({skin_px} < {min_skin_pixels})",
            "wb_method": wb_method,
            "skin_pixel_count": skin_px,
        }

    rep, dom_frac, kept_frac, max_cdE = _representative_skin(
        lab_pixels, weights, k, merge_delta_e, random_state
    )
    feats = cm.skin_features(rep)
    wb_trust = _skin_wb_trustworthy(rep, feats)
    confidence = _confidence(skin_px, max_cdE, kept_frac, wb_trust)

    return {
        "valid": True,
        "reason": "ok",
        "wb_method": wb_method,
        "skin_lab": [round(float(v), 4) for v in rep],
        "ita": round(feats.ita, 4),
        "hue_angle": round(feats.hue_angle, 4),
        "chroma": round(feats.chroma, 4),
        "ita_category": feats.ita_category,
        "skin_hair_contrast": None,  # not computed -- see module docstring
        "skin_iris_contrast": None,
        "skin_pixel_count": skin_px,
        "dominant_weight_fraction": round(dom_frac, 4),
        "kept_weight_fraction": round(kept_frac, 4),
        "max_centroid_delta_e": round(max_cdE, 4),
        "confidence": confidence,
        "wb_trustworthy": wb_trust,
    }
