"""Instrumentation-only wrapper for the ROOT-CAUSE raw-image pipeline
validation experiment. Every function here CALLS existing, unmodified
production code (services.personal_color_service.analyze_personal_color,
services.eval.quality_port.assess_image_quality,
services.color_engine.season.decide_season) and exposes intermediate
values that those functions already compute internally but don't all
surface through their own public return shape. It duplicates NO algorithm:
every color/geometry/scoring computation here is a call into existing code,
never a re-derivation.

Four things live in this module:

  1. full_diagnostics() -- runs the real pipeline on one image and flattens
     every diagnostic value this experiment's protocol asks for into one
     dict (quality gate result, L*/a*/b*, hue/chroma/ITA, K-means
     diagnostics, all 4 season scores, winner, runner-up, margin,
     reliability).
  2. select_balanced_sample() -- deterministic, season-balanced subsampling
     of a label set (no image processing; pure bookkeeping).
  3. categorize_rejection() -- maps quality_port.py's reason strings to the
     coarse categories the rejection-accounting report needs (blur,
     exposure, face_size, head_angle, resolution, no_face, other).
  4. ablate_season_input() -- DIAGNOSTIC ABLATION, explicitly labeled as
     such everywhere it's used. Calls the UNMODIFIED decide_season() with a
     COPY of a features dict where exactly one raw input (hue_angle,
     chroma, or skin_lab's L) has been reset to season.py's own pivot value
     for that axis -- the well-defined, reproducible way to neutralize one
     axis's contribution to the distance calculation without touching
     decide_season() or SeasonModel itself. This is a sensitivity probe on
     REAL captured inputs, never a second scoring algorithm.

Known, documented gap: `dominant_weight_fraction` (one of extract.py's
K-means diagnostics) is computed internally by extract_skin_features() but
is NOT part of analyze_personal_color()'s public return dict -- only
kept_weight_fraction and max_centroid_delta_e are. Surfacing it here would
require calling extract_skin_features() a second time on the same masked
pixels (duplicate work, not duplicate algorithm) purely for one extra
diagnostic field; not done, noted here instead of silently omitted.
"""

from __future__ import annotations

import copy

from services.color_engine import color_math as cm
from services.color_engine.season import DEFAULT_MODEL, decide_season
from services.eval.quality_port import assess_image_quality
from services.personal_color_service import analyze_personal_color

SEASONS = ["Spring", "Summer", "Autumn", "Winter"]

REJECTION_CATEGORY_MAP = [
    # (substring in a quality_port.py reason string, category)
    ("resolution_too_low", "resolution"),
    ("resolution_low", "resolution"),
    ("no_face_detected", "no_face"),
    ("face_too_small", "face_size"),
    ("face_small", "face_size"),
    ("head_angle", "head_angle"),
    ("severely_underexposed", "exposure"),
    ("underexposed", "exposure"),
    ("severely_overexposed", "exposure"),
    ("overexposed", "exposure"),
    ("high_clipping", "exposure"),
    ("severely_blurry", "blur"),
    ("possibly_blurry", "blur"),
]


def categorize_rejection(reasons) -> str:
    """Maps quality_port.py's own reason strings (verbatim, not re-derived)
    to one coarse category for the rejection-accounting summary. Returns
    "other" for anything unrecognized rather than raising -- a category
    scheme drifting out of sync with quality_port.py's actual strings
    should degrade to a visible "other" bucket, not crash a whole run.
    """
    for reason in reasons or []:
        for substr, category in REJECTION_CATEGORY_MAP:
            if substr in reason:
                return category
    return "other" if reasons else "none"


def full_diagnostics(image_bytes: bytes, landmarks_norm, *, face_box=None, roll_deg=None, face_detected=True) -> dict:
    """Runs the REAL pipeline (assess_image_quality + analyze_personal_color,
    both unmodified) on one image and returns every diagnostic field this
    experiment's protocol requires, flattened into one dict.

    `face_box`/`roll_deg`/`face_detected` come from the SAME landmark export
    record scripts/export_landmarks.py already produces (faceBox, rollDeg,
    faceDetected) -- this function does not recompute geometry, it accepts
    what was already measured, exactly like scripts/evaluate_personal_color.py
    already does for its own quality-gate call.
    """
    import cv2 as cv
    import numpy as np

    file_bytes = np.frombuffer(image_bytes, np.uint8)
    bgr = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

    quality = assess_image_quality(
        bgr,
        face_width_fraction=(face_box.get("width") if face_box else None),
        roll_deg=roll_deg,
    )

    diag = {
        "face_landmark_available": bool(face_detected),
        "quality_gate_level": quality["level"],
        "quality_gate_reasons": ";".join(quality["reasons"]),
        "quality_rejection_category": categorize_rejection(quality["reasons"]) if quality["level"] == "reject" else None,
        "quality_min_dim": quality["diagnostics"]["min_dim"],
        "quality_mean_luma": quality["diagnostics"]["mean_luma"],
        "quality_blur_variance": quality["diagnostics"]["blur_variance"],
        "quality_face_width_fraction": quality["diagnostics"]["face_width_fraction"],
        "quality_roll_deg": quality["diagnostics"]["roll_deg"],
    }

    if not face_detected or quality["level"] == "reject":
        diag.update({
            "pipeline_ran": False,
            "valid": False, "reason": "no_face_landmarks" if not face_detected else "quality_gate_rejected",
            "L": None, "a": None, "b": None, "hue_angle": None, "chroma": None,
            "ita": None, "ita_category": None, "confidence": None, "reliability": None,
            "wb_method": None, "wb_trustworthy": None, "kept_weight_fraction": None,
            "max_centroid_delta_e": None, "skin_pixel_count": None,
            "season_spring_score": None, "season_summer_score": None,
            "season_autumn_score": None, "season_winter_score": None,
            "predicted_season": None, "second_season": None, "margin": None,
            "undertone": None, "sub_tone": None, "olive_adjusted": None,
        })
        return diag

    result = analyze_personal_color(image_bytes, landmarks_norm)
    diag["pipeline_ran"] = True
    diag["valid"] = result.get("valid")
    diag["reason"] = result.get("reason")

    if not result.get("valid"):
        diag.update({
            "L": None, "a": None, "b": None, "hue_angle": None, "chroma": None,
            "ita": None, "ita_category": None, "confidence": None, "reliability": None,
            "wb_method": None, "wb_trustworthy": None, "kept_weight_fraction": None,
            "max_centroid_delta_e": None, "skin_pixel_count": result.get("skin_pixel_count"),
            "season_spring_score": None, "season_summer_score": None,
            "season_autumn_score": None, "season_winter_score": None,
            "predicted_season": None, "second_season": None, "margin": None,
            "undertone": None, "sub_tone": None, "olive_adjusted": None,
        })
        return diag

    skin_lab = result.get("skin_lab") or [None, None, None]
    ranked = result.get("ranked_seasons") or []
    scores_by_season = {r["season"]: r["probability"] for r in ranked}
    winner = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None
    margin = (winner["probability"] - runner_up["probability"]) if (winner and runner_up) else None

    q = result.get("quality") or {}
    diag.update({
        "L": skin_lab[0], "a": skin_lab[1], "b": skin_lab[2],
        "hue_angle": float(cm.hue_angle(skin_lab)) if skin_lab[0] is not None else None,
        "chroma": float(cm.chroma(skin_lab)) if skin_lab[0] is not None else None,
        "ita": result.get("ita"), "ita_category": result.get("ita_category"),
        "confidence": result.get("confidence"), "reliability": result.get("reliability"),
        "wb_method": q.get("wb_method"), "wb_trustworthy": q.get("wb_trustworthy"),
        "kept_weight_fraction": q.get("kept_weight_fraction"),
        "max_centroid_delta_e": q.get("max_centroid_delta_e"),
        "skin_pixel_count": result.get("skin_pixel_count"),
        "season_spring_score": scores_by_season.get("Spring"),
        "season_summer_score": scores_by_season.get("Summer"),
        "season_autumn_score": scores_by_season.get("Autumn"),
        "season_winter_score": scores_by_season.get("Winter"),
        "predicted_season": winner["season"] if winner else None,
        "second_season": runner_up["season"] if runner_up else None,
        "margin": margin,
        "undertone": result.get("undertone"),
        "sub_tone": result.get("sub_tone"),
        "olive_adjusted": result.get("olive_adjusted"),
    })
    return diag


def select_balanced_sample(labels: dict, *, per_season: int, seed: int, season_key: str = "season") -> dict:
    """Deterministic, season-balanced subsample of a {image_id: row} label
    dict. Pure bookkeeping -- no image is opened here.

    Determinism: sorts image_ids within each season group before shuffling
    with a seeded RNG, so the SAME (labels, per_season, seed) always
    produces the SAME sample regardless of dict iteration order (which
    Python does not guarantee is stable across processes for non-insertion-
    ordered construction paths). Not cherry-picked -- selection never looks
    at any AuraMatch output, only at the season label and image_id string.
    """
    import random

    by_season = {s: [] for s in SEASONS}
    for image_id, row in labels.items():
        s = row.get(season_key)
        if s in by_season:
            by_season[s].append(image_id)

    rng = random.Random(seed)
    selected = {}
    for s in SEASONS:
        pool = sorted(by_season[s])
        rng.shuffle(pool)
        for image_id in pool[:per_season]:
            selected[image_id] = labels[image_id]
    return selected


def ablate_season_input(features: dict, *, axis: str, model=DEFAULT_MODEL):
    """DIAGNOSTIC ABLATION -- NOT PRODUCTION BEHAVIOR.

    Calls the UNMODIFIED decide_season() with a COPY of `features` where
    exactly one raw input has been reset to the value that makes its axis
    score exactly 0 (neutral) under `model`'s own pivot -- i.e. the warm/
    cool, light/deep, or bright/mute axis stops contributing to the
    season-distance calculation for this one call, while every other axis
    behaves exactly as decide_season() already defines it. `axis` must be
    one of "warm" (neutralizes hue_angle), "value" (neutralizes skin_lab's
    L), or "chroma" (neutralizes chroma). Returns the resulting
    SeasonDecision, same shape decide_season() always returns.

    `model` is read-only here (DEFAULT_MODEL unless the caller passes a
    different SeasonModel instance) -- this function never constructs or
    mutates a SeasonModel, it only reads the pivot fields already defined
    on whichever one is passed in.
    """
    if axis not in ("warm", "value", "chroma"):
        raise ValueError(f"axis must be one of 'warm'/'value'/'chroma', got {axis!r}")

    perturbed = copy.deepcopy(features)
    if axis == "warm":
        perturbed["hue_angle"] = model.hue_pivot
    elif axis == "chroma":
        perturbed["chroma"] = model.chroma_pivot
    elif axis == "value":
        skin_lab = list(perturbed.get("skin_lab") or [None, None, None])
        skin_lab[0] = model.value_pivot
        perturbed["skin_lab"] = skin_lab

    return decide_season(perturbed, model=model)
