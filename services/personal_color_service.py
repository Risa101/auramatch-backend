"""
personal_color_service.py

THE single authoritative source for Personal Color season + skin undertone.

Consolidates what used to be three independent, overwriting classifiers
(Gemini's season guess, the global-Lab-threshold "/api/season/analyze", and
an unreachable-in-production "/api/color-engine/analyze" proxy -- see the AI
accuracy audit, Critical Findings #1-#3) into one deterministic pipeline:

    image bytes + browser-computed face landmarks
      -> decode image (cv2)
      -> denormalize landmarks to pixel space
      -> landmark-restricted skin mask (services/color_engine/skin_mask.py)
      -> Shades-of-Gray white balance (services/color_engine/white_balance.py)
      -> robust (K-means-cleaned) skin Lab (services/color_engine/extract.py)
      -> 3-axis ranked season decision + reliability (services/color_engine/season.py)
      -> undertone label reused directly from the SAME warm/cool measurement
      -> blush products for that undertone (existing product_service)

Landmarks are computed client-side (MediaPipe FaceLandmarker, already
vendored in the frontend for the makeup studio) rather than re-run here --
see services/color_engine/skin_mask.py's module docstring for why: it avoids
a second, heavier MediaPipe Python dependency and a duplicate detection pass.

Known limitation (documented, not hidden): this extractor does not sample
hair or iris regions, so it never claims the "high" confidence tier the
original auramatchgenz design reserved for contrast-validated reads, and the
season model's bright/mute axis runs on chroma alone (no skin<->hair /
skin<->iris contrast term). See services/color_engine/extract.py's docstring.
"""

from __future__ import annotations

import os

import numpy as np
import cv2 as cv

from services.color_engine.skin_mask import build_skin_mask
from services.color_engine.extract import extract_skin_features
from services.color_engine.season import decide_season
from services.product_service import get_blush_products_by_undertone

# Season/subclass -> warm|cool, per season.py's own _SEASON_TARGETS
# (spring/autumn warm, summer/winter cool) -- the AI path re-derives
# undertone from THIS, not from decision.warm_cool, since that field is
# itself computed from hue_angle, the exact feature this session measured
# as having ~zero real warm/cool separability (see PERSONAL_COLOR_ENGINE
# below).
_SEASON_UNDERTONE = {"spring": "warm", "autumn": "warm", "summer": "cool", "winter": "cool"}

MIN_LANDMARKS = 468


class InvalidLandmarks(ValueError):
    pass


def _decode_image(image_bytes: bytes) -> np.ndarray:
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    bgr = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("ไม่สามารถอ่านไฟล์รูปภาพได้ (รูปแบบไฟล์ไม่ถูกต้องหรือเสียหาย)")
    return bgr


def _denormalize_landmarks(landmarks_norm, width: int, height: int) -> np.ndarray:
    """landmarks_norm: list of [x, y] or [x, y, z] in [0, 1] (fractions of the
    SAME image being analyzed here). Returns (N, 2) pixel-space float array.
    """
    arr = np.asarray(landmarks_norm, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise InvalidLandmarks("landmarks must be a list of [x, y] (or [x, y, z]) pairs")
    if arr.shape[0] < MIN_LANDMARKS:
        raise InvalidLandmarks(
            f"expected >= {MIN_LANDMARKS} face landmarks (MediaPipe face mesh topology), "
            f"got {arr.shape[0]}"
        )
    px = arr[:, :2].copy()
    px[:, 0] *= width
    px[:, 1] *= height
    return px


def analyze_personal_color(image_bytes: bytes, landmarks_norm) -> dict:
    """Run the full personal-color pipeline. Returns a dict; check ``valid``.

    Raises InvalidLandmarks / ValueError for malformed input (caller maps
    these to 400s) -- a genuine failure must surface as
    ANALYSIS_FAILED, never silently become a fabricated season.
    """
    bgr = _decode_image(image_bytes)
    h, w = bgr.shape[:2]
    landmarks_px = _denormalize_landmarks(landmarks_norm, w, h)

    skin_mask = build_skin_mask(bgr, landmarks_px)
    features = extract_skin_features(bgr, skin_mask)

    if not features.get("valid"):
        return {
            "valid": False,
            "reason": features.get("reason", "extraction_failed"),
            "skin_pixel_count": features.get("skin_pixel_count", 0),
        }

    decision = decide_season(features)
    if not decision.valid:
        return {"valid": False, "reason": decision.reason}

    # Reuse the SAME warm/cool axis for undertone -- one measurement, not a
    # second independent classifier (audit Critical Finding #1).
    undertone = decision.warm_cool  # "warm" | "cool" | "neutral"
    season_title = decision.top_season.capitalize() if decision.top_season else None
    sub_tone = decision.sub_tone
    confidence = decision.top_confidence
    ranked = [{"season": s.capitalize(), "probability": p} for s, p in decision.ranked]
    warm_cool_axis = {"label": decision.warm_cool, "score": decision.warm_cool_score}
    flags = list(decision.flags)

    # Optional AI decision step (personal_color_ai/'s tf_efficientnetv2_l,
    # 53.7% TEST accuracy, no season-collapse -- see
    # evaluation_dataset/raw_validation/RAW_PIPELINE_VALIDATION.md).
    # Overrides season/sub_tone/confidence/ranked_seasons/undertone/
    # axes.warm_cool -- NOT light_deep/bright_mute, which are independent of
    # hue_angle and stay from the classical extraction either way.
    if os.getenv("PERSONAL_COLOR_ENGINE", "classical") == "ai":
        try:
            from services.color_engine.ai_season import predict_ai

            ai_result = predict_ai(bgr)
            season_key = ai_result["season"]
            season_title = season_key.capitalize()
            sub_tone = f"{season_title} {ai_result['subclass'].capitalize()}"
            confidence = ai_result["season_confidence"]
            ranked = [{"season": r["season"].capitalize(), "probability": r["probability"]} for r in ai_result["ranked_seasons"]]
            undertone = _SEASON_UNDERTONE[season_key]
            warm_cool_axis = {"label": undertone, "score": None}
            flags.append(
                "ai_engine: season/sub_tone/confidence/ranked_seasons/undertone/axes.warm_cool "
                "from CNN (tf_efficientnetv2_l); axes.light_deep/bright_mute from classical extraction"
            )
        except Exception as exc:
            print(f"⚠️ AI personal-color engine unavailable, falling back to classical: {exc}")
            flags.append(f"ai_engine_unavailable: {exc}")

    try:
        products = get_blush_products_by_undertone(undertone)
    except Exception as exc:
        # A product-recommendation DB hiccup must not take down the season/
        # undertone result itself -- that's the primary, objective output;
        # the shopping list is secondary.
        print(f"⚠️ get_blush_products_by_undertone failed, returning []: {exc}")
        products = []

    return {
        "valid": True,
        "season": season_title,
        "sub_tone": sub_tone,
        "confidence": confidence,
        "reliability": decision.reliability,
        "ranked_seasons": ranked,
        "axes": {
            "warm_cool": warm_cool_axis,
            "light_deep": {"label": decision.light_deep, "score": decision.light_deep_score},
            "bright_mute": {"label": decision.bright_mute, "score": decision.bright_mute_score},
        },
        "olive_adjusted": decision.olive_adjusted,
        "flags": flags,
        "undertone": undertone,
        "skin_lab": features.get("skin_lab"),
        "ita": features.get("ita"),
        "ita_category": features.get("ita_category"),
        "skin_pixel_count": features.get("skin_pixel_count"),
        "quality": {
            "wb_method": features.get("wb_method"),
            "wb_trustworthy": features.get("wb_trustworthy"),
            "kept_weight_fraction": features.get("kept_weight_fraction"),
            "max_centroid_delta_e": features.get("max_centroid_delta_e"),
        },
        "recommended_products": products,
    }
