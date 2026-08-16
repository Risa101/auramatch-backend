"""
color_engine_service.py

Proxy to the auramatchgenz color-engine (FastAPI, MediaPipe + CIELAB pipeline).
Runs as a separate service (see COLOR_ENGINE_URL) — this project's own DB/auth
are untouched; we only forward the image and normalize the JSON shape.
"""

import os
import requests

COLOR_ENGINE_URL = os.environ.get("COLOR_ENGINE_URL", "http://127.0.0.1:8010")
_TIMEOUT_SECONDS = 20


class ColorEngineUnavailable(Exception):
    pass


def analyze_with_color_engine(image_bytes: bytes, filename: str, mimetype: str) -> dict:
    try:
        resp = requests.post(
            f"{COLOR_ENGINE_URL}/api/analysis",
            params={"consent": "false"},
            files={"image": (filename, image_bytes, mimetype)},
            timeout=_TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise ColorEngineUnavailable(f"Color engine unreachable at {COLOR_ENGINE_URL}: {exc}") from exc

    if resp.status_code != 200:
        raise ColorEngineUnavailable(f"Color engine returned HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()

    if not data.get("valid"):
        return {
            "success": True,
            "engine": "auramatchgenz-color-engine",
            "valid": False,
            "reason": data.get("reason"),
        }

    diagnosis = data.get("diagnosis", {})
    ranked = [
        {"season": season.capitalize(), "probability": prob}
        for season, prob in diagnosis.get("ranked", [])
    ]

    return {
        "success": True,
        "engine": "auramatchgenz-color-engine",
        "valid": True,
        "season": diagnosis.get("top_season", "").capitalize(),
        "confidence": diagnosis.get("top_confidence"),
        "sub_tone": diagnosis.get("sub_tone"),
        "warm_cool": diagnosis.get("warm_cool"),
        "ranked_seasons": ranked,
        "skin_lab": data.get("skin_lab"),
        "confidence_level": data.get("confidence"),
        "face_shape": data.get("face_shape"),
        "raw": data,
    }
