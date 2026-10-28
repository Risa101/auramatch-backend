"""
color_engine_service.py

DEPRECATED — superseded by services/personal_color_service.py.

This proxied to a separately-hosted auramatchgenz FastAPI service via
COLOR_ENGINE_URL, which defaulted (and in this project's checked-in .env,
still defaults) to http://127.0.0.1:8010 — a manual local `uvicorn` process,
never an actually-deployed second Railway service. In production this call
always raised ColorEngineUnavailable and was silently swallowed by the
frontend (see the AI accuracy audit, Critical Finding #2): the most
scientifically rigorous classifier in the whole system never ran for a real
user.

services/personal_color_service.py fixes this by porting the same math
(white balance, landmark-restricted skin mask, CIELAB/CIEDE2000, ranked
season scoring) directly into this process — no second service, no
COLOR_ENGINE_URL, no silent-fallback risk. /api/color-engine/analyze and this
module are kept only for backward compatibility with any other caller that
might still hit this route; do not add new consumers.
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
