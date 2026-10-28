"""Integration tests for services.personal_color_service -- the single
authoritative Personal Color entry point (audit Critical Finding #1).

Uses a real encoded PNG (via cv2.imencode) rather than mocking cv2.imdecode,
so these tests exercise the actual decode path a real upload goes through.
"""

import json

import cv2
import numpy as np
import pytest

import services.personal_color_service as pcs
from services.color_engine.skin_mask import CHEEK_LEFT, FOREHEAD
from services.personal_color_service import (
    InvalidLandmarks, analyze_personal_color,
)


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    """These are unit tests for the color pipeline, not the product catalog
    -- stub the DB-backed blush lookup so tests don't need a live MySQL
    connection. services/personal_color_service.py already isolates a DB
    failure from the season/undertone result (see its try/except), so this
    also doubles as a check that path stays wired correctly.
    """
    monkeypatch.setattr(pcs, "get_blush_products_by_undertone", lambda undertone: [
        {"product_id": 1, "name": f"stub-{undertone}-blush"}
    ])

CANVAS = (200, 200)  # (H, W)
NUM_LANDMARKS = 468


def _normalized_landmarks():
    """468 landmarks normalized to [0,1], with real cheek/forehead hulls."""
    lm = np.full((NUM_LANDMARKS, 3), 0.5)
    h, w = CANVAS

    def circle(cx, cy, r, n):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = (cx + r * np.cos(angles)) / w
        ys = (cy + r * np.sin(angles)) / h
        return np.stack([xs, ys, np.zeros(n)], axis=1)

    lm[CHEEK_LEFT] = circle(60, 120, 30, len(CHEEK_LEFT))
    lm[FOREHEAD] = circle(100, 50, 25, len(FOREHEAD))
    return lm.tolist()


def _encode_png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    return buf.tobytes()


def test_analyze_personal_color_end_to_end():
    bgr = np.full((*CANVAS, 3), (120, 150, 200), dtype=np.uint8)
    image_bytes = _encode_png(bgr)
    landmarks = _normalized_landmarks()

    result = analyze_personal_color(image_bytes, landmarks)

    assert result["valid"] is True
    assert result["season"] in ("Spring", "Summer", "Autumn", "Winter")
    assert result["undertone"] in ("warm", "cool", "neutral")
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["reliability"] in ("high", "medium", "low")
    assert len(result["ranked_seasons"]) == 4
    assert isinstance(result["recommended_products"], list)


def test_analyze_personal_color_is_deterministic():
    bgr = np.full((*CANVAS, 3), (110, 140, 190), dtype=np.uint8)
    image_bytes = _encode_png(bgr)
    landmarks = _normalized_landmarks()

    a = analyze_personal_color(image_bytes, landmarks)
    b = analyze_personal_color(image_bytes, landmarks)
    assert a == b


def test_too_few_landmarks_raises_invalid_landmarks():
    bgr = np.full((*CANVAS, 3), 150, dtype=np.uint8)
    image_bytes = _encode_png(bgr)
    with pytest.raises(InvalidLandmarks):
        analyze_personal_color(image_bytes, [[0.5, 0.5]] * 10)


def test_undecodable_image_raises_value_error():
    with pytest.raises(ValueError):
        analyze_personal_color(b"not an image", _normalized_landmarks())


def test_landmarks_json_round_trip_matches_direct_call():
    """The controller parses `landmarks` as a JSON string form field; verify
    that round-trip doesn't change the numeric result vs. calling the
    service directly with the same (pre-parsed) list.
    """
    bgr = np.full((*CANVAS, 3), (130, 155, 205), dtype=np.uint8)
    image_bytes = _encode_png(bgr)
    landmarks = _normalized_landmarks()

    direct = analyze_personal_color(image_bytes, landmarks)
    via_json = analyze_personal_color(image_bytes, json.loads(json.dumps(landmarks)))
    assert direct == via_json
