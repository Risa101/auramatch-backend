import numpy as np

from services.color_engine.skin_mask import CHEEK_LEFT, FOREHEAD, build_skin_mask
from services.eval.ablation import (
    baseline_a_legacy, skin_sampling_ablation, white_balance_ablation,
)

CANVAS = (120, 120)
NUM_LANDMARKS = 468


def _landmarks():
    lm = np.full((NUM_LANDMARKS, 2), 1.0)
    angles = np.linspace(0, 2 * np.pi, len(CHEEK_LEFT), endpoint=False)
    lm[CHEEK_LEFT] = np.stack([60 + 25 * np.cos(angles), 70 + 25 * np.sin(angles)], axis=1)
    angles_f = np.linspace(0, 2 * np.pi, len(FOREHEAD), endpoint=False)
    lm[FOREHEAD] = np.stack([60 + 15 * np.cos(angles_f), 30 + 15 * np.sin(angles_f)], axis=1)
    return lm


def _skin_image():
    return np.full((*CANVAS, 3), (120, 150, 200), dtype=np.uint8)


def test_skin_sampling_ablation_returns_all_four_methods():
    image = _skin_image()
    mask = build_skin_mask(image, _landmarks())
    result = skin_sampling_ablation(image, mask)
    for method in ("mean", "median", "trimmed_mean", "kmeans"):
        assert method in result
        assert len(result[method]) == 3  # L, a, b


def test_skin_sampling_ablation_methods_agree_on_a_clean_uniform_region():
    """On a perfectly uniform region there's no contamination to reject, so
    every aggregation strategy should land on essentially the same color.
    """
    image = _skin_image()
    mask = build_skin_mask(image, _landmarks())
    result = skin_sampling_ablation(image, mask)
    L_values = [result[m][0] for m in ("mean", "median", "trimmed_mean", "kmeans")]
    assert max(L_values) - min(L_values) < 0.5


def test_white_balance_ablation_reports_a_lab_delta():
    image = _skin_image()
    mask = build_skin_mask(image, _landmarks())
    result = white_balance_ablation(image, mask)
    assert result["raw"]["valid"] is True
    assert result["white_balanced"]["valid"] is True
    assert result["delta_e_raw_vs_wb"] is not None
    assert result["delta_e_raw_vs_wb"] >= 0


def test_baseline_a_legacy_runs_on_real_image_bytes():
    import cv2
    ok, buf = cv2.imencode(".png", _skin_image())
    assert ok
    result = baseline_a_legacy(buf.tobytes())
    assert "season" in result
