"""Developer-only visual debugging for the real-photo evaluation harness
(audit real-photo brief, section 22): "when the AI produces an obviously
wrong result, developers need to SEE what pixels the algorithm actually
used."

Never wired into the production API -- only scripts/evaluate_personal_color.py's
--debug flag calls this, and only writes into the evaluator's own local
--output directory (see the brief's section 28: keep evaluation local by
default, never expose this to normal users).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from services.color_engine import color_math as cm


def _lab_to_bgr_u8(lab) -> tuple:
    rgb = np.clip(cm.lab_to_srgb(np.asarray(lab, dtype=float)), 0.0, 1.0)
    r, g, b = (np.round(rgb * 255.0)).astype(int)
    return int(b), int(g), int(r)


def _draw_landmarks(image_bgr, landmarks_px, color=(0, 200, 0), radius=1):
    canvas = np.array(image_bgr).copy()
    for x, y in np.rint(landmarks_px).astype(int):
        cv2.circle(canvas, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return canvas


def _draw_mask_overlay(image_bgr, skin_mask, alpha=0.5):
    canvas = np.array(image_bgr).copy()
    overlay = canvas.copy()
    overlay[skin_mask.forehead] = (0, 165, 255)  # orange, BGR
    overlay[skin_mask.cheeks] = (0, 255, 0)      # green, BGR
    blended = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0)
    region = skin_mask.mask
    canvas[region] = blended[region]
    return canvas


def save_debug_bundle(
    *,
    image_bgr,
    landmarks_px,
    skin_mask,
    wb_image,
    rep_lab,
    out_dir: Path,
    stem: str,
) -> list[str]:
    """Writes numbered files (01_original.jpg ... 06_representative_color.jpg)
    into out_dir/stem/. Returns the list of filenames written.

    Covers every item the brief's section 22 lists: original, landmarks,
    cheek+forehead masks (+ implicitly what's excluded, by what's NOT
    tinted), white-balanced image, sampled skin pixels, final representative
    color.
    """
    bundle_dir = out_dir / stem
    bundle_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def _save(name, arr):
        path = bundle_dir / name
        cv2.imwrite(str(path), arr)
        written.append(name)

    _save("01_original.jpg", image_bgr)
    _save("02_landmarks.jpg", _draw_landmarks(image_bgr, landmarks_px))
    _save("03_skin_mask_cheeks_forehead.jpg", _draw_mask_overlay(image_bgr, skin_mask))
    _save("04_white_balanced.jpg", wb_image)

    sampled = np.zeros_like(image_bgr)
    sampled[skin_mask.mask] = wb_image[skin_mask.mask]
    _save("05_sampled_skin_pixels_only.jpg", sampled)

    if rep_lab is not None:
        swatch = np.full((120, 120, 3), _lab_to_bgr_u8(rep_lab), dtype=np.uint8)
        _save("06_representative_color.jpg", swatch)

    return written
