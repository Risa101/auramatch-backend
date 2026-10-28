"""Illumination / white-balance correction for AuraMatch.

Ported verbatim from auramatchgenz/backend/app/color_engine/white_balance.py.
White balance is the #1 error source for skin-color accuracy (see the AI
accuracy audit, Critical Finding #3) -- the classical-CV endpoint this project
shipped before (services/undertone_service.py) had none at all. This runs
before ANY objective color measurement.

Method: Shades of Gray (Finlayson & Trezzi, 2004) -- a Minkowski p-norm
generalization that spans Gray World (p=1) and White-Patch / max-RGB
(p -> infinity). Defaults to p=6, the value Finlayson & Trezzi reported as
best on average.

Correction is a von Kries diagonal (per-channel gain) transform applied in
LINEAR light: linearize the sRGB input, estimate the illuminant, apply the
gains, then re-encode to sRGB (sRGB-in / sRGB-out).
"""

from __future__ import annotations

import numpy as np

from .color_math import chromatic_adaptation_matrix, linear_to_srgb, srgb_to_linear


def estimate_illuminant(image_linear, p: float = 6, mask=None):
    """Estimate the scene illuminant from a LINEAR-light image (Shades of Gray).

    Args:
        image_linear: array of shape ``(..., 3)`` in linear RGB.
        p: Minkowski norm order. ``p=1`` -> Gray World, ``p=inf`` -> White-Patch
           (max-RGB). Default 6.
        mask: optional boolean array whose shape matches the leading axes of
            ``image_linear``. Only ``True`` pixels are used.

    Returns:
        ``(3,)`` illuminant estimate, normalized to unit Euclidean length.
    """
    arr = np.asarray(image_linear, dtype=float).reshape(-1, 3)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool).reshape(-1)]
    if arr.size == 0:
        raise ValueError("no pixels to estimate the illuminant from")

    if np.isinf(p):
        est = arr.max(axis=0)
    else:
        est = np.power(np.mean(np.power(arr, p), axis=0), 1.0 / p)

    norm = np.linalg.norm(est)
    if norm == 0:
        raise ValueError("degenerate (all-black) region; cannot estimate illuminant")
    return est / norm


def correct(image_linear, illuminant):
    """Apply a von Kries diagonal correction in linear light.

    The gains are ``mean(illuminant) / illuminant``, which is invariant to the
    scaling of ``illuminant`` and leaves a neutral scene's brightness unchanged.
    """
    illum = np.asarray(illuminant, dtype=float)
    gains = illum.mean() / illum
    return np.asarray(image_linear, dtype=float) * gains


def white_balance(image, p: float = 6, mask=None, return_illuminant: bool = False,
                  method: str = "vonkries"):
    """White-balance an sRGB image. sRGB in, sRGB out.

    Args:
        image: ``uint8`` array in [0, 255] (returns ``uint8``) or float array in
            [0, 1] (returns float). Shape ``(H, W, 3)``.
        p: Minkowski norm order for the illuminant estimate (default 6).
        mask: optional region mask, see :func:`estimate_illuminant`.
        return_illuminant: if True, also return the ``(3,)`` unit illuminant
            estimate (in linear light).
        method: ``"vonkries"`` (default, per-channel RGB diagonal) or
            ``"bradford"`` (von Kries in Bradford LMS cone space -- more
            accurate undertone for off-neutral colors under a strong cast).

    Returns:
        The corrected image (same dtype as input), or ``(image, illuminant)``.
    """
    arr = np.asarray(image)
    is_uint8 = arr.dtype == np.uint8
    srgb = arr.astype(float) / 255.0 if is_uint8 else arr.astype(float)

    lin = srgb_to_linear(srgb)
    illum = estimate_illuminant(lin, p=p, mask=mask)
    if method == "bradford":
        m = chromatic_adaptation_matrix(illum)
        lin_wb = np.clip(lin @ m.T, 0.0, 1.0)
    elif method == "vonkries":
        lin_wb = np.clip(correct(lin, illum), 0.0, 1.0)
    else:
        raise ValueError(f"method must be 'vonkries' or 'bradford', got {method!r}")
    out = linear_to_srgb(lin_wb)

    if is_uint8:
        out = np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
    return (out, illum) if return_illuminant else out
