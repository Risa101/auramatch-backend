"""
skin_model.py

Season classifier trained on skin color only: multinomial logistic regression
over the six numbers extract_skin_features() already returns (L*, a*, b*,
chroma, hue angle, ITA). Trained on the Deep Armocromia TRAIN partition
(expert labels) plus 73 Thai adult photos labeled with colorwise.me, the Thai
photos weighted x20 (personal_color_ai/skin_model/train_skin_model.py and
colorwise_thai/README.md). Coefficients below are copied from that fit.

Held-out results:
  Thai adults, 4-fold CV x10 (Thai photos never in their own test fold):
      acc 52.1%, balanced 44.5%   (Armocromia-only model: 48.0% / 38.9%;
      classical season.py: 28.6% / 33.5%)
  Deep Armocromia TEST (n=910): acc 36.3%, balanced 37.0% (classical 27.4%)

Pure Python (no numpy/sklearn at inference) so it adds nothing to the deploy.
"""
from __future__ import annotations

import math

CLASSES = ("spring", "summer", "autumn", "winter")
FEATURES = ("L", "a", "b", "chroma", "hue_angle", "ita")

_MEAN = [72.32059103, 7.80188000, 13.53043859, 16.55691727, 71.62453571, 55.60164100]
_SCALE = [11.78901755, 7.10516979, 9.13196765, 10.18267456, 49.79162148, 29.51864652]
_COEF = [
    [0.12257092, -0.07392966, -0.07321110, 0.05901153, -0.04439600, -0.09326024],
    [-0.03648748, 0.35002235, -0.02317917, -0.36458588, 0.02915713, 0.38405705],
    [-0.16838607, -0.10741546, 0.61916186, -0.21791884, 0.05578400, -0.08810463],
    [0.08230263, -0.16867723, -0.52277159, 0.52349319, -0.04054513, -0.20269217],
]
_INTERCEPT = [0.05943689, -0.04530525, -0.05856669, 0.04443504]


def feature_vector(features: dict) -> list[float]:
    L, a, b = features["skin_lab"]
    return [float(L), float(a), float(b), float(features["chroma"]), float(features["hue_angle"]), float(features["ita"])]


def predict_season(features: dict) -> list[tuple[str, float]]:
    """Ranked [(season, probability)], highest first."""
    x = [(v - m) / s for v, m, s in zip(feature_vector(features), _MEAN, _SCALE)]
    logits = [sum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(_COEF, _INTERCEPT)]
    top = max(logits)
    exps = [math.exp(z - top) for z in logits]
    total = sum(exps)
    probs = [(CLASSES[i], round(e / total, 4)) for i, e in enumerate(exps)]
    return sorted(probs, key=lambda kv: kv[1], reverse=True)
