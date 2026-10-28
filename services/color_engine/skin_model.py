"""
skin_model.py

Season classifier trained on skin color only: multinomial logistic regression
over the six numbers extract_skin_features() already returns (L*, a*, b*,
chroma, hue angle, ITA). Trained on the Deep Armocromia TRAIN partition
(expert labels) by personal_color_ai/skin_model/train_skin_model.py; the
coefficients below are that script's output, copied verbatim.

Held-out results (see that script's results/skin_model_results.json):
  Deep Armocromia TEST (n=910): acc 0.359, balanced 0.365
      (season.py classical on the same photos: acc 0.274, balanced 0.299)
  Thai adults vs colorwise.me (n=59, never trained on): acc 0.475, balanced 0.399
      (classical: acc 0.305, balanced 0.346)

Pure Python (no numpy/sklearn at inference) so it adds nothing to the deploy.
"""
from __future__ import annotations

import math

CLASSES = ("spring", "summer", "autumn", "winter")
FEATURES = ("L", "a", "b", "chroma", "hue_angle", "ita")

_MEAN = [72.42901043, 7.75270692, 13.49497640, 16.49939153, 71.71232988, 55.80467053]
_SCALE = [11.81780164, 7.11206723, 9.16332275, 10.22432960, 49.74119900, 29.59499321]
_COEF = [
    [0.11967195, -0.09798470, -0.23685204, 0.25289474, -0.04958460, -0.00076530],
    [0.00541725, 0.25396749, -0.01030384, -0.42357322, -0.05374505, 0.11638031],
    [-0.18437744, -0.10775035, 0.28212673, 0.11129417, 0.07843731, 0.06425361],
    [0.05928824, -0.04823243, -0.03497086, 0.05938431, 0.02489233, -0.17986862],
]
_INTERCEPT = [0.02085028, -0.03505139, -0.01580157, 0.03000268]


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
