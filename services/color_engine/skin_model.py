"""
skin_model.py

Season classifier trained on skin color only: multinomial logistic regression
over the six numbers extract_skin_features() already returns (L*, a*, b*,
chroma, hue angle, ITA). Trained on the Deep Armocromia TRAIN partition
(expert labels) plus 64 Thai adult photos labeled with colorwise.me, the Thai
photos weighted x10 (personal_color_ai/skin_model/train_skin_model.py and
colorwise_thai/README.md). Coefficients below are copied from that fit.

Held-out results:
  Thai adults, 4-fold CV x10 (Thai photos never in their own test fold):
      acc 54.1%, balanced 46.5%   (Armocromia-only model: 50.0% / 40.4%;
      classical season.py: 28.6% / 33.5%)
  Deep Armocromia TEST (n=910): acc 35.6%, balanced 36.3% (classical 27.4%)

Pure Python (no numpy/sklearn at inference) so it adds nothing to the deploy.
"""
from __future__ import annotations

import math

CLASSES = ("spring", "summer", "autumn", "winter")
FEATURES = ("L", "a", "b", "chroma", "hue_angle", "ita")

_MEAN = [72.33247124, 7.79227166, 13.53136357, 16.55092935, 71.67536506, 55.61761860]
_SCALE = [11.79242652, 7.10284052, 9.13697007, 10.18914132, 49.82363788, 29.52067203]
_COEF = [
    [0.11275783, -0.10850747, -0.22816026, 0.24698625, -0.06680757, -0.03554219],
    [-0.01908903, 0.27794203, -0.13673739, -0.28649917, -0.02308508, 0.24534517],
    [-0.18116890, -0.13024928, 0.41954955, 0.00884290, 0.06295223, 0.00916945],
    [0.08750011, -0.03918528, -0.05465190, 0.03067002, 0.02694041, -0.21897243],
]
_INTERCEPT = [0.04202148, -0.04431064, -0.03652074, 0.03880990]


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
