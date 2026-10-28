"""
skin_model.py

Season classifier trained on skin color only: multinomial logistic regression
over the six numbers extract_skin_features() already returns (L*, a*, b*,
chroma, hue angle, ITA). Trained on the Deep Armocromia TRAIN partition
(expert labels) plus 66 Thai adult photos labeled with colorwise.me, the Thai
photos weighted x20 (personal_color_ai/skin_model/train_skin_model.py and
colorwise_thai/README.md). Coefficients below are copied from that fit.

Held-out results:
  Thai adults, 4-fold CV x10 (Thai photos never in their own test fold):
      acc 53.2%, balanced 45.9%   (Armocromia-only model: 51.5% / 40.8%;
      classical season.py: 28.6% / 33.5%)
  Deep Armocromia TEST (n=910): acc 34.9%, balanced 35.7% (classical 27.4%)

Pure Python (no numpy/sklearn at inference) so it adds nothing to the deploy.
"""
from __future__ import annotations

import math

CLASSES = ("spring", "summer", "autumn", "winter")
FEATURES = ("L", "a", "b", "chroma", "hue_angle", "ita")

_MEAN = [72.32773763, 7.79376971, 13.53492874, 16.55442764, 71.67119836, 55.60575789]
_SCALE = [11.79141069, 7.10207153, 9.13601427, 10.18794606, 49.81152881, 29.51815432]
_COEF = [
    [0.11691055, -0.12594655, -0.26014791, 0.28053209, -0.07899728, -0.07517246],
    [-0.04414631, 0.27409763, -0.26907722, -0.11686570, -0.01532327, 0.37897503],
    [-0.18357844, -0.10958611, 0.63905380, -0.20145690, 0.06382363, -0.04008406],
    [0.11081421, -0.03856496, -0.10982867, 0.03779051, 0.03049692, -0.26371851],
]
_INTERCEPT = [0.06035595, -0.05061013, -0.06010880, 0.05036298]


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
