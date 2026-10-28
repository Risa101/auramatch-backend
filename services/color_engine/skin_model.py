"""
skin_model.py

Season classifier trained on skin color only: multinomial logistic regression
over the six numbers extract_skin_features() already returns (L*, a*, b*,
chroma, hue angle, ITA). Trained on the Deep Armocromia TRAIN partition
(expert labels) plus 71 Thai adult photos labeled with colorwise.me, the Thai
photos weighted x20 (personal_color_ai/skin_model/train_skin_model.py and
colorwise_thai/README.md). Coefficients below are copied from that fit.

Held-out results:
  Thai adults, 4-fold CV x10 (Thai photos never in their own test fold):
      acc 54.1%, balanced 47.1%   (Armocromia-only model: 48.0% / 38.0%;
      classical season.py: 28.6% / 33.5%)
  Deep Armocromia TEST (n=910): acc 35.6%, balanced 36.3% (classical 27.4%)

Pure Python (no numpy/sklearn at inference) so it adds nothing to the deploy.
"""
from __future__ import annotations

import math

CLASSES = ("spring", "summer", "autumn", "winter")
FEATURES = ("L", "a", "b", "chroma", "hue_angle", "ita")

_MEAN = [72.32122387, 7.80057760, 13.53156222, 16.55729199, 71.63693010, 55.59836721]
_SCALE = [11.79201316, 7.10649001, 9.13397193, 10.18527919, 49.80038908, 29.52555367]
_COEF = [
    [0.11406394, -0.07905208, -0.09522863, 0.08426837, -0.05117327, -0.08158636],
    [-0.04713596, 0.34648564, -0.04022643, -0.33914323, 0.02259948, 0.40325976],
    [-0.16081651, -0.08415197, 0.72246357, -0.33436768, 0.07568289, -0.10675170],
    [0.09388853, -0.18328159, -0.58700850, 0.58924255, -0.04710910, -0.21492170],
]
_INTERCEPT = [0.06126122, -0.04342929, -0.06355699, 0.04572506]


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
