"""
skin_model.py

Season classifier trained on skin color only: multinomial logistic regression
over the six numbers extract_skin_features() already returns (L*, a*, b*,
chroma, hue angle, ITA). Trained on the Deep Armocromia TRAIN partition
(expert labels) plus 68 Thai adult photos labeled with colorwise.me, the Thai
photos weighted x20 (personal_color_ai/skin_model/train_skin_model.py and
colorwise_thai/README.md). Coefficients below are copied from that fit.

Held-out results:
  Thai adults, 4-fold CV x10 (Thai photos never in their own test fold):
      acc 53.8%, balanced 47.2%   (Armocromia-only model: 50.0% / 39.2%;
      classical season.py: 28.6% / 33.5%)
  Deep Armocromia TEST (n=910): acc 35.4%, balanced 36.1% (classical 27.4%)

Pure Python (no numpy/sklearn at inference) so it adds nothing to the deploy.
"""
from __future__ import annotations

import math

CLASSES = ("spring", "summer", "autumn", "winter")
FEATURES = ("L", "a", "b", "chroma", "hue_angle", "ita")

_MEAN = [72.32564718, 7.80028695, 13.53637377, 16.55989120, 71.65455382, 55.60200667]
_SCALE = [11.78931903, 7.10714985, 9.13455314, 10.18820568, 49.80534897, 29.51102014]
_COEF = [
    [0.11478846, -0.11210367, -0.18477314, 0.17908559, -0.06917224, -0.08728592],
    [-0.03863198, 0.30070401, -0.16235002, -0.20497457, -0.00131555, 0.39510673],
    [-0.18820163, -0.09632431, 0.70559990, -0.28731661, 0.07154890, -0.04703146],
    [0.11204516, -0.09227602, -0.35847673, 0.31320559, -0.00106111, -0.26078935],
]
_INTERCEPT = [0.05994819, -0.04537355, -0.06150590, 0.04693126]


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
