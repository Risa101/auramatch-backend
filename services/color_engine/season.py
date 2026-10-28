"""Season / tone decision -- color features -> Korean personal-color diagnosis.

Ported verbatim (logic unchanged) from
auramatchgenz/backend/app/color_engine/season.py. Consumes a feature mapping
(dict or object with attribute access -- see ``_get``) and produces a
**ranked, confidence-weighted** personal color diagnosis, never a single hard
label pretending to certainty it doesn't have.

Locked design:
    * Diagnose warm/cool FIRST, then season, then sub-tone. The warm/cool
      axis is weighted most heavily in the season score.
    * Three ordinal axes: warm<->cool (hue), light<->deep (value),
      bright<->mute (chroma [+ optional skin/hair/iris contrast]).
    * Output = 4 seasons (Spring/Summer/Autumn/Winter) + warm/cool, plus an
      8-tone Korean sub-type expansion within the winning season.

Olive-skin guard: olive skin reads warm on the surface (yellow lipid layer)
but has a cool undertone and high contrast -- the classic "Warm Autumn"
misclassification. Evaluates contrast first, then for a warm-surface +
high-contrast + mid-depth candidate, applies a cool correction and flags it.

IMPORTANT -- thresholds are PROVISIONAL, same as in the source project: the
pivots/half-ranges in :class:`SeasonModel` are reasonable priors, NOT
dataset-calibrated truth (this project has no labeled personal-color dataset
-- see the AI accuracy audit's "Data / Ground Truth Findings"). They live in
one dataclass on purpose: calibrating later is a config change, not a
rewrite.

Pure-math (no image/CV deps) -- unit-testable in isolation; see
tests/test_season.py.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping

SEASONS = ("spring", "summer", "autumn", "winter")

_SEASON_TARGETS = {
    "spring": (+1.0, +1.0, +1.0),  # warm,  light, bright
    "summer": (-1.0, +1.0, -1.0),  # cool,  light, mute
    "autumn": (+1.0, -1.0, -1.0),  # warm,  deep,  mute
    "winter": (-1.0, -1.0, +1.0),  # cool,  deep,  bright
}

_SUBTONE_KO = {
    "spring": ("Spring Light", "Spring Bright"),
    "summer": ("Summer Mute", "Summer Light"),
    "autumn": ("Autumn Mute", "Autumn Deep"),
    "winter": ("Winter Bright", "Winter Deep"),
}


@dataclass(frozen=True)
class SeasonModel:
    """Provisional decision constants (calibrate against a labeled dataset
    once one exists -- see the audit's ground-truth findings).

    Every axis score is ``clip((x - pivot) / half_range, -1, 1)``: at ``pivot``
    the axis is neutral (0); at ``pivot ± half_range`` it saturates to ±1.
    """

    hue_pivot: float = 50.0
    hue_half: float = 12.0

    value_pivot: float = 58.0
    value_half: float = 18.0

    chroma_pivot: float = 18.0
    chroma_half: float = 10.0

    contrast_pivot: float = 40.0
    contrast_half: float = 22.0
    contrast_weight: float = 0.5

    w_warm: float = 1.0
    w_value: float = 0.7
    w_chroma: float = 0.5

    temperature: float = 0.45

    olive_warm_min: float = 0.15
    olive_contrast_min: float = 0.30
    olive_cool_shift: float = 0.85
    olive_ita_categories: tuple = ("intermediate", "tan", "brown")

    subtone_pivot: float = 0.0


DEFAULT_MODEL = SeasonModel()


@dataclass
class SeasonDecision:
    """Ranked personal-color diagnosis for one analyzed frame."""

    valid: bool
    reason: str

    warm_cool: str
    warm_cool_score: float | None

    light_deep: str
    light_deep_score: float | None

    bright_mute: str
    bright_mute_score: float | None

    ranked: list = field(default_factory=list)
    top_season: str | None = None
    top_confidence: float | None = None

    sub_tone: str | None = None

    reliability: str = "low"  # "high" | "medium" | "low"
    olive_adjusted: bool = False
    flags: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _invalid(reason: str) -> SeasonDecision:
    return SeasonDecision(
        valid=False,
        reason=reason,
        warm_cool="unknown",
        warm_cool_score=None,
        light_deep="unknown",
        light_deep_score=None,
        bright_mute="unknown",
        bright_mute_score=None,
        reliability="low",
    )


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _axis_label(score: float, pos: str, neg: str, dead: float = 0.12) -> str:
    if score > dead:
        return pos
    if score < -dead:
        return neg
    return "neutral"


def _get(features, key, default=None):
    """Read a field from a features dataclass OR its mapping (dict)."""
    if isinstance(features, Mapping):
        return features.get(key, default)
    return getattr(features, key, default)


def decide_season(features, model: SeasonModel = DEFAULT_MODEL, top_k: int = 4) -> SeasonDecision:
    """Diagnose personal color from a feature bundle (dataclass or dict).

    Required fields (via ``_get``): ``valid``, ``skin_lab`` ([L, a, b]),
    ``hue_angle``, ``chroma``, ``ita_category``. Optional: ``confidence``
    ("high"/"medium"/"low"), ``wb_trustworthy`` (bool),
    ``skin_hair_contrast``/``skin_iris_contrast`` (float or None).

    Returns a :class:`SeasonDecision` (always; check ``.valid``).
    """
    if not _get(features, "valid", False):
        return _invalid(_get(features, "reason", "invalid_features") or "invalid_features")

    skin_lab = _get(features, "skin_lab")
    if not skin_lab:
        return _invalid("no_skin_lab")

    L = float(skin_lab[0])
    hue = _get(features, "hue_angle")
    chroma = _get(features, "chroma")
    if hue is None or chroma is None:
        return _invalid("missing_skin_features")

    warm = _clip((float(hue) - model.hue_pivot) / model.hue_half)
    value = _clip((L - model.value_pivot) / model.value_half)

    bright_chroma = _clip((float(chroma) - model.chroma_pivot) / model.chroma_half)
    contrasts = [c for c in (_get(features, "skin_hair_contrast"), _get(features, "skin_iris_contrast")) if c is not None]
    if contrasts:
        contrast_term = _clip((max(contrasts) - model.contrast_pivot) / model.contrast_half)
        bright = (bright_chroma + model.contrast_weight * contrast_term) / (1.0 + model.contrast_weight)
    else:
        contrast_term = None
        bright = bright_chroma

    flags: list[str] = []

    olive_adjusted = False
    ita_cat = _get(features, "ita_category")
    if (
        warm >= model.olive_warm_min
        and contrast_term is not None
        and contrast_term >= model.olive_contrast_min
        and (ita_cat in model.olive_ita_categories)
    ):
        warm = _clip(warm - model.olive_cool_shift)
        olive_adjusted = True
        flags.append("olive_guard: warm surface + high contrast -> cool-corrected; verify chroma")

    axes = (warm, value, bright)

    weights = (model.w_warm, model.w_value, model.w_chroma)
    dists = {}
    for season, target in _SEASON_TARGETS.items():
        d = math.sqrt(sum(w * (a - t) ** 2 for w, a, t in zip(weights, axes, target)))
        dists[season] = d

    neg = {s: -d / model.temperature for s, d in dists.items()}
    m = max(neg.values())
    exps = {s: math.exp(v - m) for s, v in neg.items()}
    z = sum(exps.values())
    probs = {s: exps[s] / z for s in SEASONS}

    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    ranked = [(s, round(p, 4)) for s, p in ranked[: max(1, top_k)]]
    top_season, top_conf = ranked[0]

    sub_tone = _sub_tone(top_season, value=value, bright=bright, model=model)

    reliability, rflags = _reliability(features, top_conf, ranked, olive_adjusted)
    flags.extend(rflags)

    return SeasonDecision(
        valid=True,
        reason="ok",
        warm_cool=_axis_label(warm, "warm", "cool"),
        warm_cool_score=round(warm, 4),
        light_deep=_axis_label(value, "light", "deep"),
        light_deep_score=round(value, 4),
        bright_mute=_axis_label(bright, "bright", "mute"),
        bright_mute_score=round(bright, 4),
        ranked=ranked,
        top_season=top_season,
        top_confidence=top_conf,
        sub_tone=sub_tone,
        reliability=reliability,
        olive_adjusted=olive_adjusted,
        flags=flags,
    )


def _sub_tone(season: str, *, value: float, bright: float, model: SeasonModel) -> str:
    low_label, high_label = _SUBTONE_KO[season]
    axis = bright if season in ("spring", "summer") else value
    return high_label if axis >= model.subtone_pivot else low_label


def _reliability(features, top_conf, ranked, olive_adjusted):
    """Coarse reliability from upstream confidence, WB trust, and season margin."""
    flags = []
    upstream = _get(features, "confidence", "low")
    wb_ok = bool(_get(features, "wb_trustworthy", False))

    margin = top_conf - (ranked[1][1] if len(ranked) > 1 else 0.0)

    score = {"high": 2, "medium": 1, "low": 0}.get(upstream, 0)
    if not wb_ok:
        score -= 1
        flags.append("wb_untrustworthy: undertone (warm/cool) suspect")
    if margin < 0.10:
        score -= 1
        flags.append(f"low_season_margin: top two within {margin:.2f}")
    if olive_adjusted:
        flags.append("olive_adjusted: confirm with targeted cool-chroma check")

    if score >= 2 and wb_ok:
        reliability = "high"
    elif score >= 1:
        reliability = "medium"
    else:
        reliability = "low"
    return reliability, flags
