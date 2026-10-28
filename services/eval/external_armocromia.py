"""Isolated exploratory-evaluation adapter: external Deep-Armocromia-derived
CIELAB features (via nypoclary/armocromia-classical-ml, MIT license, see
evaluation_dataset/DATASET_CANDIDATES.md Candidate 4) run through
AuraMatch's EXISTING, UNMODIFIED season-decision logic
(services/color_engine/season.py's decide_season).

This module changes NOTHING in services/color_engine/. It imports
decide_season() and color_math.py's existing skin_features()/delta_e_2000()
verbatim and feeds them inputs built from a third-party CSV. See
evaluation_dataset/results/external_armocromia_exploratory.md for the full
methodology writeup, results, and -- critically -- what this experiment
can and cannot validate.

Schema
------
Two column layouts are recognized, both from nypoclary/armocromia-classical-ml
(git commit 63d3338078c27a3b2a89b340e0ad38d8f2c46388):

  * "region30" (features_30_per_region.csv) -- per-region (skin/hair/eye)
    CIELAB statistics. PREFERRED: the "skin" region (source library's
    face-parsing "face"+"nose" mask) is the closer proxy to AuraMatch's own
    landmark-restricted skin mask, and hair/eye columns let the optional
    contrast terms be computed (see build_season_input).
  * "global10" (features_10_merged.csv) -- CIELAB statistics over "all
    non-background facial pixels" (skin+hair+eyes+brows combined, per the
    source repo's own README). WEAKER proxy for AuraMatch's skin-only
    measurement -- supported for completeness/comparison, not preferred.

Neither schema is silently reinterpreted as the other; detect_schema()
raises rather than guessing if a file matches neither exactly.

What this adapter deliberately does NOT synthesize
----------------------------------------------------
`confidence` and `wb_trustworthy` are NEVER set on the built features dict.
decide_season()'s own `_get(features, "confidence", "low")` /
`_get(features, "wb_trustworthy", False)` defaults therefore apply to every
sample here. This is a documented, load-bearing decision, not an oversight:

  * `confidence` in production (services/color_engine/extract.py's
    `_confidence()`) is derived from skin-pixel count, K-means cluster
    spread, and kept-weight-fraction -- none of which exist in this external
    CSV (no K-means was run upstream; only mean/std pixel statistics were
    exported). There is no honest way to reconstruct it.
  * `wb_trustworthy` in production checks whether AuraMatch's OWN
    Shades-of-Gray white-balance step produced a plausible warm-skin Lab
    point. No AuraMatch white-balance step ever ran on these images --
    whatever illumination correction (if any) happened upstream in Deep
    Armocromia's own image pipeline is undocumented and unverified (see
    the exploratory report's Feature Compatibility section). Synthesizing
    a "trustworthy" flag here would inject an unearned assumption.

Direct, measurable consequence (verified while building this adapter, not
guessed): with both inputs absent, decide_season()'s `_reliability()`
ALWAYS returns "low" for every sample, regardless of margin -- because
`wb_ok` is always False, which alone caps `score` below the "medium"
threshold. The `reliability` field in this experiment's output is therefore
constant and CANNOT be used to test "does reliability correlate with
correctness" (task requirement). `margin` (top season probability minus
runner-up) is used as the substitute decision-boundary signal instead --
it is computed purely from decide_season()'s own ranked output and requires
no synthesized input.

skin_hair_contrast / skin_iris_contrast
----------------------------------------
Production's extract.py ALWAYS passes `skin_hair_contrast=None,
skin_iris_contrast=None` -- hair/iris sampling was never wired into the
production extractor (see extract.py's module docstring: "deliberately NOT
ported here"). The FAITHFUL default here (`use_region_contrast=False`)
matches that exactly: contrasts are left None, so season.py's `bright/mute`
axis falls back to chroma alone, identical to what production computes
today for every real analysis.

`use_region_contrast=True` is a clearly-separate, explicitly-labeled
HYPOTHETICAL mode: it computes skin<->hair and skin<->eye CIEDE2000
contrast from this dataset's region means (via color_math.delta_e_2000,
unchanged production code) for rows where that region isn't NaN. It
exercises a season.py code path production's own extractor never
populates today. Never conflate its results with "AuraMatch's current
behavior" -- see the exploratory report's clear separation of the two runs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from services.color_engine import color_math as cm
from services.color_engine.season import DEFAULT_MODEL, decide_season

SEASONS = ["Spring", "Summer", "Autumn", "Winter"]

REQUIRED_REGION30 = ["L_mean_skin", "a_mean_skin", "b_mean_skin", "season", "partition", "celeba"]
REQUIRED_GLOBAL10 = ["L_mean", "a_mean", "b_mean", "season", "partition", "celeba"]


class SchemaError(ValueError):
    """Raised when the input CSV doesn't match a recognized schema, or a
    required field is missing/malformed. Never silently guessed past."""


# --- schema detection --------------------------------------------------


def detect_schema(fieldnames) -> str:
    fs = set(fieldnames or [])
    if set(REQUIRED_REGION30).issubset(fs):
        return "region30"
    if set(REQUIRED_GLOBAL10).issubset(fs):
        return "global10"
    missing_30 = sorted(set(REQUIRED_REGION30) - fs)
    missing_10 = sorted(set(REQUIRED_GLOBAL10) - fs)
    raise SchemaError(
        "Unrecognized CSV schema -- refusing to guess. Expected either the "
        f"'region30' columns (missing: {missing_30}) or the 'global10' "
        f"columns (missing: {missing_10}). This adapter only understands "
        "nypoclary/armocromia-classical-ml's features_30_per_region.csv / "
        "features_10_merged.csv column layout (commit "
        "63d3338078c27a3b2a89b340e0ad38d8f2c46388)."
    )


# --- row model -----------------------------------------------------------


@dataclass
class ExternalSample:
    row_index: int
    sample_id: str
    source_label: str
    source_subtype: Optional[str]
    partition: str
    celeba: bool
    skin_lab: Optional[tuple]
    hair_lab: Optional[tuple]
    eye_lab: Optional[tuple]
    exclude_reason: Optional[str] = None


def _to_float(row, key, *, row_index):
    raw = row.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        raise SchemaError(f"row {row_index}: field {key!r} = {raw!r} is not numeric")


def _to_bool(v) -> bool:
    return str(v).strip().upper() == "TRUE"


def _lab_triplet(row, l_key, a_key, b_key, *, row_index):
    """A blank cell (all three fields empty) returns None -- graceful, since
    a missing measurement is normal (matches labels.csv's own blank-cell
    convention: excluded from the metric, not a fatal error). A field that's
    PRESENT but not parseable as a number (garbage/corruption, not merely
    absent) raises via `_to_float` regardless of which field this is --
    that distinction, not an `optional` flag, is what decides
    raise-vs-exclude here.
    """
    L = _to_float(row, l_key, row_index=row_index)
    a = _to_float(row, a_key, row_index=row_index)
    b = _to_float(row, b_key, row_index=row_index)
    if L is None or a is None or b is None:
        return None
    return (L, a, b)


def _sample_id(row, row_index) -> str:
    return row.get("path_rgb_masked") or f"row_{row_index}"


def _parse_row(row, schema, *, row_index) -> ExternalSample:
    season = (row.get("season") or "").strip()
    partition = (row.get("partition") or "").strip()
    celeba = _to_bool(row.get("celeba"))
    subtype = row.get("season_subtype") or None
    sample_id = _sample_id(row, row_index)

    if season not in SEASONS:
        # Kept (not dropped) for exclusion accounting -- EVALUATION_PROTOCOL.md
        # section 5's "exclusions are recorded, never silent" rule.
        return ExternalSample(
            row_index=row_index, sample_id=sample_id, source_label=season or "MISSING",
            source_subtype=subtype, partition=partition, celeba=celeba,
            skin_lab=None, hair_lab=None, eye_lab=None,
            exclude_reason=f"unsupported_label({season!r})",
        )

    if schema == "region30":
        skin = _lab_triplet(row, "L_mean_skin", "a_mean_skin", "b_mean_skin", row_index=row_index)
        hair = _lab_triplet(row, "L_mean_hair", "a_mean_hair", "b_mean_hair", row_index=row_index)
        eye = _lab_triplet(row, "L_mean_eye", "a_mean_eye", "b_mean_eye", row_index=row_index)
    else:
        skin = _lab_triplet(row, "L_mean", "a_mean", "b_mean", row_index=row_index)
        hair = eye = None

    return ExternalSample(
        row_index=row_index, sample_id=sample_id, source_label=season, source_subtype=subtype,
        partition=partition, celeba=celeba, skin_lab=skin, hair_lab=hair, eye_lab=eye,
        exclude_reason=None if skin is not None else "missing_skin_lab",
    )


def load_rows(path) -> tuple:
    """Loads + validates schema. Returns (schema_name, [ExternalSample]).

    Fails loudly (raises SchemaError) rather than silently guessing at an
    unrecognized layout, malformed numeric field, or empty file.
    """
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SchemaError(f"{path}: empty or unreadable CSV (no header row)")
        schema = detect_schema(reader.fieldnames)
        samples = [_parse_row(row, schema, row_index=i) for i, row in enumerate(reader)]
    return schema, samples


# --- season.py input construction ----------------------------------------


def build_season_input(sample: ExternalSample, *, use_region_contrast: bool) -> Optional[dict]:
    """Builds the exact dict decide_season() expects. See this module's
    docstring for exactly what is and is not synthesized, and why.
    """
    if sample.skin_lab is None:
        return None
    L, a, b = sample.skin_lab
    feats = cm.skin_features([L, a, b])  # SAME call production's extract.py makes on its representative point

    features = {
        "valid": True,
        "skin_lab": [feats.L, feats.a, feats.b],
        "hue_angle": feats.hue_angle,
        "chroma": feats.chroma,
        "ita_category": feats.ita_category,
        "skin_hair_contrast": None,
        "skin_iris_contrast": None,
    }
    if use_region_contrast:
        if sample.hair_lab is not None:
            features["skin_hair_contrast"] = float(cm.delta_e_2000([L, a, b], list(sample.hair_lab)))
        if sample.eye_lab is not None:
            features["skin_iris_contrast"] = float(cm.delta_e_2000([L, a, b], list(sample.eye_lab)))
    return features


# --- evaluation ------------------------------------------------------------


@dataclass
class EvalResult:
    sample_id: str
    source_label: str
    source_subtype: Optional[str]
    partition: str
    celeba: bool
    predicted: Optional[str]
    top_confidence: Optional[float]
    second_confidence: Optional[float]
    margin: Optional[float]
    reliability: Optional[str]
    valid: bool
    reason: str
    L: Optional[float]
    a: Optional[float]
    b: Optional[float]
    hue_angle: Optional[float]
    chroma: Optional[float]
    ita_category: Optional[str]
    used_region_contrast: bool

    def as_row(self) -> dict:
        d = self.__dict__.copy()
        return d


def evaluate_sample(sample: ExternalSample, *, model=DEFAULT_MODEL, use_region_contrast=False) -> EvalResult:
    common = dict(
        sample_id=sample.sample_id, source_label=sample.source_label,
        source_subtype=sample.source_subtype, partition=sample.partition, celeba=sample.celeba,
        used_region_contrast=use_region_contrast,
    )
    if sample.exclude_reason:
        return EvalResult(
            **common, predicted=None, top_confidence=None, second_confidence=None,
            margin=None, reliability=None, valid=False, reason=sample.exclude_reason,
            L=None, a=None, b=None, hue_angle=None, chroma=None, ita_category=None,
        )

    features = build_season_input(sample, use_region_contrast=use_region_contrast)
    decision = decide_season(features, model=model)
    L, a, b = features["skin_lab"]

    if not decision.valid:
        return EvalResult(
            **common, predicted=None, top_confidence=None, second_confidence=None,
            margin=None, reliability=None, valid=False, reason=f"season_py_invalid:{decision.reason}",
            L=L, a=a, b=b, hue_angle=features["hue_angle"], chroma=features["chroma"],
            ita_category=features["ita_category"],
        )

    ranked = decision.ranked
    top_conf = ranked[0][1] if ranked else None
    second_conf = ranked[1][1] if len(ranked) > 1 else None
    margin = (top_conf - second_conf) if (top_conf is not None and second_conf is not None) else None

    return EvalResult(
        **common,
        predicted=decision.top_season.capitalize() if decision.top_season else None,
        top_confidence=top_conf, second_confidence=second_conf, margin=margin,
        reliability=decision.reliability, valid=True, reason="ok",
        L=L, a=a, b=b, hue_angle=features["hue_angle"], chroma=features["chroma"],
        ita_category=features["ita_category"],
    )


def evaluate_all(samples, *, model=DEFAULT_MODEL, use_region_contrast=False) -> list:
    return [evaluate_sample(s, model=model, use_region_contrast=use_region_contrast) for s in samples]
