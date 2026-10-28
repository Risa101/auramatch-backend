"""ISOLATED calibration search interface (CALIBRATION_PROTOCOL.md).

Generates and evaluates CANDIDATE SeasonModel configurations WITHOUT ever
modifying services/color_engine/season.py. Every candidate is a
`dataclasses.replace(DEFAULT_MODEL, **overrides)` copy -- SeasonModel is
already `@dataclass(frozen=True)`, so this is structurally incapable of
mutating DEFAULT_MODEL or any production state; it can only ever produce a
new, separate instance. Every evaluation calls `decide_season()` (imported,
never edited) with that candidate model passed explicitly via its existing
`model=` parameter -- the same public API `scripts/evaluate_personal_color.py`
already uses, not a new code path into the algorithm.

This module NEVER:
  - writes to services/color_engine/season.py or any other production file
  - picks a "winner" and applies it anywhere
  - reads holdout-split rows (see `filter_by_split` below -- the caller is
    responsible for which split's rows it passes in, but this module
    provides the enforcement primitive so that decision is structural, not
    just a matter of remembering to pass the right CSV)

Conceptually:
    Production season.py
            |
            +-- DEFAULT_MODEL (untouched, used for baseline)
            |
            +-- candidate SeasonModel copies (built here, evaluated here,
                never written back)
"""

from __future__ import annotations

import random
import statistics
from dataclasses import fields, replace

from services.color_engine.season import DEFAULT_MODEL, SEASONS, decide_season
from services.eval import metrics

# Whitelist of fields a calibration candidate may override -- derived
# directly from SeasonModel's own dataclass fields (not hand-copied, so it
# can never silently drift out of sync with season.py's actual parameter
# set), used to reject an unknown/misspelled parameter name loudly rather
# than silently ignoring it.
VALID_PARAMETER_FIELDS = frozenset(f.name for f in fields(DEFAULT_MODEL))


class InvalidCandidateError(ValueError):
    """Raised when a candidate override dict names a field SeasonModel
    doesn't have -- caught here, at candidate-construction time, rather
    than silently no-op'ing (a typo'd parameter name must be loud, not
    quietly evaluate the unmodified baseline while claiming to be a
    different candidate)."""


def make_candidate(overrides: dict, *, base_model=DEFAULT_MODEL):
    """Returns a NEW SeasonModel with `overrides` applied on top of
    `base_model` (DEFAULT_MODEL unless a caller explicitly composes
    candidates on top of another candidate). Never mutates `base_model`.
    """
    unknown = set(overrides) - VALID_PARAMETER_FIELDS
    if unknown:
        raise InvalidCandidateError(
            f"unknown SeasonModel parameter(s): {sorted(unknown)} -- valid fields are "
            f"{sorted(VALID_PARAMETER_FIELDS)}"
        )
    return replace(base_model, **overrides)


# --- holdout enforcement (CALIBRATION_PROTOCOL.md "Final Holdout Lock") ----


def filter_by_split(rows, split_assignments: dict, *, splits_allowed) -> list:
    """rows: list of dicts, each with an "image_id" key.
    split_assignments: {image_id: "calibration"|"validation"|"holdout"}.
    splits_allowed: iterable of split names this call is permitted to see.

    Pure, filesystem-free, unit-testable in isolation -- the actual
    "may holdout be read at all right now" POLICY decision (checking
    calibration_status.json) lives in scripts/calibrate_season.py, which
    calls this with splits_allowed=("holdout",) ONLY after verifying
    calibration_status.is_frozen(). This function itself has no opinion on
    policy; it only does the filtering once the caller has decided what's
    allowed, so the filtering logic itself is trivially testable without
    touching the filesystem-backed freeze check.
    """
    allowed = set(splits_allowed)
    return [r for r in rows if split_assignments.get(r["image_id"]) in allowed]


def development_rows(rows, split_assignments: dict) -> list:
    """Convenience wrapper: calibration + validation only, holdout NEVER
    included regardless of caller intent -- this is the function every
    calibration search call should route through.
    """
    return filter_by_split(rows, split_assignments, splits_allowed=("calibration", "validation"))


# --- candidate evaluation ---------------------------------------------------


def _build_season_features(row: dict) -> dict:
    """rows here are the SAME per-image feature dicts
    scripts/evaluate_personal_color.py already builds (L/a/b/hue_angle/
    chroma/ita_category captured once from the real image pipeline) --
    this just re-shapes them into decide_season()'s expected input, no
    recomputation of any color value.
    """
    return {
        "valid": True,
        "skin_lab": [row["L"], row["a"], row["b"]],
        "hue_angle": row["hue_angle"],
        "chroma": row["chroma"],
        "ita_category": row.get("ita_category"),
        "confidence": row.get("confidence"),
        "wb_trustworthy": row.get("wb_trustworthy"),
        "skin_hair_contrast": row.get("skin_hair_contrast"),
        "skin_iris_contrast": row.get("skin_iris_contrast"),
    }


def evaluate_candidate(rows: list, model) -> dict:
    """rows: development-split rows only (caller's responsibility, enforced
    upstream via development_rows()/filter_by_split()) with L/a/b/hue_angle/
    chroma/ita_category/truth already captured, plus "truth" (the ground-
    truth season string). Calls decide_season() -- UNMODIFIED -- once per
    row with `model` passed explicitly.

    Primary metric is BALANCED ACCURACY, per EVALUATION_PROTOCOL.md
    section 7 ("the primary headline number specifically because it
    corrects for imbalance") -- NOT macro F1, which a differently-worded
    calibration brief might suggest; the existing protocol takes
    precedence over this task's own suggested wording, exactly as this
    task instructs when the two disagree.
    """
    results = []
    for row in rows:
        features = _build_season_features(row)
        decision = decide_season(features, model=model)
        predicted = decision.top_season.capitalize() if (decision.valid and decision.top_season) else None
        results.append({"image_id": row["image_id"], "truth": row["truth"], "predicted": predicted})

    truth_dist = {s: sum(1 for r in results if r["truth"] == s) for s in SEASONS_TITLE}
    pred_dist = {s: sum(1 for r in results if r["predicted"] == s) for s in SEASONS_TITLE}

    return {
        "n": len(results),
        "balanced_accuracy": metrics.balanced_accuracy(results, SEASONS_TITLE),  # PRIMARY
        "accuracy": metrics.accuracy(results),
        "macro_f1": metrics.macro_f1(results, SEASONS_TITLE),
        "confusion_matrix": metrics.confusion_matrix(results, SEASONS_TITLE),
        "per_class": {
            s: dict(zip(("precision", "recall", "f1", "support"), metrics.precision_recall_f1(results, s)))
            for s in SEASONS_TITLE
        },
        "truth_distribution": truth_dist,
        "prediction_distribution": pred_dist,
        "coverage": (sum(1 for r in results if r["predicted"] is not None) / len(results)) if results else None,
    }


SEASONS_TITLE = tuple(s.capitalize() for s in SEASONS)  # season.py's SEASONS is lowercase; labels.csv/metrics use Title case


# --- class collapse detection (task section 14) ----------------------------


def detect_class_collapse(candidate_result: dict, *, ratio_threshold: float = 2.0) -> dict:
    """Flags a season as COLLAPSED if its predicted share of the dataset is
    more than `ratio_threshold`x its ground-truth share, or less than
    1/`ratio_threshold`x -- a simple, documented heuristic (not the only
    possible one), never a hidden magic number: `ratio_threshold` is a
    parameter, defaulted here, always visible in the returned dict.
    """
    n = candidate_result["n"]
    flags = {}
    if n == 0:
        return {"ratio_threshold": ratio_threshold, "flags": {s: None for s in SEASONS_TITLE}, "any_collapsed": False}

    for s in SEASONS_TITLE:
        truth_share = candidate_result["truth_distribution"].get(s, 0) / n
        pred_share = candidate_result["prediction_distribution"].get(s, 0) / n
        if truth_share == 0:
            flags[s] = None  # no ground-truth support for this class -- collapse is undefined, not "no collapse"
            continue
        ratio = pred_share / truth_share
        flags[s] = ratio >= ratio_threshold or ratio <= (1.0 / ratio_threshold)

    return {
        "ratio_threshold": ratio_threshold,
        "flags": flags,
        "any_collapsed": any(v for v in flags.values() if v is not None),
    }


# --- baseline vs candidate comparison (task section 13) --------------------


def compare_candidates(baseline_result: dict, candidate_result: dict, *, candidate_overrides: dict) -> dict:
    """Delta report ONLY -- never auto-selects a winner. `complexity` is
    simply the number of parameters the candidate overrides from baseline
    (0 for baseline itself) -- the simplest available proxy for "how much
    did this change," consistent with section 13's "prefer the smallest
    change that explains the evidence" without inventing a fancier
    complexity metric this task never asked for.
    """
    delta_balanced_acc = (
        candidate_result["balanced_accuracy"] - baseline_result["balanced_accuracy"]
        if candidate_result["balanced_accuracy"] is not None and baseline_result["balanced_accuracy"] is not None
        else None
    )
    return {
        "baseline_balanced_accuracy": baseline_result["balanced_accuracy"],
        "candidate_balanced_accuracy": candidate_result["balanced_accuracy"],
        "delta_balanced_accuracy": delta_balanced_acc,
        "complexity_baseline": 0,
        "complexity_candidate": len(candidate_overrides),
        "candidate_overrides": candidate_overrides,
        "recommendation": (
            "REVIEW -- delta not computable (missing balanced_accuracy on one side)" if delta_balanced_acc is None else
            "REVIEW -- candidate changes params but does not clearly improve balanced accuracy; baseline (simpler) "
            "is the safer default absent a specific, evidenced reason to prefer this candidate"
            if delta_balanced_acc <= 0 else
            "REVIEW -- candidate improves balanced accuracy; still requires human review "
            "(class-collapse check, boundary-case inspection) before any approval -- never auto-applied"
        ),
    }


# --- k-fold cross-validation (task section 12) ------------------------------


def k_fold_indices(image_ids: list, k: int, seed: int) -> list:
    """Deterministic fold assignment: sorts ids first (so shuffle input
    order is itself deterministic, matching scripts/assign_splits.py's own
    convention), then a seeded Fisher-Yates shuffle, then contiguous
    slicing into k roughly-equal folds. Returns a list of k lists of
    image_ids (the held-out fold for each of the k runs).
    """
    if k < 2:
        raise ValueError(f"k must be >= 2 for k-fold cross-validation, got {k}")
    ordered = sorted(image_ids)
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    folds = [[] for _ in range(k)]
    for i, image_id in enumerate(shuffled):
        folds[i % k].append(image_id)
    return folds


def cross_validate(rows: list, model, *, k: int, seed: int) -> dict:
    """rows: development-split rows only (caller's responsibility). Runs k
    folds, each fold held out once and evaluated with `evaluate_candidate`
    against the SAME unmodified model -- this measures FOLD-TO-FOLD
    VARIANCE of a given model's performance across different subsets of
    the SAME development data, not a train/tune step (season.py has no
    trainable parameters; there is nothing to "fit" per fold). If the
    dataset is too small for this to mean anything, callers should check
    `n` per fold before trusting the reported variance -- this function
    does not refuse to run on a tiny dataset, it just reports honestly
    small per-fold n alongside the numbers (see scripts/calibrate_season.py
    for the "too small to be meaningful" caller-side check and message).
    """
    ids = [r["image_id"] for r in rows]
    by_id = {r["image_id"]: r for r in rows}
    folds = k_fold_indices(ids, k, seed)

    fold_results = []
    for i, fold_ids in enumerate(folds):
        fold_rows = [by_id[fid] for fid in fold_ids if fid in by_id]
        result = evaluate_candidate(fold_rows, model)
        fold_results.append({"fold": i, "n": result["n"], **{
            k2: v for k2, v in result.items() if k2 in ("balanced_accuracy", "accuracy", "macro_f1", "coverage")
        }})

    balanced_accs = [f["balanced_accuracy"] for f in fold_results if f["balanced_accuracy"] is not None]
    return {
        "k": k, "seed": seed, "folds": fold_results,
        "mean_balanced_accuracy": statistics.fmean(balanced_accs) if balanced_accs else None,
        "variance_balanced_accuracy": statistics.pvariance(balanced_accs) if len(balanced_accs) > 1 else None,
    }


# --- boundary-case analysis (task section 15) -------------------------------


def margin_for_row(row: dict, model) -> dict:
    """Runs decide_season() once (UNMODIFIED) and returns winner/runner-up/
    margin for one row -- the per-row primitive
    scripts/calibrate_season.py's boundary-case report sorts by. Confusion-
    pair feature stats themselves are NOT reimplemented here -- reuse
    services.eval.stratify.confusion_pair_feature_stats directly, which
    already does exactly that (see CALIBRATION_PROTOCOL.md).
    """
    features = _build_season_features(row)
    decision = decide_season(features, model=model)
    if not decision.valid or not decision.ranked:
        return {"image_id": row["image_id"], "truth": row["truth"], "predicted": None, "margin": None}
    ranked = decision.ranked
    top_conf = ranked[0][1]
    second_conf = ranked[1][1] if len(ranked) > 1 else 0.0
    return {
        "image_id": row["image_id"], "truth": row["truth"],
        "predicted": decision.top_season.capitalize(),
        "margin": top_conf - second_conf,
    }
